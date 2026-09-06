# ActionPotential.jl

module ActionPotentialModel

using DifferentialEquations, Optim, Plots, Printf, Statistics
using Sobol, DataFrames, CSV, Distributed
using QuadGK, StaticArrays, BlackBoxOptim
using CUDA, DiffEqGPU
using Logging   # to silence solver dt_min_unstable warnings during fitting
# successful_retcode is reexported by DifferentialEquations (defined in SciMLBase).
# Use it instead of `retcode == :Success`: retcode is now a ReturnCode.T enum,
# so comparing it to the Symbol :Success is ALWAYS false (silently breaks solves).

export ActionPotential, optimize!, display_action_potential,
       extract_ap_features, profile_likelihood_gNa,
       update_model!, get_full_trace_details, create_ap_plot,
       _score_from_trace

# ---------------------------------------------------------------------------
# Rate-function helpers (standalone for type-stability)
# ---------------------------------------------------------------------------

# Nernst equilibrium potential (V); z = valence, concentrations in mM.
nernst(z, cons_out, cons_in) = (8.31446 * 296.15) / (z * 96.485) * log(cons_out / cons_in)

# Numerically-stable softplus approximating max(0, x).
# Avoids derivative discontinuities that destabilise the ODE solver.
function smooth_max_zero(x, k=20.0)
    val = k * x
    if val > 0
        return (val + log1p(exp(-val))) / k
    else
        return log1p(exp(val)) / k
    end
end

# ---------------------------------------------------------------------------
# Hodgkin-Huxley gating-variable rate functions (all voltages in mV,
# rates in ms⁻¹).  Parameter naming convention:
#   _1 = pre-exponential scale, _2 = voltage threshold / shift,
#   _3 = slope (exponential), _4/_5/_6 = additional shape parameters.
# ---------------------------------------------------------------------------
alpha_n(V, p) = p.N_1 * smooth_max_zero(V - p.N_2)
beta_n(V, p)  = exp((V + p.N_7) / p.N_6)

alpha_m(V, p) = p.M_1 * smooth_max_zero(V - p.M_2)
beta_m(V, p)  = exp((V + p.M_7) / p.M_6)

# H_1 is the pre-exponential scale factor for alpha_h (analogous to N_1 for n
# and M_1 for m).  Its presence is required to independently set both infty_h
# and tau_h; without it the two are coupled through the remaining H parameters.
alpha_h(V, p) = p.H_1 * exp((V + p.H_6) / p.H_3)
beta_h(V, p)  = 1.0 / (1.0 + exp((V + p.H_4) / p.H_5))

# Steady-state (infinity) values
infty_n(V, p) = alpha_n(V, p) / (alpha_n(V, p) + beta_n(V, p))
infty_m(V, p) = alpha_m(V, p) / (alpha_m(V, p) + beta_m(V, p))
infty_h(V, p) = alpha_h(V, p) / (alpha_h(V, p) + beta_h(V, p))

# Power-law stimulus that rises from 0 to stim_h over duration stim_d.
# Clamped to 0 for t ≤ 0 — raising a negative base to a Float64 exponent
# (even an integer-valued one like 2.0) causes a DomainError in Julia.
stim_function(t, stim_d, stim_h, stim_dim) =
    (t <= 0.0 || stim_d <= 0.0) ? 0.0 : stim_h * (t / stim_d)^stim_dim

# Analytical integral of stim_function from 0 to T:
#   ∫₀ᵀ stim_h·(t/d)^dim dt  =  stim_h·d/(dim+1)·(T/d)^(dim+1)
# Guards against stim_d ≤ 0: NelderMead's simplex reflection/contraction
# arithmetic can produce negative stim_d even from valid simplex vertices.
stim_integral(T, stim_d, stim_h, stim_dim) =
    (T <= 0.0 || stim_d <= 0.0) ? 0.0 :
    stim_h * stim_d / (stim_dim + 1) * (T / stim_d)^(stim_dim + 1)

# ---------------------------------------------------------------------------
# Hodgkin-Huxley ODE system
# ---------------------------------------------------------------------------
function hodgkin_huxley(u, p, t)
    V, n, m, h = u

    E_K    = nernst(1, 5.4,  143)    # K⁺ Nernst potential (mV)
    E_Na   = nernst(1, 145,  9.6)    # Na⁺ Nernst potential (mV)
    E_Leak = p.RMP                    # Leak reversal = RMP (no net leak at rest)

    I_K    = p.g_K   * n^4     * (V - E_K)
    I_Na   = p.g_Na  * m^3 * h * (V - E_Na)
    I_Leak = p.g_Leak           * (V - E_Leak)

    I_stim = (p.tot_wait < t < p.tot_wait + p.stim_d) ?
             stim_function(t - p.tot_wait, p.stim_d, p.stim_h, p.stim_dim) : 0.0

    dV = I_stim - (I_K + I_Na + I_Leak)
    dn = alpha_n(V, p) * (1 - n) - beta_n(V, p) * n
    dm = alpha_m(V, p) * (1 - m) - beta_m(V, p) * m
    dh = alpha_h(V, p) * (1 - h) - beta_h(V, p) * h

    return @SVector [dV, dn, dm, dh]
end

# ---------------------------------------------------------------------------
# Pure simulation (no side effects)
# ---------------------------------------------------------------------------
function _simulate_trace(params::NamedTuple, time_points::Vector{Float64}, dt::Float64)
    u0    = @SVector [params.RMP,
                      infty_n(params.RMP, params),
                      infty_m(params.RMP, params),
                      infty_h(params.RMP, params)]
    tspan = (time_points[1], time_points[end])
    prob  = ODEProblem(hodgkin_huxley, u0, tspan, params)

    stim_on  = params.tot_wait
    stim_off = params.tot_wait + params.stim_d

    # Force several solver stops ACROSS the stimulus, not just at its edges.
    # The pre-stimulus baseline is flat, so an adaptive solver would otherwise
    # take one large step and under-resolve (or skip) the brief depolarising
    # pulse. Landing on each tstop guarantees ≥n_stim_stops steps through the
    # pulse; the solver remains free to take large steps elsewhere (no global
    # dtmax), so this captures the upstroke without slowing the whole solve.
    n_stim_stops = max(6, ceil(Int, params.stim_d / dt))
    stim_stops   = collect(range(stim_on, stim_off; length=n_stim_stops))

    # reltol/abstol 1e-4: tightening to 1e-5 multiplied every solve's cost (and
    # there are tens of thousands of solves per fit), with no visible change to
    # the AP shape. The forced stimulus tstops already guard the brief pulse.
    # dtmin/maxiters: unstable parameter sets (which the optimiser probes
    # constantly) otherwise make the adaptive solver shrink dt toward machine
    # epsilon — dozens of rejected stiff steps — before giving up. Capping dtmin
    # makes such solves abort almost immediately (returning a failure retcode,
    # handled below as Inf), which is a large speed-up for the fitting loops and
    # also stops the dt-epsilon shrink before the warning-spam threshold.
    sol = with_logger(NullLogger()) do
        solve(prob, Rosenbrock23(),
              saveat = dt, reltol = 1e-4, abstol = 1e-4,
              tstops = stim_stops, dtmin = 1e-6, maxiters = 50_000,
              force_dtmin = false)
    end

    if !successful_retcode(sol) || length(sol.u) != length(time_points)
        return fill(Inf, length(time_points))
    end
    return [v[1] for v in sol.u]
end

# ---------------------------------------------------------------------------
# Scoring functions (pure, no side effects)
# ---------------------------------------------------------------------------

# The three scoring-window index masks (quiescence / peak / post-peak) depend
# only on tot_wait, time_points, dt, stabil_time and trace_data_len — ALL fixed
# for the whole duration of one optimisation call (tot_wait is set once by
# find_foot! and is never itself an optimised parameter). Recomputing these
# BitVectors from scratch inside the objective — as the old _score_from_trace
# did — allocates 3 fresh length-~1500 arrays on EVERY one of the tens of
# thousands of CPU evals, or every one of up to 500_000 GPU trajectories in
# gpu_grid_search!'s host-side output_func. Precomputing them once per
# optimisation call (see global_optimize/optimize_model/gpu_grid_search!) and
# reusing them here removes that redundant allocation without changing the
# score's value at all.
function _score_masks(tot_wait::Float64, time_points::Vector{Float64}, dt::Float64,
                       stabil_time::Float64, trace_data_len::Int)
    settle     = 0.5   # ms: ignore only the very first relaxation step from V_init.
                       # (Larger values let a spontaneous AP fire in the unscored
                       #  window before the stimulus — a "2nd spike" the optimiser
                       #  never sees; 0.5 ms forces rest from the start.)
    pk_dur_est = 5.0
    peak_start = tot_wait
    peak_end   = peak_start + pk_dur_est
    trace_end  = stabil_time + trace_data_len * dt

    # QUIESCENCE regions: before the stimulus and after the AP the model must sit
    # at rest (experimental_trace_padded = RMP there). Scoring them penalises
    # spontaneous / oscillatory firing, forcing the STIMULUS to drive the single
    # AP instead of letting a chance intrinsic oscillation land in the window
    # (which produced the spurious extra spikes seen before this term was added).
    idx_quiet     = ((time_points .>= settle)   .& (time_points .<  peak_start)) .|
                    ((time_points .>  trace_end) .& (time_points .<= time_points[end]))
    idx_peak      = (time_points .>= peak_start) .& (time_points .<= peak_end)
    idx_post_peak = (time_points .>  peak_end)   .& (time_points .<= trace_end)
    return (idx_quiet = idx_quiet, idx_peak = idx_peak, idx_post_peak = idx_post_peak)
end

# Core scoring given an already-computed voltage vector and precomputed masks
# (see _score_masks). Separated from _calculate_score so GPU grid search can
# use the GPU-solved trajectory directly without re-solving on CPU.
function _score_from_masks(simulated_Vs::Vector{Float64}, masks::NamedTuple,
                            experimental_trace_padded::Vector{Float64})
    isempty(simulated_Vs) && return Inf
    any(isinf, simulated_Vs) && return Inf
    score_val = (
        sum((simulated_Vs[masks.idx_quiet]     .- experimental_trace_padded[masks.idx_quiet]).^2)     +
        5.0 * sum((simulated_Vs[masks.idx_peak] .- experimental_trace_padded[masks.idx_peak]).^2)      +
        sum((simulated_Vs[masks.idx_post_peak] .- experimental_trace_padded[masks.idx_post_peak]).^2)
    )
    return isfinite(score_val) ? score_val : Inf
end

# Convenience wrapper computing masks fresh each call — kept for callers that
# score only occasionally (analysis scripts) where recomputation cost is
# irrelevant. Hot optimisation loops should precompute masks once via
# _score_masks and call _score_from_masks directly (see global_optimize,
# optimize_model, profile_likelihood_gNa, optimize_split!, gpu_grid_search!).
function _score_from_trace(simulated_Vs::Vector{Float64},
                            tot_wait::Float64,
                            time_points::Vector{Float64},
                            dt::Float64,
                            experimental_trace_padded::Vector{Float64},
                            stabil_time::Float64,
                            trace_data_len::Int)
    isempty(simulated_Vs) && return Inf
    any(isinf, simulated_Vs) && return Inf
    masks = _score_masks(tot_wait, time_points, dt, stabil_time, trace_data_len)
    return _score_from_masks(simulated_Vs, masks, experimental_trace_padded)
end

# Simulate and score in one call (used by CPU optimisers).
function _calculate_score(params::NamedTuple,
                           time_points::Vector{Float64},
                           dt::Float64,
                           experimental_trace_padded::Vector{Float64},
                           stabil_time::Float64,
                           trace_data_len::Int)
    simulated_Vs = _simulate_trace(params, time_points, dt)
    isinf(simulated_Vs[1]) && return Inf
    return _score_from_trace(simulated_Vs, params.tot_wait, time_points, dt,
                              experimental_trace_padded, stabil_time, trace_data_len)
end

# Fast path for hot optimisation loops: takes masks precomputed once via
# _score_masks instead of recomputing them from tot_wait/stabil_time/
# trace_data_len on every evaluation (see _score_masks for why this is safe).
function _calculate_score(params::NamedTuple,
                           time_points::Vector{Float64},
                           dt::Float64,
                           experimental_trace_padded::Vector{Float64},
                           masks::NamedTuple)
    simulated_Vs = _simulate_trace(params, time_points, dt)
    isinf(simulated_Vs[1]) && return Inf
    return _score_from_masks(simulated_Vs, masks, experimental_trace_padded)
end

# ---------------------------------------------------------------------------
# Bounds-penalty helper
# ---------------------------------------------------------------------------
function _bounds_penalty(p_vec, param_names::Tuple, bounds::NamedTuple)
    penalty = 0.0
    for (k, v) in zip(param_names, p_vec)
        if haskey(bounds, k)
            lb, ub = bounds[k]
            if v < lb
                penalty += 1e6 * (lb - v)^2
            elseif v > ub
                penalty += 1e6 * (v - ub)^2
            end
        end
    end
    return penalty
end

# ---------------------------------------------------------------------------
# Main ActionPotential struct
# ---------------------------------------------------------------------------
mutable struct ActionPotential
    name::String
    params::NamedTuple
    trace_data::Vector{Float64}
    time_points::Vector{Float64}
    dt::Float64
    stim_d::Float64
    stim_h::Float64
    stim_dim::Float64
    stabil_time::Float64
    tot_wait::Float64
    V_init::Float64
    Vs::Vector{Float64}
    AP_val::Vector{Float64}
end

# ---------------------------------------------------------------------------
# Robust resting-membrane-potential estimate.
#
# These recordings have very little pre-foot baseline — the AP upstroke can
# begin <1 ms into the trace — so the naïve mean(first 2 ms) is contaminated by
# the upstroke and over-estimates RMP by ~20 mV, shifting the entire model.
# Instead: take a rough rest from a low percentile (robust because the AP sits
# ABOVE rest), detect the foot, then average the genuinely pre-foot samples.
function estimate_rmp(trace::AbstractVector{<:Real}, dt::Real)
    isempty(trace) && return 0.0
    n2    = min(round(Int, 2.0 / dt), length(trace))
    rough = quantile(collect(Float64, @view trace[1:n2]), 0.10)
    fi    = detect_foot_index(collect(Float64, trace), Float64(rough))
    return fi > 1 ? Float64(mean(@view trace[1:fi-1])) : Float64(rough)
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
function ActionPotential(p_in::Union{NamedTuple, Dict}, trace, time; name="Nameless")
    initial_params = typeof(p_in) <: Dict ? NamedTuple(p_in) : p_in
    dt          = round(time[2] - time[1], digits=3)
    sim_time    = 30.0
    stabil_time = 10.0
    t_sim       = 0:dt:sim_time

    # Robust pre-foot baseline (see estimate_rmp): mean(first 2 ms) is corrupted
    # by the early upstroke in these short-baseline traces.
    calculated_RMP = estimate_rmp(trace, dt)

    params    = merge(initial_params, (RMP = calculated_RMP,))
    V_init    = params.RMP
    start_idx = round(Int, stabil_time / dt) + 1
    end_idx   = start_idx + length(trace) - 1
    AP_val    = fill(params.RMP, length(t_sim))
    AP_val[start_idx:min(end_idx, length(AP_val))] = trace[1:min(length(trace), length(AP_val) - start_idx + 1)]

    ap = ActionPotential(
        name, params, trace, collect(t_sim), dt,
        0.64, 54.874, 2.0,      # stim_d, stim_h, stim_dim defaults
        stabil_time, stabil_time, # stabil_time, tot_wait
        V_init,
        zeros(length(t_sim)), AP_val
    )
    update_model!(ap, ap.params)
    return ap
end

# ---------------------------------------------------------------------------
# Detailed state extraction (for decomposition plots)
# ---------------------------------------------------------------------------
function get_full_trace_details(ap::ActionPotential)
    p  = merge(ap.params, (stim_d=ap.stim_d, stim_h=ap.stim_h,
                            stim_dim=ap.stim_dim, tot_wait=ap.tot_wait))
    u0 = @SVector [ap.V_init,
                   infty_n(ap.V_init, p),
                   infty_m(ap.V_init, p),
                   infty_h(ap.V_init, p)]
    tspan = (ap.time_points[1], ap.time_points[end])
    prob  = ODEProblem(hodgkin_huxley, u0, tspan, p)
    # NOTE: no `verbose=false` here — OrdinaryDiffEq no longer accepts a Bool for
    # `verbose` (it now takes a DEVerbosity object; passing Bool raises
    # ArgumentError). This function is called per-group for decomposition plots,
    # not in a hot loop, so default solver verbosity is harmless; the hot path
    # (_simulate_trace) already suppresses logging via `with_logger(NullLogger())`.
    sol   = solve(prob, Rosenbrock23(), saveat=ap.dt,
                  tstops=[p.tot_wait, p.tot_wait + p.stim_d])

    V  = [u[1] for u in sol.u]
    n  = [u[2] for u in sol.u]
    m  = [u[3] for u in sol.u]
    h  = [u[4] for u in sol.u]

    E_K  = nernst(1, 5.4, 143)
    E_Na = nernst(1, 145, 9.6)
    IKs    = p.g_K   .* (n.^4)       .* (V .- E_K)
    INas   = p.g_Na  .* (m.^3) .* h  .* (V .- E_Na)
    ILeaks = p.g_Leak             .* (V .- p.RMP)
    Istims = [p.tot_wait < t < p.tot_wait+p.stim_d ?
              stim_function(t-p.tot_wait, p.stim_d, p.stim_h, p.stim_dim) : 0.0
              for t in ap.time_points]
    gKs    = p.g_K  .* (n.^4)
    gNas   = p.g_Na .* (m.^3) .* h

    return (t=ap.time_points, V=V, n=n, m=m, h=h,
            IKs=IKs, INas=INas, ILeaks=ILeaks, Istims=Istims,
            gKs=gKs, gNas=gNas)
end

# ---------------------------------------------------------------------------
# AP feature extraction
#
# Features are computed from the *simulated* trace (ap.Vs) by default, or
# from the experimental trace (ap.AP_val) when use_experimental=true.
# All times are relative to the stimulus onset (ap.tot_wait).
#
# Key features for group discrimination and g_Na validation:
#   max_dvdt   — maximum upstroke velocity (mV/ms); directly proportional to
#                peak Na current: I_Na_peak ≈ C_m * max(dV/dt)
#   V_peak     — peak depolarisation (mV)
#   V_threshold — membrane voltage at max dV/dt (threshold for AP firing)
#   APD50      — action potential duration at 50% repolarisation (ms)
#   AHP_depth  — afterhyperpolarisation below RMP (mV; negative = below rest)
# ---------------------------------------------------------------------------
function extract_ap_features(ap::ActionPotential; use_experimental::Bool=false)
    V_full = use_experimental ? ap.AP_val : ap.Vs
    t      = ap.time_points
    dt     = ap.dt

    # Index range: from stimulus onset to end of experimental trace
    stim_idx  = round(Int, ap.tot_wait  / dt) + 1
    trace_end = round(Int, (ap.stabil_time + length(ap.trace_data) * dt) / dt) + 1
    trace_end = min(trace_end, length(V_full))

    if stim_idx >= trace_end
        @warn "extract_ap_features: stimulus onset index ≥ trace end; returning nothing"
        return nothing
    end

    V_seg = V_full[stim_idx:trace_end]
    t_seg = t[stim_idx:trace_end]
    rmp   = ap.params.RMP

    # Peak
    peak_idx   = argmax(V_seg)
    V_peak     = V_seg[peak_idx]
    t_peak     = t_seg[peak_idx] - t_seg[1]
    AP_amplitude = V_peak - rmp

    AP_amplitude <= 0 && return nothing  # no AP fired

    # Maximum upstroke dV/dt and threshold voltage
    dVdt           = diff(V_seg) ./ dt
    max_dvdt_idx   = argmax(dVdt)
    max_dvdt       = dVdt[max_dvdt_idx]
    t_threshold    = t_seg[max_dvdt_idx] - t_seg[1]
    V_threshold    = V_seg[max_dvdt_idx]

    # APD50: duration at half-amplitude above RMP
    V_half = rmp + AP_amplitude / 2.0
    rising_cross = findfirst(V_seg .>= V_half)
    # Search for falling crossing only after the peak
    fall_offset  = findfirst(@view(V_seg[peak_idx:end]) .<= V_half)
    APD50 = (!isnothing(rising_cross) && !isnothing(fall_offset)) ?
            (peak_idx + fall_offset - 1 - rising_cross) * dt : NaN

    # Afterhyperpolarisation (AHP): minimum voltage after peak, relative to RMP
    AHP_depth = peak_idx < length(V_seg) ?
                minimum(V_seg[peak_idx:end]) - rmp : NaN

    return (
        RMP          = rmp,
        V_peak       = V_peak,
        AP_amplitude = AP_amplitude,
        t_peak_ms    = t_peak,
        V_threshold  = V_threshold,
        t_threshold_ms = t_threshold,
        max_dvdt     = max_dvdt,
        APD50        = APD50,
        AHP_depth    = AHP_depth
    )
end

# ---------------------------------------------------------------------------
# Profile likelihood for g_Na
#
# Fixes g_Na at n_points values spanning gNa_range × current estimate, and
# minimises the score over all other optimised parameters at each fixed value.
# The resulting score-vs-g_Na curve is the profile likelihood.
#
# A narrow, well-defined minimum confirms g_Na is identifiable from the trace.
# A flat profile indicates g_Na cannot be reliably estimated.
# ---------------------------------------------------------------------------
function profile_likelihood_gNa(ap::ActionPotential, opt_param_names::Tuple;
                                  n_points::Int  = 25,
                                  gNa_range      = (0.3, 3.0),
                                  bounds::Union{NamedTuple,Nothing} = nothing)
    println("Computing profile likelihood for g_Na ($n_points points)...")
    current_gNa = ap.params.g_Na
    gNa_values  = collect(LinRange(current_gNa * gNa_range[1],
                                   current_gNa * gNa_range[2], n_points))

    remaining = Tuple(p for p in opt_param_names if p != :g_Na)
    static_data = (
        time_points    = ap.time_points,
        dt             = ap.dt,
        experimental_trace = ap.AP_val,
        stim_d         = ap.stim_d,
        stim_h         = ap.stim_h,
        stim_dim       = ap.stim_dim,
        tot_wait       = ap.tot_wait,
        stabil_time    = ap.stabil_time,
        trace_data_len = length(ap.trace_data)
    )
    # tot_wait/stabil_time/trace_data_len are fixed across every evaluation at
    # every g_Na grid point, so the scoring-window masks (see _score_masks) can
    # be computed once and shared by all threads instead of rebuilt on every
    # Nelder-Mead evaluation.
    masks = _score_masks(static_data.tot_wait, static_data.time_points, static_data.dt,
                         static_data.stabil_time, static_data.trace_data_len)

    # Each g_Na value is an independent optimisation — run in parallel threads.
    profile_scores = Vector{Float64}(undef, n_points)
    Threads.@threads for i in 1:n_points
        gNa_fixed = gNa_values[i]
        fixed_p   = merge(ap.params, (g_Na = gNa_fixed,))
        function obj(p_vec)
            iter_p = merge(fixed_p, static_data, (; zip(remaining, p_vec)...))
            score  = _calculate_score(iter_p, static_data.time_points, static_data.dt,
                                      static_data.experimental_trace, masks)
            if !isnothing(bounds)
                score += _bounds_penalty(p_vec, remaining, bounds)
            end
            return score
        end
        init_vec = [fixed_p[k] for k in remaining]
        res      = optimize(obj, init_vec, NelderMead(),
                            Optim.Options(iterations=2000, f_reltol=1e-8))
        profile_scores[i] = Optim.minimum(res)
        @printf("  g_Na = %6.1f mS/cm²: score = %.4g\n", gNa_fixed, profile_scores[i])
    end

    best_idx = argmin(profile_scores)
    return (
        gNa_values   = gNa_values,
        scores       = profile_scores,
        optimal_gNa  = gNa_values[best_idx],
        optimal_score = profile_scores[best_idx]
    )
end

# ---------------------------------------------------------------------------
# Model update (resimulate with current parameters)
# ---------------------------------------------------------------------------
function update_model!(ap::ActionPotential, params::NamedTuple)
    ap.params = params
    sim_params = merge(params, (stim_d=ap.stim_d, stim_h=ap.stim_h,
                                stim_dim=ap.stim_dim, tot_wait=ap.tot_wait))
    ap.Vs = _simulate_trace(sim_params, ap.time_points, ap.dt)
end

# ---------------------------------------------------------------------------
# Foot-finding: constant-charge method
# ---------------------------------------------------------------------------
#
# The stimulus must REPRODUCE the foot: its cumulative charge ∫I dt (the voltage
# it produces by charging the membrane, ≈ V because channel currents are small on
# the foot) should trace the experimental foot's curvature, with the onset near
# the start of the trace — so the stimulus *causes* the initial depolarisation
# rather than being a late impulse the channels race ahead of. We therefore fit
# the onset t_0, duration d, and shape exponent dim together so that
#   RMP + ∫₀ᵗ I_stim   ≈   experimental trace   over [tot_wait, tot_wait+d],
# under a fixed total charge (the constant-charge assumption: same injected
# charge per trace, variable electrode coupling). This is the original R
# `find_foot` objective. The stimulus is fitted FIRST; the channel parameters are
# then optimised to match the rest of the AP given this stimulus.
#
# detect_foot_index is still used by estimate_rmp (robust pre-foot baseline).
function detect_foot_index(trace::Vector{Float64}, RMP::Float64; frac::Float64=0.05)
    isempty(trace) && return 1
    peak_idx = argmax(trace)
    peak_idx <= 1 && return 1
    peak_v   = trace[peak_idx]
    thr      = RMP + frac * (peak_v - RMP)
    # Walk BACK from the peak to the last sample at/below threshold = the foot.
    # Anchoring to the peak (rather than scanning forward from t=0) makes this
    # robust to sub-threshold baseline noise that would otherwise trip an early
    # forward-scan crossing.
    fi = peak_idx
    @inbounds while fi > 1 && trace[fi-1] > thr
        fi -= 1
    end
    return fi
end

function find_foot!(ap::ActionPotential; num_fits=10)
    println("\n--- Searching for AP foot (constant-charge method) ---")
    init_p         = (d=ap.stim_d, h=ap.stim_h, dim=ap.stim_dim)
    const_integral = (init_p.h * init_p.d) / (init_p.dim + 1)   # fixed total charge (R's stim_A)
    peak_approx    = 2.0

    # Fit onset t_0, duration d, and shape dim together so the cumulative-charge
    # curve (RMP + ∫I) traces the experimental foot. The onset is free to sit at
    # the start of the trace; the stimulus thus drives the initial depolarisation.
    function objective(foot_params)
        t_0, stim_d, stim_dim = foot_params
        if t_0 < 0.0 || t_0 > peak_approx || stim_d <= 0.0 || stim_d > peak_approx || stim_dim < 1.0
            return Inf
        end
        stim_h_new = const_integral * (stim_dim + 1) / stim_d
        (!isfinite(stim_h_new) || stim_h_new < 0) && return Inf
        tot_wait  = ap.stabil_time + t_0
        start_idx = round(Int, tot_wait / ap.dt) + 1
        end_idx   = start_idx + round(Int, stim_d / ap.dt)
        end_idx > length(ap.AP_val) && return Inf

        exp_window = @view ap.AP_val[start_idx:end_idx]
        n = length(exp_window)
        acc = 0.0
        for k in 1:n
            model = stim_integral(k * ap.dt, stim_d, stim_h_new, stim_dim) + ap.params.RMP
            denom = abs(exp_window[k]) > 1e-6 ? exp_window[k] : 1e-6
            acc  += ((model - exp_window[k]) / denom)^2
        end
        return stim_h_new * acc / n        # R objective: stim_h * mean(relative error²)
    end

    results = Vector{Any}(undef, num_fits)
    for i in 1:num_fits
        t0_guess   = i / num_fits          # scan onset t_0 over (0, 1] ms (R's starts)
        results[i] = optimize(objective, [t0_guess, init_p.d, init_p.dim], NelderMead())
    end
    best_result = results[argmin(Optim.minimum.(results))]

    t_0, stim_d, stim_dim = Optim.minimizer(best_result)
    ap.tot_wait = ap.stabil_time + t_0
    ap.stim_d   = stim_d
    ap.stim_dim = stim_dim
    ap.stim_h   = const_integral * (stim_dim + 1) / stim_d
    @printf("Foot: onset=%.3f ms (t_0=%.3f), d=%.3f ms, dim=%.2f, h=%.2f\n",
            ap.tot_wait, t_0, ap.stim_d, ap.stim_dim, ap.stim_h)
    update_model!(ap, ap.params)
end

# ---------------------------------------------------------------------------
# Local refinement (NelderMead + optional bounds penalty)
# ---------------------------------------------------------------------------
function optimize_model(ap::ActionPotential, opt_param_names::Tuple;
                         bounds::Union{NamedTuple,Nothing} = nothing,
                         ref::Union{NamedTuple,Nothing} = nothing)
    println("Starting local refinement (NelderMead)...")
    initial_params = ap.params
    static_data = (
        time_points    = ap.time_points,
        dt             = ap.dt,
        experimental_trace = ap.AP_val,
        stim_d         = ap.stim_d,
        stim_h         = ap.stim_h,
        stim_dim       = ap.stim_dim,
        tot_wait       = ap.tot_wait,
        stabil_time    = ap.stabil_time,
        trace_data_len = length(ap.trace_data)
    )
    # tot_wait/stabil_time/trace_data_len never change across the ~5000
    # Nelder-Mead evaluations below, so precompute the scoring masks once
    # (see _score_masks) instead of rebuilding them on every eval.
    masks = _score_masks(static_data.tot_wait, static_data.time_points, static_data.dt,
                         static_data.stabil_time, static_data.trace_data_len)

    # Parameter scaling (equivalent to R's optim parscale): optimise in
    # normalised space x = p / |p₀| so every parameter steps by the same
    # RELATIVE amount. Without this, NelderMead's simplex (which has a ~0.025
    # absolute floor) cannot meaningfully move the smallest parameters
    # (e.g. N_1 ≈ 0.004) while taking huge strides on the largest (g_Na ≈ 120).
    init_vec = [initial_params[k] for k in opt_param_names]
    scale    = [v == 0 ? 1.0 : abs(v) for v in init_vec]

    function objective(x)
        p_vec  = x .* scale
        iter_p = merge(initial_params, static_data, (; zip(opt_param_names, p_vec)...))
        score  = _calculate_score(iter_p, static_data.time_points, static_data.dt,
                                  static_data.experimental_trace, masks)
        if !isnothing(bounds)
            score += _bounds_penalty(p_vec, opt_param_names, bounds)
        end
        if !isnothing(ref)
            score += _ref_penalty(p_vec, opt_param_names, ref)
        end
        return score
    end

    x0     = init_vec ./ scale
    result = optimize(objective, x0, NelderMead(),
                      Optim.Options(iterations=5000, f_reltol=1e-9))

    final_vec    = Optim.minimizer(result) .* scale
    final_subset = (; zip(opt_param_names, final_vec)...)
    full_params  = merge(initial_params, final_subset)

    println("Local refinement complete. Best score: ", Optim.minimum(result))
    return Dict("par" => full_params, "value" => Optim.minimum(result),
                "convergence" => Optim.converged(result))
end

# ---------------------------------------------------------------------------
# Coordinate-descent split optimisation (R's optimize_split)
#
# Fits the Na parameters against the UPSTROKE window and the K parameters against
# the BASELINE + REPOLARISATION-TAIL window, alternating. Fitting each conductance
# against the phase it dominates avoids the "no-fire" local optima that a single
# joint search falls into (especially for deep-RMP traces where the constant-
# charge stimulus lands further from threshold), making per-trace fits reliable.
# ---------------------------------------------------------------------------
function _windowed_sse(simVs::Vector{Float64}, exp_padded::Vector{Float64}, mask::BitVector)
    (isempty(simVs) || any(isinf, simVs)) && return Inf
    s = 0.0
    @inbounds for i in eachindex(simVs)
        mask[i] && (s += (simVs[i] - exp_padded[i])^2)
    end
    return isfinite(s) ? s : Inf
end

# Soft penalty keeping each parameter within [lo, hi]× of a reference (par_0)
# value — R's optimize_split `check_penalty`. Without it the coordinate descent
# runs parameters to the bounds into degenerate, non-firing regimes (high g_Na
# cancelled by an inactivated h-gate, g_K pinned at its ceiling, etc.).
function _ref_penalty(p_vec, free::Tuple, ref::NamedTuple; lo=0.2, hi=3.0)
    pen = 0.0
    @inbounds for (k, v) in zip(free, p_vec)
        r = ref[k]
        r == 0.0 && continue
        ratio = v / r                       # same-sign params ⇒ ratio > 0
        ratio < lo && (pen += 1e6 * (lo - ratio)^2)
        ratio > hi && (pen += 1e6 * (ratio - hi)^2)
    end
    return pen
end

function optimize_split!(ap::ActionPotential;
                         bounds::Union{NamedTuple,Nothing} = nothing,
                         n_cycles::Int = 6,
                         opt_Na = (:M_1, :M_2, :M_6, :M_7, :H_1, :H_3, :H_4, :H_5, :H_6, :g_Na),
                         opt_K  = (:N_1, :N_2, :N_6, :N_7, :g_K))
    println("Coordinate-descent split optimisation ($n_cycles cycles)...")
    tp        = ap.time_points
    dt        = ap.dt
    tw        = ap.tot_wait
    settle    = 2.0
    peak_dur  = 2.0
    exp_pad   = ap.AP_val
    static    = (stim_d=ap.stim_d, stim_h=ap.stim_h, stim_dim=ap.stim_dim, tot_wait=tw)
    ref       = ap.params                          # par_0 anchor for check_penalty

    # Na dominates the upstroke + peak; K dominates the pre-stimulus baseline and
    # the repolarisation tail (which also enforces return to a quiescent rest).
    mask_Na = (tp .>= tw) .& (tp .<= tw + peak_dur)
    mask_K  = ((tp .>= settle) .& (tp .< tw)) .|
              ((tp .>  tw + peak_dur) .& (tp .<= tp[end]))

    function fit_subset(free::Tuple, mask::BitVector, cur::NamedTuple)
        init  = [cur[k] for k in free]
        scale = [v == 0 ? 1.0 : abs(v) for v in init]
        function obj(x)
            p  = x .* scale
            ip = merge(cur, static, (; zip(free, p)...))
            sv = _simulate_trace(ip, tp, dt)
            isinf(sv[1]) && return Inf
            s  = _windowed_sse(sv, exp_pad, mask) + _ref_penalty(p, free, ref)
            isnothing(bounds) || (s += _bounds_penalty(p, free, bounds))
            return s
        end
        r = optimize(obj, init ./ scale, NelderMead(),
                     Optim.Options(iterations=1000, f_reltol=1e-8))
        return merge(cur, (; zip(free, Optim.minimizer(r) .* scale)...))
    end

    p = ap.params
    p = fit_subset(opt_K, mask_K, p)                # establish a K baseline first
    for _ in 1:n_cycles
        p = fit_subset(opt_Na, mask_Na, p)          # Na against the upstroke
        p = fit_subset(opt_K,  mask_K,  p)          # K against baseline + tail
    end
    update_model!(ap, p)

    # Final joint refinement on the FULL quiescence-aware objective, still
    # anchored to par_0 so the fit cannot drift into a degenerate optimum.
    all_free = Tuple(vcat(collect(opt_Na), collect(opt_K)))
    init  = [p[k] for k in all_free]
    scale = [v == 0 ? 1.0 : abs(v) for v in init]
    sd = (time_points=tp, dt=dt, experimental_trace=exp_pad,
          stim_d=ap.stim_d, stim_h=ap.stim_h, stim_dim=ap.stim_dim, tot_wait=tw,
          stabil_time=ap.stabil_time, trace_data_len=length(ap.trace_data))
    fobj_masks = _score_masks(tw, tp, dt, ap.stabil_time, length(ap.trace_data))
    function fobj(x)
        pv = x .* scale
        ip = merge(p, sd, (; zip(all_free, pv)...))
        s  = _calculate_score(ip, tp, dt, exp_pad, fobj_masks) +
             _ref_penalty(pv, all_free, ref)
        isnothing(bounds) || (s += _bounds_penalty(pv, all_free, bounds))
        return s
    end
    fr = optimize(fobj, init ./ scale, NelderMead(),
                  Optim.Options(iterations=3000, f_reltol=1e-9))
    final_p = merge(p, (; zip(all_free, Optim.minimizer(fr) .* scale)...))
    update_model!(ap, final_p)
    return Dict("par" => final_p, "value" => Optim.minimum(fr),
                "convergence" => Optim.converged(fr))
end

# ---------------------------------------------------------------------------
# Global search (BlackBoxOptim)
# ---------------------------------------------------------------------------
function global_optimize(ap::ActionPotential, opt_param_names::Tuple;
                          max_evals=2500, range=0.5,
                          bounds::Union{NamedTuple,Nothing} = nothing)
    println("Starting global search (BlackBoxOptim, max_evals=$max_evals)...")
    initial_params = ap.params
    static_data = (
        time_points    = ap.time_points,
        dt             = ap.dt,
        experimental_trace = ap.AP_val,
        stim_d         = ap.stim_d,
        stim_h         = ap.stim_h,
        stim_dim       = ap.stim_dim,
        tot_wait       = ap.tot_wait,
        stabil_time    = ap.stabil_time,
        trace_data_len = length(ap.trace_data)
    )
    # BlackBoxOptim's max_evals evaluations all share the same scoring window
    # (see _score_masks) — compute it once rather than per evaluation.
    masks = _score_masks(static_data.tot_wait, static_data.time_points, static_data.dt,
                         static_data.stabil_time, static_data.trace_data_len)

    function objective(p_vec)
        iter_p = merge(initial_params, static_data, (; zip(opt_param_names, p_vec)...))
        return _calculate_score(iter_p, static_data.time_points, static_data.dt,
                                static_data.experimental_trace, masks)
    end

    # Use physiological bounds when supplied; otherwise ±range% of initial value.
    search_range = Tuple{Float64, Float64}[]
    for name in opt_param_names
        if !isnothing(bounds) && haskey(bounds, name)
            push!(search_range, bounds[name])
        else
            val = initial_params[name]
            lb  = val > 0 ? val * (1 - range) : val * (1 + range)
            ub  = val > 0 ? val * (1 + range) : val * (1 - range)
            lb > ub && ((lb, ub) = (ub, lb))
            push!(search_range, (lb, ub))
        end
    end

    result    = bboptimize(objective;
                           SearchRange  = search_range,
                           NumDimensions = length(opt_param_names),
                           MaxFuncEvals = max_evals,
                           TraceMode    = :silent)
    final_vec = best_candidate(result)
    full_params = merge(initial_params, (; zip(opt_param_names, final_vec)...))
    println("Global search complete. Best score: ", best_fitness(result))
    return Dict("par" => full_params, "value" => best_fitness(result))
end

# ---------------------------------------------------------------------------
# Full two-stage optimisation pipeline
#
# use_gpu=true  — replace BlackBoxOptim global search with GPU grid search.
#                 Requires CUDA.jl and a CUDA-capable GPU.  Pass
#                 num_trajectories to control the GPU search budget
#                 (default 100_000; use 500_000+ on a server GPU).
# use_gpu=false — standard CPU-only pipeline (BlackBoxOptim + NelderMead).
# ---------------------------------------------------------------------------
function optimize!(ap::ActionPotential, opt_param_names::Tuple;
                   bounds::Union{NamedTuple,Nothing} = nothing,
                   use_gpu::Bool = false,
                   num_trajectories::Int = 100_000,
                   max_evals::Int = 20_000)
    find_foot!(ap)

    if use_gpu
        println("GPU mode: running grid search with $num_trajectories trajectories...")
        global_result = gpu_grid_search!(ap, opt_param_names;
                                          num_trajectories=num_trajectories, range=0.9,
                                          bounds=bounds)
    else
        # 20k (was 500k): the profile likelihood shows the g_Na/g_K/kinetics
        # landscape is degenerate, so extra global-search precision lands in the
        # same flat valley — a 25× speed-up with no meaningful loss of fit.
        global_result = global_optimize(ap, opt_param_names;
                                         max_evals=max_evals, range=0.9, bounds=bounds)
    end
    update_model!(ap, global_result["par"])

    final_result = optimize_model(ap, opt_param_names; bounds=bounds)
    update_model!(ap, final_result["par"])

    return Dict("par" => final_result["par"], "value" => final_result["value"],
                "convergence" => final_result["convergence"])
end

# ---------------------------------------------------------------------------
# GPU grid search (requires CUDA)
# ---------------------------------------------------------------------------
function gpu_grid_search!(ap::ActionPotential, opt_param_names::Tuple;
                           num_trajectories=10000, range=0.5,
                           bounds::Union{NamedTuple,Nothing} = nothing)
    println("Starting GPU grid search ($num_trajectories trajectories)...")
    initial_params = ap.params
    dt             = ap.dt
    time_points    = ap.time_points
    experimental_trace = ap.AP_val

    # Sample within physiological bounds when supplied — matching
    # global_optimize's search_range (bounds[name] when available, else
    # ±range% of the initial value) — NOT unconditional ±range% of initial
    # value for every parameter.
    #
    # This is not just a consistency nit: without it, the GPU search explores
    # a fundamentally different (and unphysiologically wide) parameter space
    # than the CPU search always has, and was empirically the trigger for a
    # reproducible EnsembleGPUKernel HANG (see the stim_dtmax comment below
    # for the tstops half of this story) — e.g. M_6's default is -17.6, but
    # ±90% of the *initial value* alone allows -1.76, an order of magnitude
    # closer to zero than par_bounds.M_6 = (-40, -3) permits; near M_6=0 the
    # beta_m exponential's sensitivity to V blows up, driving the explicit
    # GPU integrator into a step count blow-up (or worse) that dtmin/maxiters
    # did not reliably bound in testing. Bounding the search the same way the
    # CPU path already does removes those combinations from consideration.
    opt_subset = NamedTuple(k => initial_params[k] for k in opt_param_names)
    lb = Float64[]; ub = Float64[]
    for k in opt_param_names
        val = opt_subset[k]
        if !isnothing(bounds) && haskey(bounds, k)
            push!(lb, bounds[k][1]); push!(ub, bounds[k][2])
        else
            lo = val > 0 ? val * (1 - range) : val * (1 + range)
            hi = val > 0 ? val * (1 + range) : val * (1 - range)
            lo > hi && ((lo, hi) = (hi, lo))
            push!(lb, lo); push!(ub, hi)
        end
    end
    s          = SobolSeq(lb, ub)
    param_sets = Vector{NamedTuple}(undef, num_trajectories)
    for i in 1:num_trajectories
        p_vec         = next!(s)
        param_sets[i] = (; zip(keys(opt_subset), p_vec)...)
    end

    template_p = merge(initial_params,
                       (stim_d=ap.stim_d, stim_h=ap.stim_h,
                        stim_dim=ap.stim_dim, tot_wait=ap.tot_wait))
    u0    = @SVector [ap.V_init,
                      infty_n(ap.V_init, template_p),
                      infty_m(ap.V_init, template_p),
                      infty_h(ap.V_init, template_p)]
    tspan = (time_points[1], time_points[end])
    prob  = ODEProblem(hodgkin_huxley, u0, tspan, template_p)

    # Same stimulus-timing safeguard as the CPU path (_simulate_trace): every
    # trajectory shares the SAME tot_wait/stim_d (only opt_param_names vary —
    # neither is ever one of them), so a single step-size bound across the
    # pulse applies to the whole ensemble. Without this the adaptive GPU
    # integrator could take one large step across the brief depolarising pulse
    # on the flat pre-stimulus baseline and never resolve it.
    #
    # IMPORTANT — do NOT pass `tstops` here, only `dtmax`. Empirically (see
    # analysis/gpu_vs_cpu_benchmark.jl and the audit notes), passing `tstops`
    # to EnsembleGPUKernel/GPUTsit5 together with the wide (±90%) Sobol
    # parameter range used by this search HANGS INDEFINITELY on at least one
    # of ~200+ trajectories — reproduced directly, not a timeout guess: dtmin,
    # maxiters and force_dtmin below do NOT stop it either. `dtmax` alone (no
    # tstops) on the identical trajectory set completes reliably. This looks
    # like a genuine EnsembleGPUKernel/tstops interaction bug with
    # stiff/unstable trajectories (many Sobol draws at this range produce
    # non-firing or oscillatory dynamics) rather than anything under our
    # control — re-test if DiffEqGPU is upgraded.
    #
    # dtmax is deliberately set to stim_d itself, NOT stim_d/6 (which is what
    # the CPU path's tstops density effectively achieves). tstops only forces
    # fine resolution WITHIN the pulse while leaving the solver free to take
    # large strides over the other ~29 ms of flat baseline; dtmax has no such
    # locality — it caps the step size for the WHOLE tspan. With real data
    # (dt as fine as 0.005 ms), stim_d/6 forces >=6000 steps across the entire
    # 30 ms simulation for every trajectory (not just the ~0.7 ms pulse) and
    # was measured to turn a <70 s solve into a 20+ minute one. dtmax=stim_d
    # still guarantees no single step can skip over the ENTIRE pulse (the
    # actual failure mode being guarded against) while keeping the global
    # minimum step count modest (~30/stim_d, a few dozen). This resolves the
    # pulse coarsely rather than in >=6 sub-steps, but that trade-off is
    # necessary given tstops is unusable here — see the report.jmd note.
    stim_dtmax = ap.stim_d

    function prob_func(prob, ctx)
        i      = ctx.sim_id
        iter_p = merge(template_p, param_sets[i])
        u0_i   = @SVector [iter_p.RMP,
                            infty_n(iter_p.RMP, iter_p),
                            infty_m(iter_p.RMP, iter_p),
                            infty_h(iter_p.RMP, iter_p)]
        remake(prob, u0=u0_i, p=iter_p)
    end

    # Track the best result via a closure — output_func runs on the host side
    # after each GPU kernel completes, so a ReentrantLock makes it thread-safe.
    best_idx_ref   = Ref{Int}(1)
    best_score_ref = Ref{Float64}(Inf)
    lk             = ReentrantLock()

    # tot_wait is identical for every trajectory (only opt_param_names vary —
    # tot_wait is never one of them), so the scoring-window masks (see
    # _score_masks) can be built once and reused by output_func instead of
    # rebuilt from scratch for every one of up to num_trajectories calls. This
    # matters far more here than on the CPU path: output_func runs serially,
    # under a lock, once per GPU trajectory (up to 500_000+), so the previous
    # per-call BitVector allocation was a serial bottleneck that ate into (and
    # could exceed) whatever throughput the GPU integration itself gained.
    masks = _score_masks(ap.tot_wait, time_points, dt, ap.stabil_time, length(ap.trace_data))

    function output_func(sol, ctx)
        i = ctx.sim_id
        if !successful_retcode(sol) || length(sol.u) != length(time_points)
            return (Inf, false)
        end
        simV  = [u[1] for u in sol.u]
        score = _score_from_masks(simV, masks, experimental_trace)
        lock(lk) do
            if score < best_score_ref[]
                best_score_ref[] = score
                best_idx_ref[]   = i
            end
        end
        return (score, false)
    end

    ensemble_prob = EnsembleProblem(prob; prob_func=prob_func, output_func=output_func,
                                    reduction=(u, data, I) -> (append!(u, [data[1]]), false),
                                    u_init = Float64[])

    # --- Size the trajectory batch to fit GPU memory ---
    # EnsembleGPUKernel allocates the full solution (SVector{4,Float64}) and time
    # (Float64) arrays of size (n_save × batch) on the device at once. Solving all
    # trajectories in one batch needs n_save × num_trajectories × 40 bytes — e.g.
    # ~23 GiB for 100k trajectories × 6k save points, which OOMs a 16 GB card.
    # Batching keeps device memory bounded; trajectory indices stay global, so the
    # best-result tracking in output_func remains correct across batches.
    n_save         = length(time_points)
    bytes_per_traj = n_save * (sizeof(SVector{4, Float64}) + sizeof(Float64))
    free_mem       = CUDA.free_memory()
    batch_size     = clamp(floor(Int, 0.25 * free_mem / bytes_per_traj),
                           1, num_trajectories)
    @printf("GPU batch size: %d trajectories (free %.2f GiB, %d save points)\n",
            batch_size, free_mem / 2^30, n_save)

    # dtmax only — NOT tstops (see the long comment above stim_dtmax: tstops
    # reproducibly hangs here). dtmin/maxiters/force_dtmin mirror the CPU
    # path's safety net so a numerically pathological trajectory (common
    # among wide Sobol draws) fails fast with a bad retcode — scored as Inf
    # by output_func below — instead of the solver stalling on it.
    # NOTE: no `verbose=false` here (see the identical note in
    # get_full_trace_details) — this happens to still work through DiffEqGPU's
    # EnsembleGPUKernel dispatch today, but OrdinaryDiffEq's core solve() path
    # now rejects a Bool `verbose` outright, so dropping it here too avoids
    # relying on that difference persisting across DiffEqGPU versions.
    sol = solve(ensemble_prob, GPUTsit5(),
                DiffEqGPU.EnsembleGPUKernel(CUDABackend());
                trajectories=num_trajectories, batch_size=batch_size, saveat=dt,
                dtmax=stim_dtmax, dtmin=1e-6, maxiters=50_000, force_dtmin=false)

    best_idx    = best_idx_ref[]
    best_score  = best_score_ref[]
    full_params = merge(initial_params, param_sets[best_idx])
    println("GPU grid search complete. Best score: ", best_score,
            "  (trajectory ", best_idx, " of ", num_trajectories, ")")
    return Dict("par" => full_params, "value" => best_score)
end

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
function create_ap_plot(ap::ActionPotential)
    p = plot(ap.time_points, ap.Vs,
             label     = "Model",
             lw        = 2,
             title     = ap.name,
             xlabel    = "Time (ms)",
             ylabel    = "Voltage (mV)",
             legend    = :topleft)

    start_time = ap.stabil_time
    end_time   = start_time + (length(ap.trace_data) - 1) * ap.dt
    exp_time   = collect(start_time:ap.dt:end_time)
    if length(exp_time) > length(ap.trace_data)
        exp_time = exp_time[1:length(ap.trace_data)]
    end
    plot!(p, exp_time, ap.trace_data,
          label = "Experimental", ls = :dash, color = :red)

    stim_trace = [ap.tot_wait < t < ap.tot_wait + ap.stim_d ?
                  ap.stim_h * ((t - ap.tot_wait) / ap.stim_d)^ap.stim_dim : 0.0
                  for t in ap.time_points]
    p_twin = twinx(p)
    plot!(p_twin, ap.time_points, stim_trace,
          label   = "Stimulus",
          color   = :green,
          ls      = :dot,
          lw      = 2,
          ylabel  = "Current (μA/cm²)",
          legend  = :topright)
    return p
end

function display_action_potential(ap::ActionPotential)
    display(create_ap_plot(ap))
end

end # module ActionPotentialModel
