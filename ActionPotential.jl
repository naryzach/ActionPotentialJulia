# ActionPotential.jl

module ActionPotentialModel

using DifferentialEquations, Optim, Plots, Printf, Statistics
using Sobol, DataFrames, CSV, Distributed
using QuadGK, StaticArrays, BlackBoxOptim
using CUDA, DiffEqGPU

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

    sol = solve(prob, Rosenbrock23(),
                saveat = dt, reltol = 1e-4, abstol = 1e-4,
                tstops = [stim_on, stim_off])

    if sol.retcode != :Success || length(sol.u) != length(time_points)
        return fill(Inf, length(time_points))
    end
    return [v[1] for v in sol.u]
end

# ---------------------------------------------------------------------------
# Scoring functions (pure, no side effects)
# ---------------------------------------------------------------------------

# Core scoring given an already-computed voltage vector.
# Separated from _calculate_score so GPU grid search can use the GPU-solved
# trajectory directly without re-solving on CPU.
function _score_from_trace(simulated_Vs::Vector{Float64},
                            tot_wait::Float64,
                            time_points::Vector{Float64},
                            dt::Float64,
                            experimental_trace_padded::Vector{Float64},
                            stabil_time::Float64,
                            trace_data_len::Int)
    isempty(simulated_Vs) && return Inf
    any(isinf, simulated_Vs) && return Inf

    pk_dur_est = 5.0
    peak_start = tot_wait
    peak_end   = peak_start + pk_dur_est

    idx_baseline  = (time_points .>= stabil_time) .& (time_points .<  peak_start)
    idx_peak      = (time_points .>= peak_start)  .& (time_points .<= peak_end)
    idx_post_peak = (time_points .>  peak_end)    .& (time_points .<  (stabil_time + trace_data_len * dt))

    score_val = (
        sum((simulated_Vs[idx_baseline]  .- experimental_trace_padded[idx_baseline]).^2)  +
        5.0 * sum((simulated_Vs[idx_peak] .- experimental_trace_padded[idx_peak]).^2)     +
        sum((simulated_Vs[idx_post_peak] .- experimental_trace_padded[idx_post_peak]).^2)
    )
    return isfinite(score_val) ? score_val : Inf
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
# Constructor
# ---------------------------------------------------------------------------
function ActionPotential(p_in::Union{NamedTuple, Dict}, trace, time; name="Nameless")
    initial_params = typeof(p_in) <: Dict ? NamedTuple(p_in) : p_in
    dt          = round(time[2] - time[1], digits=3)
    sim_time    = 30.0
    stabil_time = 10.0
    t_sim       = 0:dt:sim_time

    # RMP is the mean of the pre-stimulus baseline (first 2 ms of the recording).
    # This is more robust than a derived formula and avoids aliasing with the
    # AP upstroke or afterhyperpolarization.
    baseline_end_idx = min(round(Int, 2.0 / dt), length(trace))
    early_trace      = trace[1:baseline_end_idx]
    calculated_RMP   = isempty(early_trace) ? trace[1] : mean(early_trace)

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

    # Each g_Na value is an independent optimisation — run in parallel threads.
    profile_scores = Vector{Float64}(undef, n_points)
    Threads.@threads for i in 1:n_points
        gNa_fixed = gNa_values[i]
        fixed_p   = merge(ap.params, (g_Na = gNa_fixed,))
        function obj(p_vec)
            iter_p = merge(fixed_p, static_data, (; zip(remaining, p_vec)...))
            score  = _calculate_score(iter_p, static_data.time_points, static_data.dt,
                                      static_data.experimental_trace,
                                      static_data.stabil_time, static_data.trace_data_len)
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
function find_foot!(ap::ActionPotential; num_fits=10)
    println("\n--- Searching for AP foot (constant-charge method) ---")
    init_p        = (d=ap.stim_d, h=ap.stim_h, dim=ap.stim_dim)
    const_integral = (init_p.h * init_p.d) / (init_p.dim + 1)
    @printf("Target stimulus integral: %.4f\n", const_integral)

    function objective(foot_params)
        t_0, stim_d, stim_dim = foot_params
        # Explicit if is more reliable than && in closures called via Optim internals.
        if t_0 < 0 || t_0 > 2.0 || stim_d <= 0.0 || stim_d > 2.0 || stim_dim < 1.0
            return Inf
        end

        stim_h_new = const_integral * (stim_dim + 1) / stim_d
        if !isfinite(stim_h_new) || stim_h_new < 0
            return Inf
        end

        tot_wait  = ap.stabil_time + t_0
        start_idx = round(Int, tot_wait / ap.dt) + 1
        end_idx   = start_idx + round(Int, stim_d / ap.dt)
        if end_idx > length(ap.AP_val)
            return Inf
        end

        exp_window   = ap.AP_val[start_idx:end_idx]
        model_window = similar(exp_window)
        for (i, _) in enumerate(exp_window)
            model_window[i] = stim_integral(i * ap.dt, stim_d, stim_h_new, stim_dim) + ap.params.RMP
        end
        return sum((model_window .- exp_window).^2)
    end

    # Run fits sequentially — the overhead is negligible (each NelderMead run
    # takes < 1 ms) and threading a shared closure via @threads risks Optim's
    # internal simplex workspace being corrupted by concurrent Julia task scheduling.
    results = Vector{Any}(undef, num_fits)
    for i in 1:num_fits
        t0_guess   = (i / num_fits) * 0.5
        results[i] = optimize(objective, [t0_guess, init_p.d, init_p.dim], NelderMead())
    end
    best_result = results[argmin(Optim.minimum.(results))]

    t_0, stim_d, stim_dim = Optim.minimizer(best_result)
    ap.stim_d   = stim_d
    ap.stim_dim = stim_dim
    ap.stim_h   = const_integral * (stim_dim + 1) / stim_d
    ap.tot_wait = ap.stabil_time + t_0
    @printf("Foot: t_0=%.3f ms, d=%.3f, dim=%.2f, h=%.2f\n",
            t_0, ap.stim_d, ap.stim_dim, ap.stim_h)
    update_model!(ap, ap.params)
end

# ---------------------------------------------------------------------------
# Local refinement (NelderMead + optional bounds penalty)
# ---------------------------------------------------------------------------
function optimize_model(ap::ActionPotential, opt_param_names::Tuple;
                         bounds::Union{NamedTuple,Nothing} = nothing)
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

    function objective(p_vec)
        iter_p = merge(initial_params, static_data, (; zip(opt_param_names, p_vec)...))
        score  = _calculate_score(iter_p, static_data.time_points, static_data.dt,
                                  static_data.experimental_trace,
                                  static_data.stabil_time, static_data.trace_data_len)
        if !isnothing(bounds)
            score += _bounds_penalty(p_vec, opt_param_names, bounds)
        end
        return score
    end

    init_vec = [initial_params[k] for k in opt_param_names]
    result   = optimize(objective, init_vec, NelderMead(),
                        Optim.Options(iterations=5000, f_reltol=1e-9))

    final_vec    = Optim.minimizer(result)
    final_subset = (; zip(opt_param_names, final_vec)...)
    full_params  = merge(initial_params, final_subset)

    println("Local refinement complete. Best score: ", Optim.minimum(result))
    return Dict("par" => full_params, "value" => Optim.minimum(result),
                "convergence" => Optim.converged(result))
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

    function objective(p_vec)
        iter_p = merge(initial_params, static_data, (; zip(opt_param_names, p_vec)...))
        return _calculate_score(iter_p, static_data.time_points, static_data.dt,
                                static_data.experimental_trace,
                                static_data.stabil_time, static_data.trace_data_len)
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
                   num_trajectories::Int = 100_000)
    find_foot!(ap)

    if use_gpu
        println("GPU mode: running grid search with $num_trajectories trajectories...")
        global_result = gpu_grid_search!(ap, opt_param_names;
                                          num_trajectories=num_trajectories, range=0.9)
    else
        global_result = global_optimize(ap, opt_param_names;
                                         max_evals=500_000, range=0.9, bounds=bounds)
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
                           num_trajectories=10000, range=0.5)
    println("Starting GPU grid search ($num_trajectories trajectories)...")
    initial_params = ap.params
    dt             = ap.dt
    time_points    = ap.time_points
    experimental_trace = ap.AP_val

    opt_subset = NamedTuple(k => initial_params[k] for k in opt_param_names)
    s          = SobolSeq(length(opt_subset))
    param_sets = Vector{NamedTuple}(undef, num_trajectories)
    for i in 1:num_trajectories
        p_factors   = next!(s) .* (2 * range) .+ (1.0 - range)
        p_vec       = values(opt_subset) .* p_factors
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

    function prob_func(prob, i, repeat)
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

    function output_func(sol, i)
        # Use the GPU-computed solution directly — do NOT call _simulate_trace
        # here (that would re-solve the ODE on CPU, defeating the GPU entirely).
        if sol.retcode != :Success || length(sol.u) != length(time_points)
            return (Inf, false)
        end
        simV  = [u[1] for u in sol.u]
        score = _score_from_trace(simV, sol.prob.p.tot_wait, time_points, dt,
                                   experimental_trace, ap.stabil_time,
                                   length(ap.trace_data))
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
    sol = solve(ensemble_prob, GPUTsit5(),
                DiffEqGPU.EnsembleGPUKernel(CUDABackend());
                trajectories=num_trajectories, saveat=dt)

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
