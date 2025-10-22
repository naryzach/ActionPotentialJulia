# ActionPotential.jl (High-Performance Version)

module ActionPotentialModel

using DifferentialEquations, Optim, Plots, Printf, Statistics
using Sobol, DataFrames, CSV, Distributed
using QuadGK, StaticArrays, BlackBoxOptim
using CUDA, DiffEqGPU

export ActionPotential, optimize!, display_action_potential

# --- Helper functions are now standalone for type-stability ---

# Nernst equation
nernst(z, cons_out, cons_in) = (8.31446 * 296.15) / (z * 96.485) * log(cons_out / cons_in)

# Helper for a smooth transition from 0, approximating max(0, x).
# This avoids derivative discontinuities that can cause solver instability.
function smooth_max_zero(x, k=20.0)
    # This is a numerically stable implementation of `log(1 + exp(k*x)) / k`.
    # The naive implementation `log1p(exp(k*x))/k` overflows when `k*x` is large.
    val = k * x
    if val > 0
        # log(1+exp(x)) = log(exp(x)*(exp(-x)+1)) = x + log(1+exp(-x)) = x + log1p(exp(-x))
        return (val + log1p(exp(-val))) / k
    else
        return log1p(exp(val)) / k
    end
end

# Gating variable alpha/beta rate functions
# The original formulation had a sharp corner, causing solver instability.
# We use the smooth_max_zero function to create a smooth, differentiable transition.
alpha_n(V, p) = p.N_1 * smooth_max_zero(V - p.N_2)
beta_n(V, p) = exp((V + p.N_7) / p.N_6)
alpha_m(V, p) = p.M_1 * smooth_max_zero(V - p.M_2)
beta_m(V, p) = exp((V + p.M_7) / p.M_6)
alpha_h(V, p) = exp((V + p.H_6) / p.H_3)
beta_h(V, p) = 1.0 / (1.0 + exp((V + p.H_4) / p.H_5))

# Gating variable steady-state value (infty)
infty_n(V, p) = alpha_n(V, p) / (alpha_n(V, p) + beta_n(V, p))
infty_m(V, p) = alpha_m(V, p) / (alpha_m(V, p) + beta_m(V, p))
infty_h(V, p) = alpha_h(V, p) / (alpha_h(V, p) + beta_h(V, p))

# Stimulus function
stim_function(t, stim_d, stim_h, stim_dim) = stim_h * (t / stim_d)^stim_dim

# --- Hodgkin-Huxley model differential equations ---

function hodgkin_huxley(u, p, t)
    V, n, m, h = u

    # Nernst potentials (can be pre-calculated, but fine here for clarity)
    E_K = nernst(1, 5.4, 143)
    E_Na = nernst(1, 145, 9.6)
    #E_Leak = nernst(-1, 127.7, 8.7)
    E_Leak = p.RMP # RMP is now passed in as a parameter

    # Calculate currents
    I_K = p.g_K * n^4 * (V - E_K)
    I_Na = p.g_Na * m^3 * h * (V - E_Na)
    I_Leak = p.g_Leak * (V - E_Leak)
    I_m = I_K + I_Na + I_Leak

    # Stimulus (stimulus params are now passed in via `p`)
    I_stim = (p.tot_wait < t < p.tot_wait + p.stim_d) ? stim_function((t - p.tot_wait), p.stim_d, p.stim_h, p.stim_dim) : 0.0

    # Differentials
    dV = I_stim - I_m
    dn = alpha_n(V, p) * (1 - n) - beta_n(V, p) * n
    dm = alpha_m(V, p) * (1 - m) - beta_m(V, p) * m
    dh = alpha_h(V, p) * (1 - h) - beta_h(V, p) * h

    return @SVector [dV, dn, dm, dh]
end

# This function takes parameters and returns a voltage trace, with no side effects.
function _simulate_trace(params::NamedTuple, time_points::Vector{Float64}, dt::Float64)
    u0 = @SVector [params.RMP, infty_n(params.RMP, params), infty_m(params.RMP, params), infty_h(params.RMP, params)]
    tspan = (time_points[1], time_points[end])
    prob = ODEProblem(hodgkin_huxley, u0, tspan, params)

    # To ensure the solver does not step over the stimulus, we force it to
    # stop at the exact start and end times. `tstops` is a lightweight and
    # robust way to handle this for discontinuous forcing functions.
    stim_on_time = params.tot_wait
    stim_off_time = params.tot_wait + params.stim_d

    sol = solve(prob, Rosenbrock23(), saveat=dt, reltol=1e-4, abstol=1e-4, tstops=[stim_on_time, stim_off_time])

    if sol.retcode != :Success || length(sol.u) != length(time_points)
        return fill(Inf, length(time_points)) 
    else
        return [v[1] for v in sol.u]
    end
end

# This function takes parameters and data, and returns a score, with no side effects.
function _calculate_score(params::NamedTuple, time_points::Vector{Float64}, dt::Float64, experimental_trace_padded::Vector{Float64}, stabil_time::Float64, trace_data_len::Int)
    simulated_Vs = _simulate_trace(params, time_points, dt)
    if isinf(simulated_Vs[1]); return Inf; end

    pk_dur_est = 5.0
    peak_start = params.tot_wait
    peak_end = peak_start + pk_dur_est
    
    # Define scoring windows correctly
    # 1. The entire baseline, from the start of the data to the stimulus
    idx_baseline = (time_points .>= stabil_time) .& (time_points .< peak_start)
    # 2. The peak region
    idx_peak = (time_points .>= peak_start) .& (time_points .<= peak_end)
    # 3. The post-peak region
    idx_post_peak = (time_points .> peak_end) .& (time_points .< (stabil_time + trace_data_len * dt))
    
    # Sum the weighted errors from all three regions
    score_val = sum((simulated_Vs[idx_baseline] .- experimental_trace_padded[idx_baseline]).^2) +
                5.0 * sum((simulated_Vs[idx_peak] .- experimental_trace_padded[idx_peak]).^2) +
                sum((simulated_Vs[idx_post_peak] .- experimental_trace_padded[idx_post_peak]).^2)

    return isfinite(score_val) ? score_val : Inf
end

# --- Main Action Potential structure ---

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

# --- Constructor ---

function ActionPotential(p_in::Union{NamedTuple, Dict}, trace, time; name="Nameless")
    initial_params = typeof(p_in) <: Dict ? NamedTuple(p_in) : p_in
    dt = round(time[2] - time[1], digits=3)
    sim_time = 30.0; stabil_time = 10.0; t_sim = 0:dt:sim_time

    # First, calculate the RMP from the trace data
    peak_approx_time = 2.0; pk_dur_est = 5.0
    early_trace = trace[1:round(Int, peak_approx_time / dt)]
    late_trace_start = round(Int, (pk_dur_est + 1) / dt)
    late_trace_end = round(Int, (pk_dur_est + 1.2) / dt)
    late_trace = trace[late_trace_start:min(late_trace_end, end)]
    calculated_RMP = isempty(early_trace) ? trace[1] : minimum(early_trace) + 
          (maximum(late_trace) - minimum(late_trace)) / 2

    # Unconditionally overwrite the RMP from the config file with the
    # value calculated from the data. This ensures the model ALWAYS starts
    # with the data-driven RMP.
    params = merge(initial_params, (RMP = calculated_RMP,))
    
    # The rest of the constructor uses this corrected `params` object
    V_init = params.RMP
    start_idx = round(Int, stabil_time / dt) + 1
    end_idx = start_idx + length(trace) - 1
    AP_val = fill(params.RMP, length(t_sim))
    AP_val[start_idx:end_idx] = trace
    
    ap = ActionPotential(
        name, params, trace, collect(t_sim), dt,
        0.64, 54.874, 2.0, # stim defaults
        stabil_time, stabil_time, # stabil_time, tot_wait
        V_init,
        zeros(length(t_sim)), AP_val
    )

    update_model!(ap, ap.params)
    return ap
end

# Helper function to get detailed state variables from a simulation
# This is needed for the detailed plots below.
function get_full_trace_details(ap::ActionPotential)
    p = merge(ap.params, (stim_d=ap.stim_d, stim_h=ap.stim_h, stim_dim=ap.stim_dim, tot_wait=ap.tot_wait))
    u0 = @SVector [ap.V_init, 
            infty_n(ap.V_init, p), 
            infty_m(ap.V_init, p), 
            infty_h(ap.V_init, p)]
    tspan = (ap.time_points[1], ap.time_points[end])
    prob = ODEProblem(hodgkin_huxley, u0, tspan, p)
    stim_on_time = p.tot_wait; stim_off_time = p.tot_wait + p.stim_d
    sol = solve(prob, Rosenbrock23(), saveat=ap.dt, tstops=[stim_on_time, stim_off_time])

    # Unpack solution
    V = [u[1] for u in sol.u]
    n = [u[2] for u in sol.u]
    m = [u[3] for u in sol.u]
    h = [u[4] for u in sol.u]
    
    # Recalculate currents and conductances
    E_K = nernst(1, 5.4, 143)
    E_Na = nernst(1, 145, 9.6)
    IKs = p.g_K .* (n.^4) .* (V .- E_K)
    INas = p.g_Na .* (m.^3) .* h .* (V .- E_Na)
    ILeaks = p.g_Leak .* (V .- p.RMP)
    Istims = [p.tot_wait < t < p.tot_wait+p.stim_d ? p.stim_h*((t-p.tot_wait)/p.stim_d)^p.stim_dim : 0.0 for t in ap.time_points]
    
    gKs = p.g_K .* (n.^4)
    gNas = p.g_Na .* (m.^3) .* h

    return (t=ap.time_points, V=V, IKs=IKs, INas=INas, ILeaks=ILeaks, Istims=Istims, gKs=gKs, gNas=gNas)
end

# --- Model Methods ---

# Update the model trace based on parameters
function update_model!(ap::ActionPotential, params::NamedTuple)
    ap.params = params
    # The full parameter set for simulation includes stimulus parameters
    sim_params = merge(params, (stim_d=ap.stim_d, stim_h=ap.stim_h, stim_dim=ap.stim_dim, tot_wait=ap.tot_wait))
    ap.Vs = _simulate_trace(sim_params, ap.time_points, ap.dt)
end

# Find the shape of the foot of the action potential
function find_foot!(ap::ActionPotential; num_fits=10)
    println("\n--- Searching for AP foot (Constant Charge Method) ---")
    
    initial_params = (d=ap.stim_d, h=ap.stim_h, dim=ap.stim_dim)
    const_integral_A = (initial_params.h * initial_params.d) / (initial_params.dim + 1)
    @printf("Target stimulus integral (A) set to: %.4f\n", const_integral_A)

    function objective(foot_params)
        t_0, stim_d, stim_dim = foot_params
        if t_0 < 0 || t_0 > 2.0 || stim_d < 0.1 || stim_d > 2.0 || stim_dim < 1.0; return Inf; end
        
        stim_h_new = const_integral_A * (stim_dim + 1) / stim_d
        if !isfinite(stim_h_new) || stim_h_new < 0; return Inf; end

        tot_wait = ap.stabil_time + t_0
        start_idx = round(Int, tot_wait/ap.dt)+1; end_idx = start_idx + round(Int, stim_d/ap.dt)
        if end_idx > length(ap.AP_val) return Inf end
        
        exp_trace_stim_window = ap.AP_val[start_idx:end_idx]
        stim_integral_trace = similar(exp_trace_stim_window)
        for (i, _) in enumerate(exp_trace_stim_window)
            integral, _ = quadgk(t -> stim_function(t, stim_d, stim_h_new, stim_dim), 0, i * ap.dt)
            stim_integral_trace[i] = integral + ap.params.RMP
        end
        return sum((stim_integral_trace .- exp_trace_stim_window).^2)
    end

    best_result = nothing; min_value = Inf
    for i in 1:num_fits
        t_0_guess=(i/num_fits)*0.5; initial_opt_params = [t_0_guess, initial_params.d, initial_params.dim]
        result = optimize(objective, initial_opt_params, NelderMead())
        if Optim.minimum(result) < min_value; min_value=Optim.minimum(result); best_result=result; end
    end
    
    best_params = Optim.minimizer(best_result)
    t_0, stim_d, stim_dim = best_params
    
    ap.stim_d = stim_d
    ap.stim_dim = stim_dim
    ap.stim_h = const_integral_A * (stim_dim + 1) / stim_d
    ap.tot_wait = ap.stabil_time + t_0
    
    @printf("Foot found. Optimal t_0: %.3f ms, d: %.3f, m: %.2f, constrained h: %.2f\n", t_0, ap.stim_d, ap.stim_dim, ap.stim_h)
    update_model!(ap, ap.params)
end

# Standarized full optimization routine
function optimize!(ap::ActionPotential, opt_param_names::Tuple)
    # First, ensure the foot is found
    find_foot!(ap)

    # Stage 1: Global Search
    global_result = global_optimize(ap, opt_param_names, max_evals=500000, range=0.9)
    global_params = global_result["par"]
    
    # Update the model with the result of the global search
    update_model!(ap, global_params)
    
    # Stage 2: Local Refinement
    final_result = optimize_model(ap, opt_param_names)

    update_model!(ap, final_result["par"])

    return Dict("par" => final_result["par"], "value" => final_result["value"], "convergence" => final_result["convergence"])
end

# Optimization of model parameters to fit the experimental trace
function optimize_model(ap::ActionPotential, opt_param_names::Tuple)
    println("Starting local refinement with Optim.jl...")
    initial_params = ap.params
    static_data = (time_points = ap.time_points, dt = ap.dt, experimental_trace = ap.AP_val, stim_d = ap.stim_d, stim_h = ap.stim_h, stim_dim = ap.stim_dim, tot_wait = ap.tot_wait, stabil_time = ap.stabil_time, trace_data_len = length(ap.trace_data))

    function objective(p_vec)
        iter_params_subset = (; zip(opt_param_names, p_vec)...)
        full_iter_params = merge(initial_params, static_data, iter_params_subset)
        return _calculate_score(full_iter_params, static_data.time_points, static_data.dt, static_data.experimental_trace, static_data.stabil_time, static_data.trace_data_len)
    end

    initial_params_vec = [initial_params[k] for k in opt_param_names]
    result = optimize(objective, initial_params_vec, NelderMead(), Optim.Options(iterations=5000, f_reltol=1e-9))
    
    final_p_vec = Optim.minimizer(result); final_params_subset = (;zip(opt_param_names, final_p_vec)...)
    full_final_params = merge(initial_params, final_params_subset)

    println("Local refinement finished. Best score: ", Optim.minimum(result))
    return Dict("par" => full_final_params, "value" => Optim.minimum(result), "convergence" => Optim.converged(result))
end

# Nudge optimization: multiple fits from perturbed initial conditions in parallel
function global_optimize(ap::ActionPotential, opt_param_names::Tuple; max_evals=2500, range=0.5)
    println("Starting global optimization with BlackBoxOptim...")
    initial_params = ap.params
    static_data = (
        time_points = ap.time_points, dt = ap.dt, experimental_trace = ap.AP_val,
        stim_d = ap.stim_d, stim_h = ap.stim_h, stim_dim = ap.stim_dim, tot_wait = ap.tot_wait,
        stabil_time = ap.stabil_time, trace_data_len = length(ap.trace_data)
    )

    function objective(p_vec)
        iter_params_subset = (; zip(opt_param_names, p_vec)...)
        full_iter_params = merge(initial_params, static_data, iter_params_subset)
        return _calculate_score(full_iter_params, static_data.time_points, static_data.dt, static_data.experimental_trace, static_data.stabil_time, static_data.trace_data_len)
    end

    # Define the search space for the optimizer using the new `range` parameter
    search_range = Tuple{Float64, Float64}[];
    for name in opt_param_names
        val = initial_params[name]
        # Use the `range` argument to define the search window
        lower_bound = val > 0 ? val * (1.0 - range) : val * (1.0 + range)
        upper_bound = val > 0 ? val * (1.0 + range) : val * (1.0 - range)
        if lower_bound > upper_bound; lower_bound, upper_bound = upper_bound, lower_bound; end
        push!(search_range, (lower_bound, upper_bound))
    end
    
    result = bboptimize(objective; 
                        SearchRange = search_range,
                        NumDimensions = length(opt_param_names),
                        MaxFuncEvals = max_evals,
                        TraceMode = :silent)
    
    final_p_vec = best_candidate(result)
    final_params_subset = (;zip(opt_param_names, final_p_vec)...)
    full_final_params = merge(initial_params, final_params_subset)

    println("Global optimization finished. Best score: ", best_fitness(result))
    return Dict("par" => full_final_params, "value" => best_fitness(result))
end

# Use the GPU to accelerate the simulation (if available)
function gpu_grid_search!(ap::ActionPotential, opt_param_names::Tuple; num_trajectories=10000, range=0.5)
    println("Starting GPU-based grid search with $(num_trajectories) parameter sets...")

    # --- Setup
    initial_params = ap.params
    dt = ap.dt
    time_points = ap.time_points
    experimental_trace = ap.AP_val

    # Generate Sobol samples for parameters
    opt_params_subset = NamedTuple(k => initial_params[k] for k in opt_param_names)
    s = SobolSeq(length(opt_params_subset))
    param_sets = Vector{NamedTuple}(undef, num_trajectories)
    for i in 1:num_trajectories
        p_factors = next!(s) .* (2 * range) .+ (1.0 - range)
        p_vec = values(opt_params_subset) .* p_factors
        param_sets[i] = (; zip(keys(opt_params_subset), p_vec)...)
    end

    # Template problem (only HH params, no data arrays!)
    template_p = merge(initial_params, (stim_d=ap.stim_d, stim_h=ap.stim_h, stim_dim=ap.stim_dim, tot_wait=ap.tot_wait))
    u0 = @SVector [ap.V_init, infty_n(ap.V_init, template_p), infty_m(ap.V_init, template_p), infty_h(ap.V_init, template_p)]
    tspan = (time_points[1], time_points[end])
    prob = ODEProblem(hodgkin_huxley, u0, tspan, template_p)

    # Define how to remake each problem for different param sets
    function prob_func(prob, i, repeat)
        iter_params = merge(template_p, param_sets[i])
        u0 = @SVector [iter_params.RMP, infty_n(iter_params.RMP, iter_params),
                       infty_m(iter_params.RMP, iter_params),
                       infty_h(iter_params.RMP, iter_params)]
        remake(prob, u0=u0, p=iter_params)
    end

    # Score function directly from GPU solution
    function output_func(sol, i)
        if sol.retcode != :Success
            return Inf
        end
        simV = [u[1] for u in sol.u]
        return _calculate_score(sol.prob.p, time_points, dt, experimental_trace,
                                ap.stabil_time, length(ap.trace_data))
    end

    # Ensemble setup
    ensemble_prob = EnsembleProblem(prob; prob_func=prob_func, output_func=output_func,
                                    reduction=(u, data, I) -> (min(u, data), false))

    # GPU solve
    sol = solve(ensemble_prob,
            GPUTsit5(),
            DiffEqGPU.EnsembleGPUKernel(CUDABackend());
            trajectories=num_trajectories,
            saveat=dt)

    best_score = sol.u  # reduction stores the min score here
    println("GPU grid search finished. Best score: ", best_score)

    # Find the params corresponding to that score (brute force search on CPU side)
    scores = [output_func(solve(prob_func(prob, i, 0), RK4(), dt=dt), i) for i in 1:num_trajectories]
    best_idx = argmin(scores)
    best_params_subset = param_sets[best_idx]
    full_final_params = merge(initial_params, best_params_subset)

    return Dict("par" => full_final_params, "value" => best_score)
end

#--- Visualization ---

function create_ap_plot(ap::ActionPotential)
    p = plot(ap.time_points, ap.Vs, 
             label="Model Trace", 
             lw=2, 
             title=ap.name, 
             xlabel="Time (ms)", 
             ylabel="Voltage (mV)",
             legend=:topleft)
    
    # The experimental data always starts after the fixed stabilization time.
    start_time_data = ap.stabil_time
    end_time_data = start_time_data + (length(ap.trace_data) - 1) * ap.dt
    experimental_time = start_time_data:ap.dt:end_time_data

    if length(experimental_time) > length(ap.trace_data)
        experimental_time = experimental_time[1:length(ap.trace_data)]
    end
    
    # Plot the experimental data at its fixed, correct location
    plot!(p, experimental_time, ap.trace_data, label="Experimental Data", ls=:dash, color=:red)
    
    # The stimulus trace correctly uses the optimized 'tot_wait'
    stim_trace = zeros(length(ap.time_points))
    for (i, t) in enumerate(ap.time_points)
        if ap.tot_wait < t < ap.tot_wait + ap.stim_d
            stim_trace[i] = ap.stim_h * ((t - ap.tot_wait) / ap.stim_d)^ap.stim_dim
        end
    end
    
    p_twin = twinx(p)
    plot!(p_twin, ap.time_points,
        stim_trace, 
        label="Stimulus Current", 
        color=:green, 
        ls=:dot, 
        lw=2, 
        ylabel="Current (μA/cm²)", 
        legend=:topright
    )
    
    return p
end

# Display the action potential trace with experimental data overlay
function display_action_potential(ap::ActionPotential)
    p = create_ap_plot(ap)
    display(p)
end

end # end module