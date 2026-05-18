# action_potential_plots.jl

module ActionPotentialPlots

using Plots, StatsPlots, DataFrames, Printf, LaTeXStrings, Statistics, StaticArrays
using ..ActionPotentialModel

# ---------------------------------------------------------------------------
# Raw experimental traces
# ---------------------------------------------------------------------------
function plot_raw_traces(df_raw::DataFrame, group_name::String)
    p = plot(title="Raw Traces: $(group_name)", xlabel="Time (ms)", ylabel="Voltage (mV)")
    num_traces = Int(ncol(df_raw) / 2)
    for i in 1:num_traces
        trace = collect(skipmissing(df_raw[:, 2*i]))
        time  = collect(skipmissing(df_raw[:, 2*i-1]))
        plot!(p, time, trace, label=names(df_raw)[2*i], lw=1, alpha=0.7)
    end
    return p
end

# ---------------------------------------------------------------------------
# Parameter distribution boxplots
# ---------------------------------------------------------------------------
function plot_param_boxplots(sim_data::DataFrame, params_to_plot::Tuple)
    plots_list = []
    for param in params_to_plot
        p = @df sim_data boxplot(:group, cols(param),
                                  group=:group, legend=false,
                                  title=string(param),
                                  ylabel=string(param))
        push!(plots_list, p)
    end
    ncols = min(3, length(plots_list))
    nrows = ceil(Int, length(plots_list) / ncols)
    return plot(plots_list..., layout=(nrows, ncols), size=(1200, 350 * nrows), dpi=150)
end

# ---------------------------------------------------------------------------
# Average simulated action potential per group
# ---------------------------------------------------------------------------
function plot_average_traces(sim_data::DataFrame)
    avg = combine(groupby(sim_data, :group), names(sim_data, Real) .=> mean)
    rename!(avg, Dict(old => Symbol(replace(String(old), "_mean" => ""))
                      for old in names(avg)))

    p = plot(title="Average Fitted AP by Group",
             xlabel="Time (ms)", ylabel="Voltage (mV)")
    dummy_time  = collect(0.0:0.02:20.0)
    dummy_trace = fill(-70.0, length(dummy_time))

    for row in eachrow(avg)
        avg_ap = ActionPotentialModel.ActionPotential(NamedTuple(row), dummy_trace, dummy_time,
                                                       name=row.group)
        plot!(p, avg_ap.time_points, avg_ap.Vs, label=row.group, lw=2)
    end
    return p
end

# ---------------------------------------------------------------------------
# Decomposed ionic currents
# ---------------------------------------------------------------------------
function plot_currents(ap::ActionPotential, group_name::String)
    d = ActionPotentialModel.get_full_trace_details(ap)
    plot(d.t, [d.IKs d.INas d.ILeaks],
         xlabel="Time (ms)", ylabel="Current (μA/cm²)",
         label=["I_K" "I_Na" "I_Leak"],
         plot_title="Ionic Currents — $(group_name)",
         lw=2)
end

# ---------------------------------------------------------------------------
# Decomposed conductances
# ---------------------------------------------------------------------------
function plot_conductances(ap::ActionPotential, group_name::String)
    d = ActionPotentialModel.get_full_trace_details(ap)
    plot(d.t, [d.gKs d.gNas],
         xlabel="Time (ms)", ylabel="Conductance (mS/cm²)",
         label=["g_K" "g_Na"],
         plot_title="Conductances — $(group_name)",
         lw=2)
end

# ---------------------------------------------------------------------------
# Gating-variable state (m, n, h) over time
# ---------------------------------------------------------------------------
function plot_gating_variables(ap::ActionPotential, group_name::String)
    d = ActionPotentialModel.get_full_trace_details(ap)
    p1 = plot(d.t, d.V,  label="V (mV)",  ylabel="Voltage (mV)", lw=2, color=:black)
    p2 = plot(d.t, [d.m d.n d.h],
              label=[L"m" L"n" L"h"],
              ylabel="Gate probability", xlabel="Time (ms)", lw=2)
    return plot(p1, p2, layout=(2,1), size=(800,600), dpi=150,
                plot_title="Gating Variables — $(group_name)")
end

# ---------------------------------------------------------------------------
# Subunit steady-state kinetics and time constants
# ---------------------------------------------------------------------------
function plot_subunit_kinetics(params::NamedTuple, group_name::String)
    V_range = -100:1:60

    m_inf = ActionPotentialModel.infty_m.(V_range, Ref(params))
    n_inf = ActionPotentialModel.infty_n.(V_range, Ref(params))
    h_inf = ActionPotentialModel.infty_h.(V_range, Ref(params))

    tau_m = 1 ./ (ActionPotentialModel.alpha_m.(V_range, Ref(params)) .+
                  ActionPotentialModel.beta_m.(V_range,  Ref(params)))
    tau_n = 1 ./ (ActionPotentialModel.alpha_n.(V_range, Ref(params)) .+
                  ActionPotentialModel.beta_n.(V_range,  Ref(params)))
    tau_h = 1 ./ (ActionPotentialModel.alpha_h.(V_range, Ref(params)) .+
                  ActionPotentialModel.beta_h.(V_range,  Ref(params)))

    p1 = plot(V_range, [m_inf n_inf h_inf],
              title="Steady-State Activation / Inactivation",
              xlabel="Voltage (mV)", ylabel="Probability",
              label=[L"m_{\infty}" L"n_{\infty}" L"h_{\infty}"],
              lw=2, legend=:right)
    # Mark the resting potential
    if haskey(params, :RMP)
        vline!(p1, [params.RMP], label="RMP", ls=:dash, color=:grey, lw=1)
    end

    p2 = plot(V_range, [tau_m tau_n tau_h],
              title="Time Constants",
              xlabel="Voltage (mV)", ylabel="τ (ms)",
              label=[L"\tau_m" L"\tau_n" L"\tau_h"],
              lw=2, legend=:top)

    return plot(p1, p2, layout=(2,1), size=(800, 700), dpi=150,
                plot_title="Subunit Kinetics — $(group_name)")
end

# ---------------------------------------------------------------------------
# Alpha / beta rate functions
# ---------------------------------------------------------------------------
function plot_alpha_beta(params::NamedTuple, group_name::String)
    V_range = -100:1:60
    pn = plot(V_range, [ActionPotentialModel.alpha_n.(V_range, Ref(params))
                         ActionPotentialModel.beta_n.(V_range,  Ref(params))],
              title="n-gate", label=["α_n" "β_n"], lw=2)
    pm = plot(V_range, [ActionPotentialModel.alpha_m.(V_range, Ref(params))
                         ActionPotentialModel.beta_m.(V_range,  Ref(params))],
              title="m-gate", label=["α_m" "β_m"], lw=2)
    ph = plot(V_range, [ActionPotentialModel.alpha_h.(V_range, Ref(params))
                         ActionPotentialModel.beta_h.(V_range,  Ref(params))],
              title="h-gate", label=["α_h" "β_h"], lw=2)
    return plot(pn, pm, ph, layout=(3,1),
                xlabel="Voltage (mV)", ylabel="Rate (ms⁻¹)",
                size=(800, 900), dpi=150,
                plot_title="α/β Rate Functions — $(group_name)")
end

# ---------------------------------------------------------------------------
# AP features: violin / box plots comparing groups
# ---------------------------------------------------------------------------
function plot_ap_features(features_df::DataFrame)
    feature_cols = [:V_peak, :AP_amplitude, :max_dvdt, :V_threshold,
                    :t_peak_ms, :APD50, :AHP_depth]
    # Filter to columns that actually exist
    available = [f for f in feature_cols if f in propertynames(features_df)]
    isempty(available) && return plot(title="No feature data available")

    plots_list = []
    for feat in available
        p = @df features_df violin(:group, cols(feat),
                                    group=:group, legend=false,
                                    title=string(feat),
                                    ylabel=string(feat),
                                    alpha=0.7)
        @df features_df dotplot!(:group, cols(feat), group=:group,
                                  mode=:none, marker=(:black, 3), legend=false)
        push!(plots_list, p)
    end
    ncols = min(3, length(plots_list))
    nrows = ceil(Int, length(plots_list) / ncols)
    return plot(plots_list..., layout=(nrows, ncols),
                size=(1200, 350 * nrows), dpi=150,
                plot_title="AP Feature Distributions by Group")
end

# ---------------------------------------------------------------------------
# Profile likelihood for g_Na
# ---------------------------------------------------------------------------
function plot_profile_likelihood(profile::NamedTuple; group_name::String="")
    min_score = minimum(profile.scores)
    # 95% confidence boundary under χ² approximation (Δ score = 1.92)
    ci_threshold = min_score + 1.92

    p = plot(profile.gNa_values, profile.scores,
             lw=2, marker=:circle, markersize=4,
             xlabel="g_Na (mS/cm²)",
             ylabel="Profile score (SSE)",
             title="Profile Likelihood — g_Na$(isempty(group_name) ? "" : " ($group_name)")",
             legend=:topright,
             label="Profile score")
    hline!(p, [ci_threshold], ls=:dash, color=:red, lw=1, label="95% CI boundary")
    vline!(p, [profile.optimal_gNa], ls=:dot, color=:blue, lw=1, label="Optimal g_Na")
    return p
end

# ---------------------------------------------------------------------------
# Pairwise scatter plots
# ---------------------------------------------------------------------------
function plot_pairwise_params(sim_data::DataFrame, params_to_plot::Tuple)
    plots_list = []
    for i in 1:length(params_to_plot)
        for j in (i+1):length(params_to_plot)
            p1_name = params_to_plot[i]
            p2_name = params_to_plot[j]
            p = @df sim_data scatter(cols(p1_name), cols(p2_name),
                                     xlabel=string(p1_name),
                                     ylabel=string(p2_name),
                                     group=:group, alpha=0.4, legend=false)
            push!(plots_list, p)
        end
    end
    sz = ceil(Int, sqrt(length(plots_list)))
    return plot(plots_list..., layout=(sz, sz), size=(1000, 1000), dpi=100)
end

end # module ActionPotentialPlots
