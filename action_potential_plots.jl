# AP_plots.jl

module ActionPotentialPlots

using Plots, StatsPlots, DataFrames, Printf, LaTeXStrings, Statistics, StaticArrays
# This assumes ActionPotential.jl is in the same directory and defines ActionPotentialModel
# The `..` syntax allows this module to see the one defined outside of it.
using ..ActionPotentialModel 

# Plot all raw experimental traces from a single group file
function plot_raw_traces(df_raw::DataFrame, group_name::String)
    p = plot(title="Raw Traces: $(group_name)", xlabel="Time (ms)", ylabel="Voltage (mV)")
    num_traces = Int(ncol(df_raw) / 2)
    for i in 1:num_traces
        trace = collect(skipmissing(df_raw[:, 2*i]))
        time = collect(skipmissing(df_raw[:, 2*i-1]))
        plot!(p, time, trace, label=names(df_raw)[2*i], lw=1)
    end
    return p
end

# Create box plots of parameter distributions for each experimental group
function plot_param_boxplots(sim_data::DataFrame, params_to_plot::Tuple)
    plots_list = []
    for param in params_to_plot
        p = @df sim_data boxplot(:group, cols(param), 
                                 group=:group, 
                                 legend=false, 
                                 title=string(param))
        push!(plots_list, p)
    end
    return plot(plots_list..., layout=(3, 3), size=(1200, 1000), dpi=150)
end

# Plot the average simulated action potential for each group
function plot_average_traces(sim_data::DataFrame)
    avg_params_by_group = combine(groupby(sim_data, :group), names(sim_data, Real) .=> mean)

    rename!(avg_params_by_group, Dict(old_name => Symbol(replace(String(old_name), "_mean" => "")) for old_name in names(avg_params_by_group)))

    p_avg = plot(title="Average Fitted Action Potential by Group", xlabel="Time (ms)", ylabel="Voltage (mV)")
    
    dummy_time = 0.0:0.02:20.0
    dummy_trace = fill(-70.0, length(dummy_time))
    
    for row in eachrow(avg_params_by_group)
        group_name = row.group
        avg_params = NamedTuple(row)
        
        avg_ap = ActionPotentialModel.ActionPotential(avg_params, dummy_trace, dummy_time, name=group_name)
        plot!(p_avg, avg_ap.time_points, avg_ap.Vs, label=group_name)
    end
    return p_avg
end

# NPlot the decomposed currents for the model
function plot_currents(ap::ActionPotential, group_name::String)
    d = ActionPotentialModel.get_full_trace_details(ap)
    plot(d.t, [d.IKs d.INas d.ILeaks],# d.Istims],
        xlabel="Time (ms)", ylabel="Current (μA/cm²)",
        label=["IK" "INa" "ILeak" "Istim"],
        plot_title="Decomposed Currents ($(group_name))")
end

# Plot the decomposed conductances for the model
function plot_conductances(ap::ActionPotential, group_name::String)
    d = ActionPotentialModel.get_full_trace_details(ap)
    plot(d.t, [d.gKs d.gNas],
        xlabel="Time (ms)", ylabel="Conductance (mS/cm²)",
        label=["gK" "gNa"],
        plot_title="Decomposed Conductances ($(group_name))")
end

# Plot subunit kinetics (steady-state and time constants)
function plot_subunit_kinetics(params::NamedTuple, group_name::String)
    V_range = -100:1:50

    # Calculate steady-state values
    m_inf = ActionPotentialModel.infty_m.(V_range, Ref(params))
    n_inf = ActionPotentialModel.infty_n.(V_range, Ref(params))
    h_inf = ActionPotentialModel.infty_h.(V_range, Ref(params))

    # Calculate time constants
    tau_m = 1 ./ (ActionPotentialModel.alpha_m.(V_range, Ref(params)) .+ ActionPotentialModel.beta_m.(V_range, Ref(params)))
    tau_n = 1 ./ (ActionPotentialModel.alpha_n.(V_range, Ref(params)) .+ ActionPotentialModel.beta_n.(V_range, Ref(params)))
    tau_h = 1 ./ (ActionPotentialModel.alpha_h.(V_range, Ref(params)) .+ ActionPotentialModel.beta_h.(V_range, Ref(params)))

    # Create plots
    p1 = plot(V_range, [m_inf n_inf h_inf],
              title="Steady-State Activation/Inactivation",
              xlabel="Voltage (mV)", ylabel="Probability",
              label=[L"m_{\infty}" L"n_{\infty}" L"h_{\infty}"],
              legend=:right)

    p2 = plot(V_range, [tau_m tau_n tau_h],
              title="Time Constants",
              xlabel="Voltage (mV)", ylabel="Time (ms)",
              label=[L"\tau_m" L"\tau_n" L"\tau_h"],
              legend=:top)
    
    return plot(p1, p2, layout=(2,1), size=(800, 700), dpi=150, plot_title="Subunit Kinetics ($(group_name))")
end

# Plot the alpha and beta rate functions for each subunit
function plot_alpha_beta(params::NamedTuple, group_name::String)
    V_range = -100:1:50
    plot_n = plot(V_range, [ActionPotentialModel.alpha_n.(V_range, Ref(params)) ActionPotentialModel.beta_n.(V_range, Ref(params))], title="n-gate rates", label=["alpha" "beta"])
    plot_m = plot(V_range, [ActionPotentialModel.alpha_m.(V_range, Ref(params)) ActionPotentialModel.beta_m.(V_range, Ref(params))], title="m-gate rates", label=["alpha" "beta"])
    plot_h = plot(V_range, [ActionPotentialModel.alpha_h.(V_range, Ref(params)) ActionPotentialModel.beta_h.(V_range, Ref(params))], title="h-gate rates", label=["alpha" "beta"])
    return plot(plot_n, plot_m, plot_h, layout=(3,1), xlabel="Voltage (mV)", ylabel="Rate (1/ms)", size=(800, 900), dpi=150, plot_title="Alpha/Beta Rates ($(group_name))")
end

# Create pairwise scatter plots for specified parameters to show correlations
function plot_pairwise_params(sim_data::DataFrame, params_to_plot::Tuple)
    plots_list = []
    for i in 1:length(params_to_plot)
        for j in (i+1):length(params_to_plot)
            param1 = params_to_plot[i]
            param2 = params_to_plot[j]
            p = @df sim_data scatter(cols(param1), cols(param2), 
                                     xlabel=string(param1), 
                                     ylabel=string(param2),
                                     group=:group,
                                     alpha=0.5,
                                     legend=false)
            push!(plots_list, p)
        end
    end
    # Arrange plots in a grid
    num_plots = length(plots_list)
    layout_size = ceil(Int, sqrt(num_plots))
    return plot(plots_list..., layout=(layout_size, layout_size), size=(1000, 1000))
end

end # end module