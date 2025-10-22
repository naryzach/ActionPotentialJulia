# workflow_group_trace.jl

using Distributed
if nprocs() < 4; addprocs(max(1, 4 - nprocs())); end

@everywhere include("config.jl")
@everywhere include("ActionPotential.jl")

using .ActionPotentialModel
using CSV, DataFrames, Printf, Dates, Sobol, MixedModels, CategoricalArrays, Plots, StatsPlots, Statistics

@everywhere const opt_par_group_names = (:N_6, :N_7, :M_6, :M_7, :M_1, :M_2, :g_Na, :g_K, :RMP)

function get_fixed_parameter_sets(par_0, fixed_par_names, num_sets)
    range = 0.25
    fixed_par_subset = NamedTuple(k => par_0[k] for k in fixed_par_names)
    s = SobolSeq(length(fixed_par_subset))
    param_sets = DataFrame()
    for i in 1:num_sets
        p_factors = next!(s) .* (2 * range) .+ (1 - range)
        p_varied_vec = values(fixed_par_subset) .* p_factors
        push!(param_sets, (; zip(keys(fixed_par_subset), p_varied_vec)...))
    end
    return param_sets
end

function main_group_trace()
    println("\n--- Starting: Group Analysis Workflow ---")
    trace_files = ["Atratus_WT.csv", "Atratus_P.csv", "Atratus_EPN.csv"]
    group_names = ["WT", "P", "EPN"]
    num_tables = 25
    
    
    # --- Define Output Directories ---
    # A stable directory for the most recent results
    latest_dir = joinpath(output_folder, "Group_Trace", "latest")
    mkpath(latest_dir)

    # A unique, timestamped directory for archiving this specific run
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    archive_dir = joinpath(output_folder, "Group_Trace", "archive_$(timestamp)")
    mkpath(archive_dir)

    all_par_names = keys(par_0)
    fixed_par_names = Tuple(setdiff(Set(all_par_names), Set(opt_par_group_names)))
    sobol_fixed_sets = get_fixed_parameter_sets(par_0, fixed_par_names, num_tables)

    tasks = []
    for tbl_idx in 1:num_tables
        fixed_params = NamedTuple(sobol_fixed_sets[tbl_idx, :])
        for (grp_idx, file) in enumerate(trace_files)
            filepath = joinpath(data_folder, file)
            header_line = readlines(filepath)[4]
            num_cols = length(split(header_line, ','))
            type_map = [Float64 for _ in 1:num_cols]
            df_raw = CSV.read(filepath, DataFrame; header=4, types=type_map, missingstring="")
            for indiv_idx in 1:Int(ncol(df_raw) / 2)
                trace = collect(skipmissing(df_raw[:, 2*indiv_idx]))
                time = collect(skipmissing(df_raw[:, 2*indiv_idx-1]))
                seed_params = merge(par_0, fixed_params)
                push!(tasks, (tbl_idx, grp_idx, indiv_idx, seed_params, trace, time, opt_par_group_names))
            end
        end
    end

    @everywhere function run_group_fit(task)
        tbl, grp, indiv, seed_p, trace, time, opn = task; redirect_stdout(devnull) do
            ap = ActionPotentialModel.ActionPotential(seed_p, trace, time, name="T$tbl-G$grp-I$indiv")
            
            # Run standard optimization
            result = ActionPotentialModel.optimize!(ap, opn)

            #res_dict = Dict{Symbol, Any}(pairs(result["par"]))
            res_dict = Dict{Symbol, Any}(pairs(result["par"]))
            res_dict[:tbl] = tbl
            res_dict[:group_id] = grp
            res_dict[:indiv] = indiv
            #res_dict[:value] = result["value"]
            res_dict[:value] = result["value"]
            return res_dict
        end
    end

    println("Starting $(length(tasks)) optimization jobs across $num_tables tables...")
    all_results = pmap(run_group_fit, tasks)
    sim_data = DataFrame(all_results)
    sim_data.group = [group_names[id] for id in sim_data.group_id]
    CSV.write(joinpath(latest_dir, "All_sim_data.csv"), sim_data)
    println("Full simulation data saved.")

    println("Generating summary plots...")
    for param in opt_par_group_names
        p = @df sim_data boxplot(:group, cols(param), group=:group, legend=false, title="Distribution of $(param)")
        savefig(joinpath(latest_dir, "boxplot_$(param).png"))
    end

    avg_params_by_group = combine(groupby(sim_data, :group), names(sim_data, Real) .=> mean)
    
    p_avg = plot(title="Average Fitted Action Potential by Group", xlabel="Time (ms)", ylabel="Voltage (mV)");
    dummy_time = 0.0:0.02:20.0; dummy_trace = fill(-70.0, length(dummy_time));
    for row in eachrow(avg_params_by_group)
        group_name = row.group
        
        calculated_means = NamedTuple(row)
        full_avg_params = merge(par_0, calculated_means)

        avg_ap = ActionPotentialModel.ActionPotential(full_avg_params, dummy_trace, dummy_time, name=group_name)
        plot!(p_avg, avg_ap.time_points, avg_ap.Vs, label=group_name, lw=2)
    end
    savefig(joinpath(latest_dir, "average_traces.png"))    
    println("Summary plots saved to: ", latest_dir)

    stats_output_path = joinpath(latest_dir, "statistical_summary.txt")
    println("\n--- Running Linear Mixed-Effects Models ---"); 
    
    # Open the file to write the results
    open(stats_output_path, "w") do file_handle
        println(file_handle, "--- Linear Mixed-Effects Model Summary ---")
        println(file_handle, "Generated on: $(now())\n")

        for param in opt_par_group_names
            # Print header to console for real-time feedback
            println("\n--- Analyzing Parameter: $param ---")
            # Write header to the file
            println(file_handle, "\n------------------------------------------")
            println(file_handle, "### Analysis for Parameter: $param ###")
            println(file_handle, "------------------------------------------\n")

            try
                formula = @eval @formula($param ~ 1 + group + (1 + group | tbl))
                model = fit(MixedModel, formula, sim_data)
                
                # Print to console and also write to file
                println(model)
                show(file_handle, model)
                println(file_handle, "\n") # Add spacing in the file

            catch e
                error_msg = "Could not fit model for $param. Error: $e"
                println(error_msg)
                println(file_handle, error_msg)
            end
        end
    end
    println("\nStatistical summary saved to: $(stats_output_path)")

    # --- Archive Results ---
    println("\nArchiving group analysis results...")
    for item in readdir(latest_dir)
        src_path = joinpath(latest_dir, item)
        dest_path = joinpath(archive_dir, item)
        cp(src_path, dest_path, force=true)
    end
    println("Results successfully archived to: $(archive_dir)")
end

