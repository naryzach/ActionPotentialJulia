# workflow_read_traces.jl

using Distributed
if nprocs() < 4; addprocs(max(1, 4 - nprocs())); end

# These includes load the code on the main process and all workers
@everywhere include("config.jl")
@everywhere include("ActionPotential.jl")

using .ActionPotentialModel
using CSV, DataFrames, Printf, Dates, Plots

function process_trace_file(file::String, output_dir::String)
    filepath = joinpath(data_folder, file)
    
    header_line = readlines(filepath)[4]
    num_cols = length(split(header_line, ','))
    type_map = [Float64 for _ in 1:num_cols]
    df_raw = CSV.read(filepath, DataFrame; 
                      header=4, 
                      types=type_map,
                      missingstring="")

    num_traces = Int(ncol(df_raw) / 2)
    
    tasks = []
    for i in 1:num_traces
        trace = collect(skipmissing(df_raw[:, 2*i])); time = collect(skipmissing(df_raw[:, 2*i-1]))
        name = names(df_raw)[2*i]
        push!(tasks, (par_0, trace, time, name, opt_par_names))
    end
    
    # This function runs on the worker processes
    @everywhere function fit_trace(task)
        p0, trace, time, name, opn = task
        
        # Use the module name to call the functions on the worker
        redirect_stdout(devnull) do
            ap = ActionPotentialModel.ActionPotential(p0, trace, time, name=name)
            
            # Run standard optimization
            result = ActionPotentialModel.optimize!(ap, opn)
            
            return (
                name=name, params=result["par"], value=result["value"],
                convergence=result["convergence"], RMP=ap.params.RMP,
                final_stim_d=ap.stim_d, final_stim_dim=ap.stim_dim, final_tot_wait=ap.tot_wait
            )
        end
    end
    
    println("Starting full optimization for $(length(tasks)) traces in $file...")
    results = pmap(fit_trace, tasks)
    
    output_basename = replace(file, ".csv" => "")
    results_df = DataFrame()
    
    println("Saving results and generating plots for $file...")
    for (i, res) in enumerate(results)
        # Save numerical data to the DataFrame
        res_dict = Dict{Symbol, Any}(pairs(res.params))
        res_dict[:name] = res.name
        res_dict[:value] = res.value
        res_dict[:RMP] = res.RMP
        res_dict[:convergence] = res.convergence
        push!(results_df, res_dict, cols=:union)

        # Generate and save a plot for this specific trace
        _, original_trace, original_time, name, _ = tasks[i]
        plot_ap = ActionPotentialModel.ActionPotential(res.params, original_trace, original_time, name=name)
        plot_ap.stim_d = res.final_stim_d
        plot_ap.stim_dim = res.final_stim_dim
        plot_ap.tot_wait = res.final_tot_wait

        # Re-simulate the model with the correct stimulus timing before plotting
        ActionPotentialModel.update_model!(plot_ap, plot_ap.params) 

        final_plot = ActionPotentialModel.create_ap_plot(plot_ap)

        plot_filename = joinpath(output_dir, "$(output_basename)_trace_$(i)_fit.png")
        savefig(final_plot, plot_filename)
    end
    
    # Save the complete DataFrame to a CSV file
    csv_filename = joinpath(output_dir, "$(output_basename)_trace_parameters.csv")
    CSV.write(csv_filename, results_df)
    @printf("Saved parameters to %s\n", csv_filename)
end

function main_read_traces()
    println("\n--- Starting: Individual Traces Workflow ---")
    trace_files = ["Atratus_WT.csv", "Atratus_P.csv", "Atratus_EPN.csv"]
    
    # --- Define Output Directories ---
    latest_dir = joinpath(output_folder, "Trace", "latest")
    mkpath(latest_dir)

    # A unique, timestamped directory for archiving this specific run
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    archive_dir = joinpath(output_folder, "Trace", "archive_$(timestamp)")
    mkpath(archive_dir)
    
    # --- Run Analysis ---
    # The analysis is always performed on the "latest" directory
    for file in trace_files
        process_trace_file(file, latest_dir)
    end

    # --- Archive Results ---
    println("\nArchiving results...")
    # Copy the contents of the 'latest' directory to the timestamped archive folder
    for item in readdir(latest_dir)
        src_path = joinpath(latest_dir, item)
        dest_path = joinpath(archive_dir, item)
        cp(src_path, dest_path)
    end
    println("Results successfully archived to: $(archive_dir)")
end