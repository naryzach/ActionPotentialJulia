# workflow_read_traces.jl
#
# Fits the Hodgkin-Huxley model to every individual trace in each group file,
# extracts AP features, saves per-trace plots, and writes parameter CSVs.
#
# Parallelism strategy:
#   CPU mode  — pmap distributes one trace per distributed worker
#   GPU mode  — map runs traces sequentially on the main process; the GPU
#               ensemble (100 K+ trajectories) provides parallelism per trace.
#               Running gpu_grid_search! inside a distributed worker causes
#               Julia 1.12 world-age errors with DiffEqGPU closures.

using Distributed
if nprocs() < 4; addprocs(max(1, 4 - nprocs())); end

@everywhere include("config.jl")
@everywhere include("ActionPotential.jl")

using .ActionPotentialModel
using CSV, DataFrames, Printf, Dates, Plots

# ---------------------------------------------------------------------------
# Top-level worker function (must be defined outside any other function to
# avoid Julia 1.12 world-age errors when called from distributed workers).
# ---------------------------------------------------------------------------
@everywhere function fit_trace(task)
    p0, bounds, trace, time, name, opn, use_gpu, num_traj = task
    redirect_stdout(devnull) do
        ap     = ActionPotentialModel.ActionPotential(p0, trace, time, name=name)
        result = ActionPotentialModel.optimize!(ap, opn; bounds=bounds,
                                                use_gpu=use_gpu, num_trajectories=num_traj)
        feats  = ActionPotentialModel.extract_ap_features(ap)
        return (
            name           = name,
            params         = result["par"],
            value          = result["value"],
            convergence    = result["convergence"],
            RMP            = ap.params.RMP,
            final_stim_d   = ap.stim_d,
            final_stim_dim = ap.stim_dim,
            final_tot_wait = ap.tot_wait,
            features       = feats
        )
    end
end

# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
function process_trace_file(file::String, output_dir::String;
                            use_gpu::Bool=false, num_trajectories::Int=100_000)
    filepath    = joinpath(data_folder, file)
    header_line = readlines(filepath)[4]
    num_cols    = length(split(header_line, ','))
    type_map    = [Float64 for _ in 1:num_cols]
    df_raw      = CSV.read(filepath, DataFrame;
                            header=4, types=type_map, missingstring="")

    num_traces = Int(ncol(df_raw) / 2)
    tasks = []
    for i in 1:num_traces
        trace = collect(skipmissing(df_raw[:, 2*i]))
        time  = collect(skipmissing(df_raw[:, 2*i-1]))
        name  = names(df_raw)[2*i]
        push!(tasks, (par_0, par_bounds, trace, time, name, opt_par_names, use_gpu, num_trajectories))
    end

    println("Fitting $(length(tasks)) traces from $file...")
    # GPU: run on main process (closures inside gpu_grid_search! must not cross
    #      into distributed workers — Julia 1.12 world-age restriction).
    # CPU: distribute one trace per worker via pmap.
    results = use_gpu ? map(fit_trace, tasks) : pmap(fit_trace, tasks)

    output_basename = replace(file, ".csv" => "")
    results_df      = DataFrame()

    println("Saving results and plots for $file...")
    for (i, res) in enumerate(results)
        res_dict = Dict{Symbol, Any}(pairs(res.params))
        res_dict[:name]        = res.name
        res_dict[:score]       = res.value
        res_dict[:convergence] = res.convergence
        res_dict[:RMP]         = res.RMP

        if !isnothing(res.features)
            for (k, v) in pairs(res.features)
                res_dict[Symbol("feat_", k)] = v
            end
        end
        push!(results_df, res_dict; cols=:union)

        # Reconstruct AP for plotting using the fitted params
        _, _, orig_trace, orig_time, name, _ = tasks[i]
        plot_ap = ActionPotentialModel.ActionPotential(res.params, orig_trace, orig_time, name=name)
        plot_ap.stim_d   = res.final_stim_d
        plot_ap.stim_dim = res.final_stim_dim
        plot_ap.tot_wait = res.final_tot_wait
        ActionPotentialModel.update_model!(plot_ap, plot_ap.params)

        final_plot    = ActionPotentialModel.create_ap_plot(plot_ap)
        plot_filename = joinpath(output_dir, "$(output_basename)_trace_$(i)_fit.png")
        savefig(final_plot, plot_filename)
    end

    csv_filename = joinpath(output_dir, "$(output_basename)_trace_parameters.csv")
    CSV.write(csv_filename, results_df)
    @printf("Parameters saved to %s\n", csv_filename)
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
function main_read_traces(; use_gpu::Bool=false, num_trajectories::Int=100_000)
    println("\n--- Starting: Individual Traces Workflow (GPU=$use_gpu) ---")
    trace_files = ["Atratus_WT.csv", "Atratus_P.csv", "Atratus_EPN.csv"]

    latest_dir  = joinpath(output_folder, "Trace", "latest")
    mkpath(latest_dir)

    timestamp   = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    archive_dir = joinpath(output_folder, "Trace", "archive_$(timestamp)")
    mkpath(archive_dir)

    for file in trace_files
        process_trace_file(file, latest_dir; use_gpu=use_gpu, num_trajectories=num_trajectories)
    end

    println("\nArchiving results to $archive_dir ...")
    for item in readdir(latest_dir)
        cp(joinpath(latest_dir, item), joinpath(archive_dir, item))
    end
    println("Done.")
end
