# workflow_group_trace.jl
#
# Group-level analysis: fits each trace from N randomised initial conditions
# (Sobol sequence) and runs linear mixed-effects models to test for group
# differences in HH parameters.
#
# Parallelism strategy:
#   CPU mode  — pmap distributes one fit per distributed worker
#   GPU mode  — map runs fits sequentially on the main process; the GPU
#               ensemble provides parallelism per fit.

using Distributed
if nprocs() < 4; addprocs(max(1, 4 - nprocs())); end

@everywhere include("config.jl")
@everywhere include("ActionPotential.jl")

using .ActionPotentialModel
using CSV, DataFrames, Printf, Dates, Sobol, MixedModels, CategoricalArrays
using Plots, StatsPlots, Statistics

# Parameters optimised in the group workflow (RMP included — free within bounds)
@everywhere const opt_par_group_names = (:N_6, :N_7, :M_6, :M_7, :M_1, :M_2, :g_Na, :g_K, :RMP)

# ---------------------------------------------------------------------------
# Top-level worker function (must be defined outside any other function to
# avoid Julia 1.12 world-age errors when called from distributed workers).
# ---------------------------------------------------------------------------
@everywhere function run_group_fit(task)
    tbl, grp, indiv, seed_p, bounds, trace, time, opn, use_gpu, num_traj = task
    redirect_stdout(devnull) do
        ap     = ActionPotentialModel.ActionPotential(seed_p, trace, time,
                                                      name="T$tbl-G$grp-I$indiv")
        result = ActionPotentialModel.optimize!(ap, opn; bounds=bounds,
                                                use_gpu=use_gpu, num_trajectories=num_traj)
        feats  = ActionPotentialModel.extract_ap_features(ap)

        res = Dict{Symbol, Any}(pairs(result["par"]))
        res[:tbl]      = tbl
        res[:group_id] = grp
        res[:indiv]    = indiv
        res[:score]    = result["value"]

        if !isnothing(feats)
            for (k, v) in pairs(feats)
                res[Symbol("feat_", k)] = v
            end
        end
        return res
    end
end

# ---------------------------------------------------------------------------
# Generate a table of Sobol-randomised starting values for the *fixed*
# parameters (those not being optimised) across `num_sets` replicates.
# ---------------------------------------------------------------------------
function get_fixed_parameter_sets(par_0, fixed_par_names, num_sets; sobol_range=0.25)
    subset = NamedTuple(k => par_0[k] for k in fixed_par_names)
    s      = SobolSeq(length(subset))
    rows   = DataFrame()
    for _ in 1:num_sets
        factors = next!(s) .* (2 * sobol_range) .+ (1 - sobol_range)
        push!(rows, (; zip(keys(subset), values(subset) .* factors)...))
    end
    return rows
end

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
function main_group_trace(; use_gpu::Bool=false, num_trajectories::Int=100_000)
    println("\n--- Starting: Group Analysis Workflow (GPU=$use_gpu) ---")
    trace_files  = ["Atratus_WT.csv", "Atratus_P.csv", "Atratus_EPN.csv"]
    group_names  = ["WT", "P", "EPN"]
    num_tables   = 25

    latest_dir  = joinpath(output_folder, "Group_Trace", "latest")
    mkpath(latest_dir)
    timestamp   = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    archive_dir = joinpath(output_folder, "Group_Trace", "archive_$(timestamp)")
    mkpath(archive_dir)

    all_par_names   = keys(par_0)
    fixed_par_names = Tuple(setdiff(Set(all_par_names), Set(opt_par_group_names)))
    sobol_sets      = get_fixed_parameter_sets(par_0, fixed_par_names, num_tables)

    # Build task list: (table_idx, group_idx, indiv_idx, params, bounds, trace, time, opt_names, gpu, ntraj)
    tasks = []
    for tbl_idx in 1:num_tables
        fixed_params = NamedTuple(sobol_sets[tbl_idx, :])
        for (grp_idx, file) in enumerate(trace_files)
            filepath    = joinpath(data_folder, file)
            header_line = readlines(filepath)[4]
            num_cols    = length(split(header_line, ','))
            type_map    = [Float64 for _ in 1:num_cols]
            df_raw      = CSV.read(filepath, DataFrame;
                                    header=4, types=type_map, missingstring="")
            for indiv_idx in 1:Int(ncol(df_raw) / 2)
                trace = collect(skipmissing(df_raw[:, 2*indiv_idx]))
                time  = collect(skipmissing(df_raw[:, 2*indiv_idx-1]))
                seed  = merge(par_0, fixed_params)
                push!(tasks, (tbl_idx, grp_idx, indiv_idx, seed, par_bounds, trace, time,
                              opt_par_group_names, use_gpu, num_trajectories))
            end
        end
    end

    println("Running $(length(tasks)) fits across $num_tables tables...")
    # GPU: run on main process — see comment in workflow_read_traces.jl.
    # CPU: distribute via pmap.
    all_results = use_gpu ? map(run_group_fit, tasks) : pmap(run_group_fit, tasks)
    sim_data    = DataFrame(all_results)
    sim_data.group = [group_names[id] for id in sim_data.group_id]

    CSV.write(joinpath(latest_dir, "All_sim_data.csv"), sim_data)
    println("Full data saved.")

    # --- Summary plots ---
    println("Generating parameter boxplots...")
    for param in opt_par_group_names
        p_box = @df sim_data boxplot(:group, cols(param), group=:group,
                                      legend=false, title="$(param) by group")
        savefig(p_box, joinpath(latest_dir, "boxplot_$(param).png"))
    end

    feat_cols = [c for c in names(sim_data) if startswith(String(c), "feat_")]
    for feat in feat_cols
        p_feat = @df sim_data boxplot(:group, cols(Symbol(feat)), group=:group,
                                       legend=false, title="$(feat) by group")
        savefig(p_feat, joinpath(latest_dir, "boxplot_$(feat).png"))
    end

    # Average-parameter trace overlay
    avg = combine(groupby(sim_data, :group), names(sim_data, Real) .=> mean)
    rename!(avg, Dict(old => Symbol(replace(String(old), "_mean" => ""))
                      for old in names(avg)))

    p_avg = plot(title="Average Fitted AP by Group",
                 xlabel="Time (ms)", ylabel="Voltage (mV)")
    dummy_time  = collect(0.0:0.02:20.0)
    dummy_trace = fill(-70.0, length(dummy_time))
    for row in eachrow(avg)
        avg_ap = ActionPotentialModel.ActionPotential(NamedTuple(row), dummy_trace, dummy_time,
                                                       name=row.group)
        plot!(p_avg, avg_ap.time_points, avg_ap.Vs, label=row.group, lw=2)
    end
    savefig(p_avg, joinpath(latest_dir, "average_traces.png"))

    # --- Linear mixed-effects models ---
    stats_path = joinpath(latest_dir, "statistical_summary.txt")
    println("\n--- Fitting Linear Mixed-Effects Models ---")
    open(stats_path, "w") do fh
        println(fh, "Linear Mixed-Effects Model Summary")
        println(fh, "Generated: $(now())\n")
        println(fh, "Model formula:  param ~ 1 + group + (1 + group | tbl)")
        println(fh, "Fixed effect:   experimental group (WT, P, EPN)")
        println(fh, "Random effect:  optimisation table (initial-condition set)\n")

        for param in opt_par_group_names
            println("\n--- $param ---")
            println(fh, "\n" * "="^60)
            println(fh, "Parameter: $param")
            println(fh, "="^60)
            try
                formula = @eval @formula($param ~ 1 + group + (1 + group | tbl))
                model   = fit(MixedModel, formula, sim_data)
                println(model)
                show(fh, model)
                println(fh, "\n")
            catch e
                msg = "Could not fit LMM for $param: $e"
                println(msg)
                println(fh, msg)
            end
        end
    end
    println("Statistical summary saved to $stats_path")

    # --- Archive ---
    println("\nArchiving to $archive_dir ...")
    for item in readdir(latest_dir)
        cp(joinpath(latest_dir, item), joinpath(archive_dir, item); force=true)
    end
    println("Done.")
end
