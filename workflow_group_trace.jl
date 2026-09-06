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

using .ActionPotentialModel
using CSV, DataFrames, Printf, Dates, Sobol, MixedModels, CategoricalArrays
using Plots, StatsPlots, Statistics
# Qualified import only — ProgressMeter exports next!/update!, which would
# otherwise clash with Sobol.next! (used in get_fixed_parameter_sets).
import ProgressMeter

# Parameters optimised in the group workflow (RMP included — free within bounds)
const opt_par_group_names = (:N_6, :N_7, :M_6, :M_7, :M_1, :M_2, :g_Na, :g_K, :RMP)

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
    # CPU: distribute via pmap.  Live bar + ETA over all fits.
    prog        = ProgressMeter.Progress(length(tasks); desc="  Group fits ", showspeed=true)
    ProgressMeter.update!(prog, 0)   # render the bar immediately at 0%
    all_results = use_gpu ? ProgressMeter.progress_map(run_group_fit, tasks; progress=prog) :
                            ProgressMeter.progress_pmap(run_group_fit, tasks; progress=prog)
    sim_data    = DataFrame(all_results)
    sim_data.group = [group_names[id] for id in sim_data.group_id]

    # Unique biological-replicate identifier: `indiv` alone repeats across the
    # three group files (1..N_traces per file), so without qualifying it by
    # group, individual 1 of WT and individual 1 of EPN would be treated as
    # the SAME random-effect level. See the (1 | indiv_id) term added to the
    # LMMs below.
    sim_data.indiv_id = string.(sim_data.group, "_", sim_data.indiv)

    CSV.write(joinpath(latest_dir, "All_sim_data.csv"), sim_data)
    println("Full data saved.")

    # --- Summary plots (robust to NaN/missing: some features, e.g. APD50/AHP,
    #     are NaN when their voltage crossings are not found) ---
    println("Generating parameter boxplots...")
    param_cols = collect(opt_par_group_names)
    feat_cols  = [Symbol(c) for c in names(sim_data) if startswith(String(c), "feat_")]
    for col in vcat(param_cols, feat_cols)
        sub = sim_data[.!ismissing.(sim_data[!, col]) .&
                       isfinite.(coalesce.(sim_data[!, col], NaN)), :]
        if nrow(sub) < 2
            @warn "Skipping boxplot for $col (no finite data)"
            continue
        end
        try
            p = @df sub boxplot(:group, cols(col), group=:group,
                                legend=false, title="$(col) by group")
            savefig(p, joinpath(latest_dir, "boxplot_$(col).png"))
        catch e
            @warn "Boxplot failed for $col: $e"
        end
    end

    # Average-parameter trace overlay (group means over finite values only)
    p_avg = plot(title="Average Fitted AP by Group",
                 xlabel="Time (ms)", ylabel="Voltage (mV)")
    dummy_time  = collect(0.0:0.02:20.0)
    dummy_trace = fill(-70.0, length(dummy_time))
    for g in group_names
        gdf = filter(r -> r.group == g, sim_data)
        nrow(gdf) == 0 && continue
        pmean = Dict{Symbol,Float64}()
        for k in keys(par_0)
            if hasproperty(gdf, k)
                v = filter(isfinite, collect(skipmissing(gdf[!, k])))
                pmean[k] = isempty(v) ? par_0[k] : mean(v)
            end
        end
        avg_ap = ActionPotentialModel.ActionPotential(merge(par_0, NamedTuple(pmean)),
                                                      dummy_trace, dummy_time, name=g)
        plot!(p_avg, avg_ap.time_points, avg_ap.Vs, label=g, lw=2)
    end
    savefig(p_avg, joinpath(latest_dir, "average_traces.png"))

    # --- Linear mixed-effects models ---
    # PRIMARY tests are on the model-free AP FEATURES (esp. feat_max_dvdt, the
    # maximum upstroke velocity), which are robust, identifiable observables.
    # Fitted conductances are reported too, but g_Na is NON-IDENTIFIABLE from the
    # AP waveform (flat profile likelihood — see report.jmd, Critical Findings),
    # so its group effect is unreliable and flagged accordingly.
    stats_path = joinpath(latest_dir, "statistical_summary.txt")
    println("\n--- Fitting Linear Mixed-Effects Models ---")

    feat_cols = [Symbol(c) for c in names(sim_data) if startswith(String(c), "feat_")]
    # Headline first: maximum upstroke velocity (the g_Na proxy).
    sort!(feat_cols; by = c -> (c === :feat_max_dvdt ? "" : String(c)))

    # Each of the N individuals per group is fit 25 times (once per Sobol
    # table) from the SAME experimental trace — those 25 fits are repeated
    # measures of one underlying data point, not 25 independent replicates.
    # Modelling only a per-table random effect (as before) ignores this
    # within-individual correlation entirely: every table×individual cell was
    # treated as an independent observation, giving N_indiv × 25 "samples" per
    # group when there are really only N_indiv independent biological
    # replicates. That inflates the effective sample size and anti-
    # conservatively shrinks standard errors on the group fixed effect — i.e.
    # it can manufacture significant group differences that would not survive
    # correcting for pseudoreplication. Adding (1 | indiv_id) absorbs the
    # per-individual repeated-measures correlation; (1 + group | tbl) is kept
    # because a given table's randomised nuisance-parameter draw can still
    # systematically shift every fit sharing that table.
    function fit_one_lmm(fh, resp)
        println("\n--- $resp ---")
        println(fh, "\n" * "="^60)
        println(fh, "Response: $resp")
        println(fh, "="^60)
        try
            formula = @eval @formula($resp ~ 1 + group + (1 + group | tbl) + (1 | indiv_id))
            model   = fit(MixedModel, formula, sim_data)
            println(model)
            show(fh, model)
            println(fh, "\n")
        catch e
            msg = "Could not fit LMM for $resp: $e"
            println(msg)
            println(fh, msg)
        end
    end

    open(stats_path, "w") do fh
        println(fh, "Linear Mixed-Effects Model Summary")
        println(fh, "Generated: $(now())\n")
        println(fh, "Model:          response ~ 1 + group + (1 + group | tbl) + (1 | indiv_id)")
        println(fh, "Fixed effect:   experimental group (WT, P, EPN)")
        println(fh, "Random effects: optimisation table (initial-condition set);")
        println(fh, "                individual trace (the 25 per-table fits of the SAME")
        println(fh, "                recording are repeated measures, not independent replicates —")
        println(fh, "                omitting this term pseudoreplicates and anti-conservatively")
        println(fh, "                inflates significance of the group effect)\n")

        println(fh, "#"^64)
        println(fh, "# PRIMARY — model-free AP features (identifiable observables).")
        println(fh, "# feat_max_dvdt (max upstroke velocity) is the key g_Na proxy:")
        println(fh, "#   C_m * dV/dt_max ≈ g_Na * m^3 h * (E_Na - V).")
        println(fh, "#"^64)
        for resp in feat_cols
            fit_one_lmm(fh, resp)
        end

        println(fh, "\n" * "#"^64)
        println(fh, "# SECONDARY — fitted HH parameters.")
        println(fh, "# WARNING: g_Na is NON-IDENTIFIABLE from the AP waveform")
        println(fh, "#   (flat profile likelihood; see report.jmd Critical Findings).")
        println(fh, "#   Treat any g_Na group effect as unreliable.")
        println(fh, "#"^64)
        for param in opt_par_group_names
            fit_one_lmm(fh, param)
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
