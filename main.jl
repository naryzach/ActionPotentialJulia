# main.jl
#
# Entry point for the Action Potential analysis pipeline.
#
# ── CPU-only (local, 8 cores) ──────────────────────────────────────────────
#   julia -t 8 main.jl --workflow traces --cores 8
#
# ── GPU-accelerated (local or server) ─────────────────────────────────────
#   julia -t 8 main.jl --workflow traces --cores 8 --gpu --trajectories 500000
#
# ── Large server run (64 cores, 500 K GPU trajectories) ───────────────────
#   julia -t 64 main.jl --workflow group --cores 64 --gpu --trajectories 500000
#
# Thread count (-t N) controls @threads parallelism (foot-finding, profile
# likelihood, etc.).  --cores N controls distributed worker count for pmap
# (one independent trace fit per worker).  Both are useful simultaneously:
#   --cores = N_workers for pmap
#   -t N    = threads per worker (default 1 unless you set JULIA_NUM_THREADS)
#
# Human-in-the-loop parameters:
#   Run interactive_sliders.jl in Pluto, click "Export Parameters", and the
#   resulting initial_params.json is loaded automatically here.

using ArgParse, Distributed, Dates, JSON3

include("workflow_read_traces.jl")
include("workflow_group_trace.jl")
include("workflow_report.jl")

# ---------------------------------------------------------------------------
# Load optional user-tuned initial parameters (from interactive_sliders.jl)
# ---------------------------------------------------------------------------
function maybe_load_user_params(base_params::NamedTuple)
    json_path = joinpath(pwd(), "initial_params.json")
    isfile(json_path) || return base_params
    println("Found initial_params.json — overriding defaults with user-tuned parameters.")
    raw    = JSON3.read(read(json_path, String))
    merged = Dict{Symbol, Any}()
    for (k, v) in pairs(raw)
        k_sym = Symbol(k)
        if !startswith(String(k_sym), "_") && haskey(base_params, k_sym)
            merged[k_sym] = Float64(v)
        end
    end
    return merge(base_params, NamedTuple(merged))
end

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
function parse_commandline()
    s = ArgParseSettings(description="Run Action Potential modelling workflows.")
    @add_arg_table s begin
        "--workflow", "-w"
            help     = "Workflow: 'traces', 'group', or 'report'"
            required = true
        "--cores", "-c"
            help     = "Distributed worker processes for pmap (one trace per worker)"
            arg_type = Int
            default  = 4
        "--gpu"
            help     = "Replace BlackBoxOptim global search with GPU grid search (requires CUDA)"
            action   = :store_true
        "--trajectories"
            help     = "Number of GPU trajectories per trace (default 100_000; use 500_000+ on server GPU)"
            arg_type = Int
            default  = 100_000
    end
    return parse_args(s)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
function main()
    args         = parse_commandline()
    workflow     = args["workflow"]
    cores        = args["cores"]
    use_gpu      = args["gpu"]
    num_traj     = args["trajectories"]

    if use_gpu
        println("GPU mode enabled — will use CUDA for global search ($(num_traj) trajectories/trace).")
        println("Julia threads available: $(Threads.nthreads())")
        println("Note: start Julia with -t N to set thread count for @threads parallelism.")
    end

    if workflow in ["traces", "group"]
        procs_to_add = max(0, cores - 1)
        added_procs  = Int[]

        try
            if procs_to_add > 0
                println("Adding $procs_to_add distributed worker processes...")
                added_procs = addprocs(procs_to_add)
            end
            println("Workers ready: $(nprocs()) total.  Julia threads: $(Threads.nthreads())")

            @everywhere begin
                include("config.jl")
                include("ActionPotential.jl")
            end
            println("Code loaded on all processes.")

            if workflow == "traces"
                main_read_traces(; use_gpu=use_gpu, num_trajectories=num_traj)
            elseif workflow == "group"
                main_group_trace(; use_gpu=use_gpu, num_trajectories=num_traj)
            end

        catch e
            println("\nError during execution:")
            showerror(stdout, e, catch_backtrace())
            println()
        finally
            if !isempty(added_procs)
                println("\nRemoving $(length(added_procs)) worker processes...")
                rmprocs(added_procs)
            end
        end

    elseif workflow == "report"
        println("\n--- Starting: Report Generation ---")
        main_report()
    else
        println("Unknown workflow '$workflow'. Choose: traces | group | report")
    end

    println("\nFinished.")
end

main()
