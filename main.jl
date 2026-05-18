# main.jl
#
# Entry point for the Action Potential analysis pipeline.
#
# Usage:
#   julia main.jl --workflow traces  [--cores N]
#   julia main.jl --workflow group   [--cores N]
#   julia main.jl --workflow report
#
# Human-in-the-loop initial parameters:
#   Run the Pluto notebook `interactive_sliders.jl` to tune starting parameters
#   and export them to `initial_params.json`.  If that file is present in the
#   project directory, the workflows load it as the initial parameter set
#   instead of the built-in par_0 defaults.

using ArgParse, Distributed, Dates, JSON3

include("workflow_read_traces.jl")
include("workflow_group_trace.jl")
include("workflow_report.jl")

# ---------------------------------------------------------------------------
# Load optional user-supplied initial parameters from interactive_sliders.jl
# ---------------------------------------------------------------------------
function maybe_load_user_params(base_params::NamedTuple)
    json_path = joinpath(pwd(), "initial_params.json")
    if !isfile(json_path)
        return base_params
    end
    println("Found initial_params.json — overriding defaults with user-tuned parameters.")
    raw    = JSON3.read(read(json_path, String))
    # Only keep keys that are actual model parameters (skip _stim_* prefixed keys)
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
# CLI
# ---------------------------------------------------------------------------
function parse_commandline()
    s = ArgParseSettings(description="Run Action Potential modelling workflows.")
    @add_arg_table s begin
        "--workflow", "-w"
            help     = "Workflow: 'traces', 'group', or 'report'"
            required = true
        "--cores", "-c"
            help     = "Number of parallel worker processes"
            arg_type = Int
            default  = 4
    end
    return parse_args(s)
end

function main()
    args     = parse_commandline()
    workflow = args["workflow"]
    cores    = args["cores"]

    if workflow in ["traces", "group"]
        println("\nSetting up '$workflow' workflow with $cores cores...")
        procs_to_add = max(0, cores - 1)
        added_procs  = Int[]

        try
            if procs_to_add > 0
                println("Adding $procs_to_add worker processes...")
                added_procs = addprocs(procs_to_add)
            end
            println("Workers ready: $(nprocs()) total processes.")

            @everywhere begin
                include("config.jl")
                include("ActionPotential.jl")
            end
            println("Code loaded on all processes.")

            if workflow == "traces"
                main_read_traces()
            elseif workflow == "group"
                main_group_trace()
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
