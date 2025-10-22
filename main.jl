# main.jl

using ArgParse, Distributed, Dates

# --- 1. Load Code First ---
# Including the files here makes their functions available globally
include("workflow_read_traces.jl")
include("workflow_group_trace.jl")
include("workflow_report.jl")


function parse_commandline()
    s = ArgParseSettings(description="Run Action Potential modeling workflows.")
    @add_arg_table s begin
        "--workflow", "-w"
            help = "The workflow to run: 'traces', 'group', or 'report'"
            required = true
        "--cores", "-c"
            help = "Number of cores to use for parallel processing"
            arg_type = Int
            default = 4
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    workflow = args["workflow"]
    cores = args["cores"]

    # Only set up parallel workers if they are needed
    if workflow in ["traces", "group"]
        println("\nSetting up for '$workflow' workflow with $cores cores...")
        # Keep track of workers we add so we can remove them later
        procs_to_add = max(0, cores - 1)
        added_procs = []

        try
            # --- 2. Setup Workers ---
            if procs_to_add > 0
                println("Adding $procs_to_add worker processes...")
                added_procs = addprocs(procs_to_add)
            end
            println("Workers ready: $(nprocs()) total processes.")

            # Load core model code onto all workers
            @everywhere begin
                include("config.jl")
                include("ActionPotential.jl")
            end
            println("Code loaded on all processes.")

            # --- 3. Run Selected Workflow ---
            if workflow == "traces"
                println("\n--- Starting: Individual Traces Workflow ---")
                main_read_traces()
            elseif workflow == "group"
                println("\n--- Starting: Group Analysis Workflow ---")
                main_group_trace()
            elseif workflow == "report"
                println("\n--- Starting: Report Generation Workflow ---")
                main_read_traces()
                main_group_trace()
                main_report()
            end

        catch e
            # This will catch any error during the workflow and print it
            println("\nAn error occurred during execution:")
            showerror(stdout, e, catch_backtrace())
            println()
        finally
            # --- 4. Guaranteed Cleanup ---
            # This block ALWAYS runs, ensuring workers are properly removed.
            if !isempty(added_procs)
                println("\nCleaning up $(length(added_procs)) worker processes...")
                rmprocs(added_procs)
                println("Workers removed.")
            end
        end
    elseif workflow == "report"
        # Report generation does not need parallel workers
        println("\n--- Starting: Report Generation Workflow ---")
        main_report()
    else
        println("Error: Unknown workflow '$workflow'. Choose 'traces', 'group', or 'report'.")
    end

    println("\nMain script finished.")
end

# --- 5. Execute the Main Function ---
main()