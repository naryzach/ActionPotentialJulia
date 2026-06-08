# setup.jl — one-time dependency installation
#
# Run once from the project root:
#   julia setup.jl
#
# For GPU support (CUDA) also run:
#   julia -e 'using CUDA; CUDA.versioninfo()'
# to confirm CUDA is working after setup.
#
# Note on memory-limited / cluster nodes:
#   Parallel precompilation of the full dependency graph can exhaust RAM and
#   segfault Julia's compiler. This script forces SERIAL precompilation
#   (JULIA_NUM_PRECOMPILE_TASKS=1) to keep peak memory bounded. Override by
#   exporting a higher value before running if your node has plenty of RAM.

using Pkg

# Force serial precompilation unless the caller already chose a value.
# This is the single most effective guard against precompile segfaults on
# memory-constrained machines.
get!(ENV, "JULIA_NUM_PRECOMPILE_TASKS", "1")

println("Installing ActionPotentialJulia dependencies...")
println("Precompilation tasks: $(ENV["JULIA_NUM_PRECOMPILE_TASKS"]) (serial = safest on limited RAM)")
println("This will take several minutes on first run.\n")

packages = [
    # ODE solving
    "DifferentialEquations",
    "DiffEqGPU",
    # Optimisation
    "Optim",
    "BlackBoxOptim",
    # Data
    "DataFrames",
    "CSV",
    # Statistics
    "StatsPlots",
    "MixedModels",
    "CategoricalArrays",
    # Numerics
    "Sobol",
    "QuadGK",
    "StaticArrays",
    # GPU
    "CUDA",
    # Visualisation
    "Plots",
    "LaTeXStrings",
    # Interactive notebook
    "Pluto",
    "PlutoUI",
    # Report generation
    "Weave",
    # CLI
    "ArgParse",
    # Parameter export (interactive_sliders.jl)
    "JSON3",
    # Image handling (report)
    "FileIO",
    "PNGFiles",
    "Images",
    # Cluster support (optional but recommended for server runs)
    "ClusterManagers",
]

# Resolve + install everything in one pass (one resolve is better than many),
# but precompile separately with retries so a transient crash doesn't abort.
Pkg.add(packages)

println("\nPrecompiling (serial) — this is the slow part...")
for attempt in 1:3
    try
        Pkg.precompile()
        break
    catch err
        if attempt == 3
            println("\n⚠ Precompilation still failing after 3 attempts.")
            println("  If you see a segfault, clear stale caches and retry serially:")
            println("    find ~/.julia/compiled -name '*.pidfile' -delete")
            println("    JULIA_NUM_PRECOMPILE_TASKS=1 julia -e 'using Pkg; Pkg.precompile()'")
            rethrow(err)
        end
        println("  Precompile attempt $attempt failed; clearing stale locks and retrying...")
        for (root, _, files) in walkdir(joinpath(DEPOT_PATH[1], "compiled"))
            for f in files
                endswith(f, ".pidfile") && rm(joinpath(root, f); force=true)
            end
        end
    end
end

println("\n✓ All packages installed.")
println("\nNext steps:")
println("  1. Verify CUDA (if using GPU):")
println("       julia -e 'using CUDA; CUDA.versioninfo()'")
println("  2. Open the interactive slider notebook:")
println("       julia -e 'using Pluto; Pluto.run(notebook=\"interactive_sliders.jl\")'")
println("  3. Run the traces workflow:")
println("       julia -t 4 main.jl --workflow traces --cores 4")
println("  4. Run the group analysis:")
println("       julia -t 4 main.jl --workflow group --cores 4")
println("  5. Compile the report:")
println("       julia main.jl --workflow report")
