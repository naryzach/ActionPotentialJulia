# setup.jl — one-time dependency installation
#
# Run once from the project root:
#   julia setup.jl
#
# For GPU support (CUDA) also run:
#   julia -e 'using CUDA; CUDA.versioninfo()'
# to confirm CUDA is working after setup.

using Pkg

println("Installing ActionPotentialJulia dependencies...")
println("This will take a few minutes on first run.\n")

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

Pkg.add(packages)

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
