# config.jl

using Printf
using CSV, DataFrames

# Initial parameter values as a NamedTuple
const par_0 = (
    # Membrane Properties
    RMP = -70.0, # arbitarily set, will be updated from data

    # Sodium Channel (Na⁺) - m & h gates
    M_1 = 0.523, M_2 = -69.7, M_6 = -17.6, M_7 = 30.8,
    H_3 = -21.1, H_4 = 68.3, H_5 = -3.67, H_6 = 161.5,

    # Potassium Channel (K⁺) - n gate
    N_1 = 0.004, N_2 = -119.0, N_6 = -23.0, N_7 = 64.5,

    # Conductances (g)
    g_Na = 120.0, g_K = 36.0, g_Leak = 0.3
)

# Subset of parameters to be optimized by default
#const opt_par_names = (:N_1, :N_2, :N_6, :N_7, :M_1, :M_2, :M_6, :M_7, :H_3, :H_4, :H_5, :H_6, :g_Na, :g_K)
const opt_par_names = (:N_7, :N_6, :M_6, :M_7, :M_1, :M_2, :g_Na, :g_K)

# Define file paths
const data_folder = joinpath(pwd(), "Cleaned_Data")
const output_folder = joinpath(pwd(), "Parameter_Files")
const notebook_folder = joinpath(pwd(), "Notebook")

# Ensure output directories exist
mkpath(output_folder)
mkpath(notebook_folder)

println("Configuration loaded.")
@printf("Data folder: %s\n", data_folder)
@printf("Output folder: %s\n", output_folder)