# config.jl

using Printf
using CSV, DataFrames

# ---------------------------------------------------------------------------
# Default Hodgkin-Huxley model parameters (absolute membrane voltage in mV).
#
# Rate functions (ms⁻¹) are expressed as:
#   alpha_m(V) = M_1 * smooth_max_zero(V - M_2)
#   beta_m(V)  = exp((V + M_7) / M_6)
#   alpha_h(V) = H_1 * exp((V + H_6) / H_3)
#   beta_h(V)  = 1 / (1 + exp((V + H_4) / H_5))
#   alpha_n(V) = N_1 * smooth_max_zero(V - N_2)
#   beta_n(V)  = exp((V + N_7) / N_6)
#
# H parameters are calibrated so that at rest (≈ −70 mV):
#   infty_h ≈ 0.75  (Na channels ≈ 75% de-inactivated at rest)
#   tau_h   ≈ 9 ms  (physiological inactivation time constant)
# This matches the standard Hodgkin-Huxley formulation converted to absolute
# voltage with V_rest = −65 mV.
# ---------------------------------------------------------------------------
const par_0 = (
    # Membrane properties
    RMP    = -70.0,   # mV — overwritten from data in ActionPotential constructor

    # Sodium channel — m gate (activation)
    M_1 = 0.523,  M_2 = -69.7,  M_6 = -17.6,  M_7 = 30.8,

    # Sodium channel — h gate (inactivation)
    # Standard HH values converted to absolute voltage (V_rest = −65 mV):
    #   alpha_h(V) = 0.07 * exp((V + 65) / (−20))
    #   beta_h(V)  = 1 / (1 + exp((V + 35) / (−10)))
    H_1 = 0.07,  H_3 = -20.0,  H_4 = 35.0,  H_5 = -10.0,  H_6 = 65.0,

    # Potassium channel — n gate (activation)
    N_1 = 0.004,  N_2 = -119.0,  N_6 = -23.0,  N_7 = 64.5,

    # Maximum conductances (mS/cm²)
    g_Na = 120.0,  g_K = 36.0,  g_Leak = 0.3
)

# ---------------------------------------------------------------------------
# Physiological parameter bounds (lower, upper).
# Passed to the optimizer to prevent non-physiological solutions.
# ---------------------------------------------------------------------------
const par_bounds = (
    RMP    = (-95.0,  -40.0),
    M_1    = ( 0.001,   5.0),
    M_2    = (-120.0, -30.0),
    M_6    = ( -40.0,  -3.0),
    M_7    = (   5.0,  80.0),
    H_1    = ( 0.001,   1.0),
    H_3    = ( -40.0,  -3.0),
    H_4    = (  10.0,  80.0),
    H_5    = ( -25.0,  -2.0),
    H_6    = (  20.0, 130.0),
    N_1    = ( 1e-4,   0.1),
    N_2    = (-160.0, -60.0),
    N_6    = ( -45.0,  -5.0),
    N_7    = (  20.0, 120.0),
    g_Na   = (  10.0, 600.0),
    g_K    = (   1.0, 200.0),
    g_Leak = (  0.01,   5.0)
)

# Parameters optimized in the default individual-trace workflow.
# H_* are fixed at their default values here; include them for a full fit.
const opt_par_names = (:N_7, :N_6, :M_6, :M_7, :M_1, :M_2, :g_Na, :g_K)

# File paths
const data_folder     = joinpath(pwd(), "Cleaned_Data")
const output_folder   = joinpath(pwd(), "Parameter_Files")
const notebook_folder = joinpath(pwd(), "Notebook")

mkpath(output_folder)
mkpath(notebook_folder)

println("Configuration loaded.")
@printf("Data folder:   %s\n", data_folder)
@printf("Output folder: %s\n", output_folder)
