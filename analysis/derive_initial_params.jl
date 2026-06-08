# derive_initial_params.jl
#
# Re-derive baseline HH parameters by fitting the AVERAGE wild-type AP — the
# way the original R analysis seeded par_0 — but now with: the de-collapsed
# (anchored) foot finder, robust pre-foot RMP, parameter-scaled local refine,
# and the current rate equations. Produces a fresh par_0 candidate to compare
# against the Hodgkin-Huxley-paper-derived defaults.
#
#   julia derive_initial_params.jl [max_evals]
#
# Writes derived_WT_fit.png and derived_par_0.json.

const _ROOT = dirname(@__DIR__)   # scripts live in analysis/; resolve repo files from root
cd(_ROOT)
include(joinpath(_ROOT, "config.jl"))
include(joinpath(_ROOT, "ActionPotential.jl"))
using .ActionPotentialModel
using CSV, DataFrames, Printf, Plots, JSON3, Dates

max_evals = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000

# --- Load the averaged WT trace (same source vars.R used: WT_avg_10ms) ---
df   = CSV.read(joinpath(data_folder, "AP_trace.csv"), DataFrame)
time = collect(skipmissing(df[:, 11]))   # Time_ms  (col 11)
volt = collect(skipmissing(df[:, 12]))   # WT_avg_10ms (col 12)
n    = min(length(time), length(volt))
time, volt = time[1:n], volt[1:n]

# --- Free all channel-shape params + conductances (including h-gate) ---
derive_opt = (:M_1, :M_2, :M_6, :M_7,            # Na activation (m)
              :H_1, :H_3, :H_4, :H_5, :H_6,      # Na inactivation (h)
              :N_1, :N_2, :N_6, :N_7,            # K activation (n)
              :g_Na, :g_K, :g_Leak)             # conductances

ap = ActionPotentialModel.ActionPotential(par_0, volt, time, name="WT_average")
@printf("Average WT trace: RMP=%.2f mV  dt=%.3f ms  n=%d  peak=%.2f mV\n",
        ap.params.RMP, ap.dt, length(volt), maximum(volt))

# --- Staged fit: anchored foot → global search → scaled local refine ---
ActionPotentialModel.find_foot!(ap)
@printf("Stimulus: onset(tot_wait)=%.3f ms  d=%.3f  dim=%.2f  h=%.1f\n",
        ap.tot_wait, ap.stim_d, ap.stim_dim, ap.stim_h)

gr = ActionPotentialModel.global_optimize(ap, derive_opt;
                                          max_evals=max_evals, range=0.9, bounds=par_bounds)
ActionPotentialModel.update_model!(ap, gr["par"])
fr = ActionPotentialModel.optimize_model(ap, derive_opt; bounds=par_bounds)
ActionPotentialModel.update_model!(ap, fr["par"])

# --- Report: derived vs current Julia par_0 ---
println("\n=== Derived par_0 (fit to average WT) ===")
@printf("%-8s %12s %12s %10s\n", "param", "derived", "par_0", "ratio")
for k in derive_opt
    d = fr["par"][k]; p = get(par_0, k, NaN)
    @printf("%-8s %12.5f %12.5f %10.3f\n", k, d, p, d/p)
end
@printf("\nFinal fit score (SSE): %.5g\n", fr["value"])

# --- Save plot + JSON ---
savefig(ActionPotentialModel.create_ap_plot(ap), "derived_WT_fit.png")
open("derived_par_0.json", "w") do io
    payload = Dict{String,Any}("_note" => "Derived from average WT on $(now())",
                               "RMP" => ap.params.RMP)
    for k in keys(fr["par"]); payload[string(k)] = fr["par"][k]; end
    JSON3.pretty(io, payload)
end
println("Saved derived_WT_fit.png and derived_par_0.json")
