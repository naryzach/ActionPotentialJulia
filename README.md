# ActionPotentialJulia

High-performance Hodgkin-Huxley fitting of *Thamnophis* muscle action potentials in Julia,
with the goal of extracting ion channel parameters — particularly maximum sodium conductance ($\bar{g}_{Na}$) —
from individual electrophysiology recordings.

## Scientific Goal

TTX-resistant garter snakes (*Thamnophis sirtalis*) carry amino acid substitutions in the skeletal-muscle
Nav1.4 sodium channel that reduce tetrodotoxin binding.  Three genotypes are studied:

| Label | Mutation | TTX resistance |
|-------|----------|---------------|
| WT    | Wild type | Sensitive |
| P     | Single point mutant | Moderate |
| EPN   | Triple point mutant | High |

The goal is to determine whether the HH parameters extracted from an AP trace — especially $\bar{g}_{Na}$ —
are different enough across genotypes to (a) classify an unknown recording and (b) diagnose the molecular
phenotype of a muscle cell from its AP alone.

## Project Structure

```
ActionPotentialJulia/
├── ActionPotential.jl       # Core HH model, ODE solver, optimiser, feature extraction
├── action_potential_plots.jl# Visualisation module
├── config.jl                # Default parameters, physiological bounds, file paths
├── main.jl                  # CLI entry point
├── workflow_read_traces.jl  # Individual-trace fitting workflow
├── workflow_group_trace.jl  # Multi-start group analysis workflow
├── workflow_report.jl       # PDF report generation (Weave.jl)
├── interactive_sliders.jl   # Pluto.jl notebook for interactive parameter tuning
├── report.jmd               # Weave source for the full analysis report
└── Cleaned_Data/
    ├── Atratus_WT.csv
    ├── Atratus_P.csv
    └── Atratus_EPN.csv
```

## Quick Start

### 1. Interactive parameter tuning (recommended first step)

Open the Pluto notebook to explore how HH parameters affect the simulated AP shape and
to set physiologically sensible starting conditions:

```bash
julia -e 'using Pluto; Pluto.run(notebook="interactive_sliders.jl")'
```

Adjust sliders in real time.  The right panel shows steady-state gating curves and time
constants alongside the simulated AP — watch $h_\infty(\text{RMP})$ to confirm the h-gate
is properly de-inactivated at rest (~0.7–0.8).  Click **Export Parameters** to save your
parameter set to `initial_params.json`; the workflows will load it automatically.

### 2. Fit individual traces

```bash
julia main.jl --workflow traces --cores 8
```

Outputs (in `Parameter_Files/Trace/latest/`):
- `*_trace_parameters.csv` — fitted parameters + AP features for each trace
- `*_trace_N_fit.png` — overlay plots of model vs experimental trace

### 3. Group-level analysis

```bash
julia main.jl --workflow group --cores 8
```

Outputs (in `Parameter_Files/Group_Trace/latest/`):
- `All_sim_data.csv` — all fitted parameters from 25 × N_traces optimisation runs
- `boxplot_*.png` — parameter distributions by genotype group
- `statistical_summary.txt` — linear mixed-effects model results

### 4. Generate the report

```bash
julia main.jl --workflow report
```

Produces a PDF in `Notebook/` (requires the traces and group workflows to have been run first).

## Model Description

The Hodgkin-Huxley ODE system uses the following rate functions (absolute voltage convention):

```
alpha_m(V) = M_1 · softplus(V − M_2)        beta_m(V) = exp((V + M_7) / M_6)
alpha_h(V) = H_1 · exp((V + H_6) / H_3)     beta_h(V) = 1 / (1 + exp((V + H_4) / H_5))
alpha_n(V) = N_1 · softplus(V − N_2)         beta_n(V) = exp((V + N_7) / N_6)
```

Default h-gate parameters are calibrated to physiological HH kinetics:
- $h_\infty(-70\,\text{mV}) \approx 0.76$ (Na channels ≈ 76% de-inactivated at rest)
- $\tau_h(-70\,\text{mV}) \approx 9\,\text{ms}$

## Optimisation Pipeline

1. **Foot finding** — constant-charge method fits the AP upstroke foot (stimulus shape) before the main optimisation.
2. **Global search** — BlackBoxOptim (Differential Evolution, 500 000 evals) within physiological bounds.
3. **Local refinement** — Nelder-Mead with a bounds-penalty term (5 000 iterations).
4. **Profile likelihood** — `profile_likelihood_gNa()` sweeps $\bar{g}_{Na}$ to confirm identifiability.

## Key API

```julia
include("config.jl")
include("ActionPotential.jl")
using .ActionPotentialModel

# Create and fit an AP object
ap = ActionPotential(par_0, trace, time; name="my_trace")
result = optimize!(ap, opt_par_names; bounds=par_bounds)

# Extract features
feats = extract_ap_features(ap)
# feats.max_dvdt  — max upstroke rate (proxy for peak I_Na)
# feats.V_peak    — peak depolarisation
# feats.APD50     — action potential duration at 50% repolarisation

# Test g_Na identifiability
profile = profile_likelihood_gNa(ap, opt_par_names; n_points=25, bounds=par_bounds)
```

## Dependencies

Install all dependencies with:

```julia
using Pkg; Pkg.instantiate()
```

Key packages: `DifferentialEquations`, `BlackBoxOptim`, `Optim`, `MixedModels`,
`Sobol`, `Plots`, `StatsPlots`, `Pluto`, `PlutoUI`, `Weave`, `JSON3`, `CUDA`, `DiffEqGPU`.

## References

1. Hodgkin & Huxley (1952) *J Physiol* 117:500–544
2. Brodie & Brodie (1990) *Evolution* 44:651–659
3. Geffeney et al. (2005) *Science* 307:1640–1642
4. Feldman et al. (2010) *J Evol Biol* 23:2624–2634
