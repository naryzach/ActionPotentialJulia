### A Pluto.jl notebook ###
# v0.20.17

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ f7312cb2-8a05-11f0-2004-890435c0dfe8
begin
	import Pkg
	Pkg.activate(pwd())

	using Plots
	using PlutoUI
	using JSON3
	using LaTeXStrings

	include("config.jl")
	include("ActionPotential.jl")
	using .ActionPotentialModel
end

# ╔═╡ intro-cell
md"""
# Interactive Action Potential Explorer

Adjust the sliders to tune Hodgkin-Huxley parameters in real time.
The left panel shows the simulated AP; the right panel shows the steady-state
gating-variable curves and time constants that result from your current parameters.

Use the **Export Parameters** button to save your parameter set to
`initial_params.json` — the main workflows will load it automatically if
it is present in the project directory.

> **Tip:** Start with conductances (g\_Na, g\_K) and the m-gate threshold (M\_2)
> to get an AP firing, then refine the shape with the kinetic parameters.
"""

# ╔═╡ d94f4e1e-c3b0-4826-a4fe-25b95e6c7370
@bind params_interactive PlutoUI.combine() do Child
	md"""
	### Tune Action Potential Parameters
	---
	#### Simulation & Stimulus
	**Resting Potential RMP (mV):** $(Child(Slider(-90.0:1.0:-45.0, default=-70.0, show_value=true)))
	**Stim Height h (μA/cm²):**    $(Child(Slider(5.0:0.5:120.0, default=55.0, show_value=true)))
	**Stim Duration d (ms):**      $(Child(Slider(0.1:0.05:2.0, default=0.64, show_value=true)))
	**Stim Shape dim:**            $(Child(Slider(1.0:0.1:4.0, default=2.0, show_value=true)))
	**Stabilisation time (ms):**   $(Child(Slider(5.0:1.0:20.0, default=10.0, show_value=true)))
	---
	#### Conductances (mS/cm²)
	**g\_Na:** $(Child(Slider(10.0:1.0:400.0, default=par_0.g_Na, show_value=true)))
	**g\_K:**  $(Child(Slider(1.0:0.5:150.0,  default=par_0.g_K,  show_value=true)))
	**g\_Leak:** $(Child(Slider(0.0:0.01:2.0, default=par_0.g_Leak, show_value=true)))
	---
	#### Na⁺ m-gate (activation)
	**M\_1 (scale):** $(Child(Slider(0.01:0.01:2.0,  default=par_0.M_1, show_value=true)))
	**M\_2 (threshold mV):** $(Child(Slider(-120.0:0.5:-30.0, default=par_0.M_2, show_value=true)))
	**M\_6 (β slope):**  $(Child(Slider(-40.0:0.5:-3.0, default=par_0.M_6, show_value=true)))
	**M\_7 (β offset):** $(Child(Slider(5.0:0.5:80.0,  default=par_0.M_7, show_value=true)))
	---
	#### Na⁺ h-gate (inactivation)
	**H\_1 (α scale):** $(Child(Slider(0.001:0.001:0.5,  default=par_0.H_1, show_value=true)))
	**H\_3 (α slope):** $(Child(Slider(-40.0:0.5:-3.0,   default=par_0.H_3, show_value=true)))
	**H\_4 (β offset mV):** $(Child(Slider(10.0:1.0:80.0, default=par_0.H_4, show_value=true)))
	**H\_5 (β slope):** $(Child(Slider(-25.0:0.5:-2.0,   default=par_0.H_5, show_value=true)))
	**H\_6 (α offset mV):** $(Child(Slider(20.0:1.0:130.0, default=par_0.H_6, show_value=true)))
	---
	#### K⁺ n-gate (activation)
	**N\_1 (scale):**    $(Child(Slider(0.0001:0.0001:0.05, default=par_0.N_1, show_value=true)))
	**N\_2 (threshold mV):** $(Child(Slider(-160.0:1.0:-60.0, default=par_0.N_2, show_value=true)))
	**N\_6 (β slope):**  $(Child(Slider(-45.0:0.5:-5.0, default=par_0.N_6, show_value=true)))
	**N\_7 (β offset):** $(Child(Slider(20.0:1.0:120.0, default=par_0.N_7, show_value=true)))
	"""
end

# ╔═╡ 51466345-d9d5-4b3e-9cd6-cae3ff3289bc
begin
	RMP_val, stim_h_val, stim_d_val, stim_dim_val, stabil_time_val,
	g_Na_val, g_K_val, g_Leak_val,
	M_1_val, M_2_val, M_6_val, M_7_val,
	H_1_val, H_3_val, H_4_val, H_5_val, H_6_val,
	N_1_val, N_2_val, N_6_val, N_7_val = params_interactive

	full_sim_params = merge(par_0, (
		RMP      = RMP_val,
		stim_h   = stim_h_val,
		stim_d   = stim_d_val,
		stim_dim = stim_dim_val,
		tot_wait = stabil_time_val,
		g_Na     = g_Na_val,  g_K = g_K_val,  g_Leak = g_Leak_val,
		M_1      = M_1_val,   M_2 = M_2_val,  M_6 = M_6_val,  M_7 = M_7_val,
		H_1      = H_1_val,   H_3 = H_3_val,  H_4 = H_4_val,
		H_5      = H_5_val,   H_6 = H_6_val,
		N_1      = N_1_val,   N_2 = N_2_val,  N_6 = N_6_val,  N_7 = N_7_val
	))

	sim_time    = 30.0
	dt_slider   = 0.02
	time_points = collect(0.0:dt_slider:sim_time)

	simulated_Vs = ActionPotentialModel._simulate_trace(full_sim_params, time_points, dt_slider)
	peak_voltage = maximum(simulated_Vs)
	fired        = peak_voltage > (RMP_val + 20)   # crude AP detection

	# --- Left panel: simulated AP ---
	p_ap = plot(time_points, simulated_Vs,
		label    = "V (mV)",
		lw       = 2,
		color    = :steelblue,
		xlabel   = "Time (ms)",
		ylabel   = "Voltage (mV)",
		ylims    = (-100, 60),
		legend   = :topright,
		title    = "Simulated AP  |  Peak: $(round(peak_voltage, digits=1)) mV" *
		            (fired ? "" : "  ⚠ no AP")
	)

	# --- Right panel: steady-state gating curves ---
	V_range = -100:1:60
	m_inf = ActionPotentialModel.infty_m.(V_range, Ref(full_sim_params))
	n_inf = ActionPotentialModel.infty_n.(V_range, Ref(full_sim_params))
	h_inf = ActionPotentialModel.infty_h.(V_range, Ref(full_sim_params))
	tau_m = 1 ./ (ActionPotentialModel.alpha_m.(V_range, Ref(full_sim_params)) .+
	               ActionPotentialModel.beta_m.(V_range,  Ref(full_sim_params)))
	tau_n = 1 ./ (ActionPotentialModel.alpha_n.(V_range, Ref(full_sim_params)) .+
	               ActionPotentialModel.beta_n.(V_range,  Ref(full_sim_params)))
	tau_h = 1 ./ (ActionPotentialModel.alpha_h.(V_range, Ref(full_sim_params)) .+
	               ActionPotentialModel.beta_h.(V_range,  Ref(full_sim_params)))

	p_inf = plot(V_range, [m_inf n_inf h_inf],
		label   = [L"m_\infty" L"n_\infty" L"h_\infty"],
		xlabel  = "V (mV)", ylabel = "Probability",
		title   = "Steady-state gating",
		lw = 2, ylims = (0, 1), legend = :right)
	vline!(p_inf, [RMP_val], ls=:dash, color=:grey, lw=1, label="RMP")

	# Display infty_h at rest as a diagnostic
	h_at_rest = ActionPotentialModel.infty_h(RMP_val, full_sim_params)

	p_tau = plot(V_range, [tau_m tau_n tau_h],
		label  = [L"\tau_m" L"\tau_n" L"\tau_h"],
		xlabel = "V (mV)", ylabel = "τ (ms)",
		title  = "Time constants  |  h∞(RMP) = $(round(h_at_rest, digits=3))",
		lw = 2, ylims = (0, min(50, maximum([tau_m; tau_n; tau_h]) * 1.1)),
		legend = :top)

	plot(p_ap, p_inf, p_tau, layout=(1, 3), size=(1400, 400), dpi=120)
end

# ╔═╡ export-cell
@bind export_button Button("💾 Export current parameters to initial_params.json")

# ╔═╡ export-action
begin
	export_button  # triggers re-run when button is clicked

	export_dict = Dict(
		"RMP"    => RMP_val,
		"g_Na"   => g_Na_val,  "g_K" => g_K_val,  "g_Leak" => g_Leak_val,
		"M_1"    => M_1_val,   "M_2" => M_2_val,  "M_6" => M_6_val,  "M_7" => M_7_val,
		"H_1"    => H_1_val,   "H_3" => H_3_val,  "H_4" => H_4_val,
		"H_5"    => H_5_val,   "H_6" => H_6_val,
		"N_1"    => N_1_val,   "N_2" => N_2_val,  "N_6" => N_6_val,  "N_7" => N_7_val,
		# Stimulus parameters are saved for reference but not loaded by the main workflow
		"_stim_h"   => stim_h_val,
		"_stim_d"   => stim_d_val,
		"_stim_dim" => stim_dim_val
	)

	out_path = joinpath(pwd(), "initial_params.json")
	open(out_path, "w") do io
		JSON3.write(io, export_dict)
	end

	md"""
	**Parameters exported** to `initial_params.json`.

	The main workflow will use these as starting conditions. Re-run the
	`traces` or `group` workflow to apply them.

	| Parameter | Value |
	|-----------|-------|
	| RMP | $(RMP_val) mV |
	| g_Na | $(g_Na_val) mS/cm² |
	| g_K | $(g_K_val) mS/cm² |
	| h∞(RMP) | $(round(ActionPotentialModel.infty_h(RMP_val, full_sim_params), digits=3)) |
	| Peak V | $(round(peak_voltage, digits=1)) mV |
	"""
end

# ╔═╡ Cell order:
# ╠═f7312cb2-8a05-11f0-2004-890435c0dfe8
# ╟─intro-cell
# ╟─d94f4e1e-c3b0-4826-a4fe-25b95e6c7370
# ╠═51466345-d9d5-4b3e-9cd6-cae3ff3289bc
# ╟─export-cell
# ╠═export-action
