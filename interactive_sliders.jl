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
	Pkg.activate(pwd()) # Ensures we use the project's packages
	
	using Plots
	using PlutoUI

	# Include your existing project code
	include("config.jl")
	include("ActionPotential.jl")
	using .ActionPotentialModel
end

# ╔═╡ d94f4e1e-c3b0-4826-a4fe-25b95e6c7370
@bind params_interactive PlutoUI.combine() do Child
	md"""
	### Tune Action Potential Parameters
	---
	#### **Simulation & Stimulus**
	**Resting Potential (RMP):** $(Child(Slider(-80.0:1.0:-55.0, default=-70.0, show_value=true)))
	**Stimulus Height (h):** $(Child(Slider(10.0:0.01:100.0, default=55.0, show_value=true)))
	**Stimulus Duration (d):** $(Child(Slider(0.1:0.05:2.0, default=0.64, show_value=true)))
	**Stimulus Shape (dim):** $(Child(Slider(1.0:0.1:4.0, default=2.0, show_value=true)))
	**Stabilization Time:** $(Child(Slider(5.0:1.0:20.0, default=10.0, show_value=true)))
	---
	#### **Conductances (g)**
	**g_Na:** $(Child(Slider(1.0:0.01:250.0, default=par_0.g_Na, show_value=true)))
	**g_K:** $(Child(Slider(1.0:0.5:75.0, default=par_0.g_K, show_value=true)))
	**g_Leak:** $(Child(Slider(0.0:0.05:1.0, default=par_0.g_Leak, show_value=true)))
	---
	#### **Sodium Channel (Na⁺) - m & h gates**
	**M_1:** $(Child(Slider(0.1:0.01:1.0, default=par_0.M_1, show_value=true)))
	**M_2:** $(Child(Slider(-90.0:0.5:-50.0, default=par_0.M_2, show_value=true)))
	**M_6:** $(Child(Slider(-25.0:0.5:-10.0, default=par_0.M_6, show_value=true)))
	**M_7:** $(Child(Slider(20.0:0.5:50.0, default=par_0.M_7, show_value=true)))
	
	**H_3:** $(Child(Slider(-30.0:0.5:-15.0, default=par_0.H_3, show_value=true)))
	**H_4:** $(Child(Slider(50.0:1.0:80.0, default=par_0.H_4, show_value=true)))
	**H_5:** $(Child(Slider(-10.0:0.1:-1.0, default=par_0.H_5, show_value=true)))
	**H_6:** $(Child(Slider(140.0:1.0:180.0, default=par_0.H_6, show_value=true)))
	---
	#### **Potassium Channel (K⁺) - n gate**
	**N_1:** $(Child(Slider(0.001:0.0001:0.01, default=par_0.N_1, show_value=true)))
	**N_2:** $(Child(Slider(-130.0:1.0:-100.0, default=par_0.N_2, show_value=true)))
	**N_6:** $(Child(Slider(-30.0:0.5:-15.0, default=par_0.N_6, show_value=true)))
	**N_7:** $(Child(Slider(50.0:1.0:80.0, default=par_0.N_7, show_value=true)))
	"""
end

# ╔═╡ 51466345-d9d5-4b3e-9cd6-cae3ff3289bc
begin
	# 1. Unpack all 20 parameter values from the interactive sliders
	RMP_val, stim_h_val, stim_d_val, stim_dim_val, stabil_time_val,
	g_Na_val, g_K_val, g_Leak_val,
	M_1_val, M_2_val, M_6_val, M_7_val,
	H_3_val, H_4_val, H_5_val, H_6_val,
	N_1_val, N_2_val, N_6_val, N_7_val = params_interactive

	# 2. Assemble a complete "snapshot" of all parameters for the simulation
	full_sim_params = merge(par_0, (
		RMP = RMP_val,
		stim_h = stim_h_val,
		stim_d = stim_d_val,
		stim_dim = stim_dim_val,
		tot_wait = stabil_time_val, # Link tot_wait to the stabil_time slider
		g_Na = g_Na_val, g_K = g_K_val, g_Leak = g_Leak_val,
		M_1 = M_1_val, M_2 = M_2_val, M_6 = M_6_val, M_7 = M_7_val,
		H_3 = H_3_val, H_4 = H_4_val, H_5 = H_5_val, H_6 = H_6_val,
		N_1 = N_1_val, N_2 = N_2_val, N_6 = N_6_val, N_7 = N_7_val
	))

	# 3. Define the time vector for the simulation
	sim_time = 30.0
	dt = 0.02
	time_points = collect(0.0:dt:sim_time)
	
	# 4. Call the pure simulation function directly
	simulated_Vs = ActionPotentialModel._simulate_trace(full_sim_params, time_points, dt)
	#simulated_Vs = ActionPotentialModel._simulate_trace_euler(full_sim_params, time_points, dt, RMP_val)
	
	# --- NEW: Add a diagnostic to see the peak voltage ---
	peak_voltage = maximum(simulated_Vs)

	# 5. Plot the result
	plot(time_points, simulated_Vs, 
		 label="Simulated Voltage", 
		 lw=2, 
		 xlabel="Time (ms)", 
		 ylabel="Voltage (mV)",
		 legend=:topright,
		 ylims=(-100, 50),
		 # Also display the peak voltage in the title
		 title = "Interactive AP (Peak: $(round(peak_voltage, digits=1)) mV)"
	)
end

# ╔═╡ Cell order:
# ╠═f7312cb2-8a05-11f0-2004-890435c0dfe8
# ╟─d94f4e1e-c3b0-4826-a4fe-25b95e6c7370
# ╠═51466345-d9d5-4b3e-9cd6-cae3ff3289bc
