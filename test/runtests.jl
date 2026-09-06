# test/runtests.jl
#
# Smoke + regression tests for the HH model core (ActionPotential.jl) and its
# supporting config. These exist to catch exactly the kind of silent drift
# this audit found: report.jmd and README.md described h-gate parameters and
# solver tolerances that no longer matched config.jl/ActionPotential.jl. Encode
# the CURRENT, intended behaviour here so future changes to config.jl or the
# scoring/solve internals fail loudly instead of only being caught by manually
# re-reading prose against code.
#
# Run from the project root (after `julia setup.jl` has instantiated the
# project environment):
#   julia --project=. test/runtests.jl

using Test

const _ROOT = dirname(@__DIR__)
cd(_ROOT)
include(joinpath(_ROOT, "config.jl"))
include(joinpath(_ROOT, "ActionPotential.jl"))
using .ActionPotentialModel
const APM = ActionPotentialModel

@testset "ActionPotentialJulia" begin

    @testset "h-gate calibration (config.jl defaults)" begin
        # Snake-muscle values: Na+ mostly INACTIVATED at rest. This is the
        # opposite of the textbook squid-axon calibration (h_inf ~ 0.75) and is
        # load-bearing — see config.jl's header comment and report.jmd
        # Critical Findings. If this test starts failing after an edit to
        # config.jl, that edit changed the model's resting behaviour and
        # report.jmd / README.md's Model Description need to be updated too.
        h_inf_rest = APM.infty_h(par_0.RMP, par_0)
        @test 0.01 < h_inf_rest < 0.06

        # infty_h/infty_m/infty_n must be in [0,1] for any physiological V —
        # they're literal channel-open probabilities.
        for V in -100.0:10.0:60.0
            @test 0.0 <= APM.infty_h(V, par_0) <= 1.0
            @test 0.0 <= APM.infty_m(V, par_0) <= 1.0
            @test 0.0 <= APM.infty_n(V, par_0) <= 1.0
        end
    end

    @testset "stimulus function and its analytic integral" begin
        d, h, dim = 0.64, 54.874, 2.0
        @test APM.stim_function(-1.0, d, h, dim) == 0.0
        @test APM.stim_function(0.0, d, h, dim) == 0.0
        @test APM.stim_function(d, d, h, dim) ≈ h
        @test APM.stim_integral(0.0, d, h, dim) == 0.0
        @test APM.stim_integral(-1.0, d, h, dim) == 0.0
        # stim_integral(T) must equal a fine-step numerical quadrature of
        # stim_function over [0, T].
        T = 0.4
        n = 200_000
        dt_fine = T / n
        numerical = sum(APM.stim_function((i - 0.5) * dt_fine, d, h, dim) * dt_fine for i in 1:n)
        @test isapprox(APM.stim_integral(T, d, h, dim), numerical; rtol=1e-3)
        # Degenerate stim_d must not throw (guards NelderMead simplex excursions
        # into stim_d <= 0 during foot-fitting).
        @test APM.stim_integral(T, -0.1, h, dim) == 0.0
        @test APM.stim_function(T, -0.1, h, dim) == 0.0
    end

    @testset "smooth_max_zero approximates max(0, x)" begin
        for x in (-50.0, -1.0, -0.01, 0.0, 0.01, 1.0, 50.0)
            @test isapprox(APM.smooth_max_zero(x), max(0.0, x); atol=0.2)
        end
        # Must stay finite/non-NaN across a wide range (no overflow in exp).
        @test isfinite(APM.smooth_max_zero(1e6))
        @test isfinite(APM.smooth_max_zero(-1e6))
    end

    @testset "_score_masks / _score_from_masks match the legacy inline computation" begin
        # Regression test for the mask-caching refactor: precomputing masks
        # once (see _score_masks) must give bit-identical scores to the
        # original per-call recomputation (_score_from_trace), for arbitrary
        # inputs — not just the ones exercised by a real fit.
        time_points = collect(0.0:0.02:30.0)
        dt = 0.02
        tot_wait = 10.8
        stabil_time = 10.0
        trace_data_len = 500
        rng_Vs  = -70.0 .+ 40.0 .* sin.(0.3 .* time_points)
        exp_pad = fill(-70.0, length(time_points))

        masks = APM._score_masks(tot_wait, time_points, dt, stabil_time, trace_data_len)
        s_fast = APM._score_from_masks(rng_Vs, masks, exp_pad)
        s_legacy = APM._score_from_trace(rng_Vs, tot_wait, time_points, dt, exp_pad,
                                          stabil_time, trace_data_len)
        @test s_fast == s_legacy

        # Inf/empty short-circuits must agree too.
        @test APM._score_from_masks(Float64[], masks, exp_pad) == Inf
        @test APM._score_from_trace(Float64[], tot_wait, time_points, dt, exp_pad,
                                     stabil_time, trace_data_len) == Inf
        bad_Vs = copy(rng_Vs); bad_Vs[10] = Inf
        @test APM._score_from_masks(bad_Vs, masks, exp_pad) == Inf
    end

    @testset "estimate_rmp / detect_foot_index on a synthetic trace" begin
        dt = 0.02
        n_pre = 100                       # 2 ms flat baseline
        rmp_true = -75.0
        baseline = fill(rmp_true, n_pre)
        upstroke = collect(range(rmp_true, 40.0; length=50))
        repol    = collect(range(40.0, rmp_true; length=200))
        trace = vcat(baseline, upstroke, repol)

        rmp_est = APM.estimate_rmp(trace, dt)
        @test isapprox(rmp_est, rmp_true; atol=1.0)

        fi = APM.detect_foot_index(trace, rmp_true)
        # Foot should land at/near the end of the flat baseline, well before
        # the peak, and strictly after the very start of the trace.
        @test n_pre - 5 <= fi <= n_pre + 5
    end

    @testset "single-trace fit pipeline runs end-to-end on real data" begin
        # Fast smoke test using a REAL recording (not synthetic): find_foot!
        # then a short simulate — confirms the ODE + solver + foot-finding
        # integrate correctly together, without paying for a full multi-
        # thousand-eval optimize! call.
        csv_path = joinpath(data_folder, "Atratus_WT.csv")
        if isfile(csv_path)
            using CSV, DataFrames
            df = CSV.read(csv_path, DataFrame; header=4, missingstring="")
            trace = collect(skipmissing(df[:, 2]))
            time  = collect(skipmissing(df[:, 1]))

            ap = APM.ActionPotential(par_0, trace, time, name="smoke_WT_1")
            @test ap.params.RMP < -40.0   # a real muscle RMP, not a garbage value
            @test all(isfinite, ap.Vs)

            APM.find_foot!(ap)
            @test 0.0 <= (ap.tot_wait - ap.stabil_time) <= 2.0   # onset within the search window
            @test ap.stim_d > 0.0

            APM.update_model!(ap, ap.params)
            @test all(isfinite, ap.Vs)
            # par_0 is documented (config.jl, report.jmd) to fire on every raw
            # trace once the foot is anchored — the model should reach at
            # least a real depolarising event above rest.
            @test maximum(ap.Vs) > ap.params.RMP + 20.0
        else
            @test_skip false   # data file not present in this checkout
        end
    end

    @testset "gpu_grid_search! respects par_bounds" begin
        # Regression test for a real bug found in this audit: gpu_grid_search!
        # used to ignore `bounds` entirely and sample ±90% of each parameter's
        # INITIAL value with no physiological floor/ceiling — unlike the CPU
        # global_optimize path, which clips to par_bounds when supplied. That
        # let the GPU Sobol grid draw numerically pathological combinations
        # (e.g. M_6 near zero) that hung the GPU solver. Skipped when no
        # functional CUDA device is present.
        cuda_ok = try
            @eval using CUDA
            CUDA.functional()
        catch
            false
        end
        if cuda_ok
            csv_path = joinpath(data_folder, "Atratus_WT.csv")
            if isfile(csv_path)
                using CSV, DataFrames
                df = CSV.read(csv_path, DataFrame; header=4, missingstring="")
                trace = collect(skipmissing(df[:, 2]))
                time  = collect(skipmissing(df[:, 1]))
                ap = APM.ActionPotential(par_0, trace, time, name="gpu_smoke_WT_1")
                APM.find_foot!(ap)
                result = APM.gpu_grid_search!(ap, opt_par_names;
                                               num_trajectories=50, range=0.9, bounds=par_bounds)
                @test isfinite(result["value"])
                for k in opt_par_names
                    lb, ub = par_bounds[k]
                    @test lb <= result["par"][k] <= ub
                end
            else
                @test_skip false
            end
        else
            @test_skip false   # no functional CUDA device in this environment
        end
    end

end
