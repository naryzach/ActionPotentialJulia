# worker_functions.jl
#
# Top-level worker functions for distributed pmap.
# Included @everywhere in main.jl AFTER workers are added, so each worker
# has these definitions in its own top-level world (Julia 1.12 requirement).

function fit_trace(task)
    p0, bounds, trace, time, name, opn, use_gpu, num_traj = task
    redirect_stdout(devnull) do
        ap     = ActionPotentialModel.ActionPotential(p0, trace, time, name=name)
        result = ActionPotentialModel.optimize!(ap, opn; bounds=bounds,
                                                use_gpu=use_gpu, num_trajectories=num_traj)
        feats  = ActionPotentialModel.extract_ap_features(ap)
        return (
            name           = name,
            params         = result["par"],
            value          = result["value"],
            convergence    = result["convergence"],
            RMP            = ap.params.RMP,
            final_stim_d   = ap.stim_d,
            final_stim_dim = ap.stim_dim,
            final_tot_wait = ap.tot_wait,
            features       = feats
        )
    end
end

function run_group_fit(task)
    tbl, grp, indiv, seed_p, bounds, trace, time, opn, use_gpu, num_traj = task
    redirect_stdout(devnull) do
        ap     = ActionPotentialModel.ActionPotential(seed_p, trace, time,
                                                      name="T$tbl-G$grp-I$indiv")
        result = ActionPotentialModel.optimize!(ap, opn; bounds=bounds,
                                                use_gpu=use_gpu, num_trajectories=num_traj)
        feats  = ActionPotentialModel.extract_ap_features(ap)

        res = Dict{Symbol, Any}(pairs(result["par"]))
        res[:tbl]      = tbl
        res[:group_id] = grp
        res[:indiv]    = indiv
        res[:score]    = result["value"]

        if !isnothing(feats)
            for (k, v) in pairs(feats)
                res[Symbol("feat_", k)] = v
            end
        end
        return res
    end
end
