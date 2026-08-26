"""Roll one frozen state forward under two different action windows.

This is the overlay that shows what "action-conditioned" actually means. Everything else
here would look identical for a passive predictor: only this forces the model to commit to
two different futures from the same state, because the only thing that changed is the plan.

Both branches start from exactly the same observation and the same world-model state. The
branch is the action window alone, so any divergence in the imagined futures is caused by
the conditioning and nothing else.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_sc2_true as R  # loader re-exec

import numpy as np
import torch

from mact_introspect import ObsDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze-at", type=int, default=-1,
                    help="step to branch from; -1 picks the step with the most visible enemies")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--targets", default="0,2", help="two enemy ids to focus, e.g. 0,2")
    ap.add_argument("--all-steps", action="store_true",
                    help="branch at EVERY step, so the fork can play live rather than frozen")
    args = ap.parse_args()

    rec = np.load(args.rec, allow_pickle=True)
    if "observations" not in rec.files:
        raise SystemExit("FORK recording has no observations; re-record with the updated tool")

    obs_all = rec["observations"]
    enemies = rec["enemies"]

    t = args.freeze_at
    if t < 0:
        alive = [(int(np.sum(enemies[i, :, 2] > 0)), i) for i in range(len(enemies))]
        # a mid-episode step where several enemies are still alive makes the fork legible
        cands = [i for c, i in alive if c >= 2]
        t = cands[len(cands) // 3] if cands else len(enemies) // 2
    print("FORK branching at step %d of %d" % (t, len(obs_all)), flush=True)

    import record_overlay_data as ROD
    env, controller, cfg, world_model = ROD.build(args.repo, args.map, args.ckpt, 1, 320, 180, 22.0)
    env.reset()
    dec = ObsDecoder(env)
    from agent.models.world_model_env import MAWorldModelEnv
    wm_env = MAWorldModelEnv(tokenizer=controller.tokenizer, world_model=world_model,
                             device="cpu", env_type=cfg.ENV_TYPE)

    n = env.n_agents
    tgt = [int(x) for x in args.targets.split(",")]
    steps = range(len(obs_all)) if args.all_steps else [t]
    names = ["focus enemy %d" % x for x in tgt]
    per_branch = [[] for _ in tgt]
    import time as _t
    t0 = _t.time()
    for si, s_idx in enumerate(steps):
        for bi, target in enumerate(tgt):
            seq = [[6 + target] * n] * args.horizon   # SMAC: action 6+i attacks enemy i
            im = ROD.imagine(wm_env, list(obs_all[s_idx]), seq, dec, n)
            per_branch[bi].append(np.array([x[0] for x in im]))
        if args.all_steps and si % 10 == 0:
            print("FORK   step %3d/%d  %.2fs/step" % (si, len(obs_all), (_t.time()-t0)/max(si,1)),
                  flush=True)
    out = {("branch_%d" % i): np.array(v) for i, v in enumerate(per_branch)}
    np.savez_compressed(args.out, freeze_at=t, all_steps=bool(args.all_steps),
                        names=np.array(names), **out)
    print("FORK branches shape: %s" % (out["branch_0"].shape,), flush=True)
    print("FORK wrote %s" % args.out, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
