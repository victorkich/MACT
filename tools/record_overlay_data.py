"""Record a rollout together with everything needed to draw MACT's internals on the frames.

Per environment step this captures the rendered frame, the camera the game actually used, the
true unit positions, the world model's imagined future, and the per-horizon AC-CPC similarity.
The overlay renderers consume the resulting .npz and draw. Nothing is computed at draw time,
so the same recording feeds all four videos.

A property of SMAC that shapes the whole thing: an agent's observation encodes other units
RELATIVE to itself (offsets divided by sight_range) and its own features are only health,
shield and unit type. There is no absolute self-position anywhere in the input, so the world
model cannot predict absolute coordinates and never could. What it predicts is the team's
relative geometry.

To place that prediction in the world honestly, a prediction made at t for horizon k is
anchored at the observing agent's TRUE position at t+k, which is known because the whole
episode is recorded. The ghost then shows exactly what the model got right or wrong, the
relative part, without being charged for an absolute position it was never given.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_sc2_true as R  # performs the loader re-exec on import

import numpy as np
import torch
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2

from mact_introspect import ObsDecoder, validate_obs_decoder


def build(repo, map_name, ckpt, seed, width, height, camw):
    """The DreamerController holds only the tokenizer and actor, because decentralized
    execution needs nothing else. The world model has to be constructed separately, through
    the same learner config training uses, so the architecture matches the checkpoint."""
    R.patch_smac_launch(width, height, camera_width=camw)
    env, controller, cfg = R.build_controller(repo, map_name, ckpt, seed)

    from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
    lcfg = DreamerLearnerConfig()
    # train.py's get_env_info, inlined: importing train.py would run training, and calling it
    # would launch a second StarCraft instance just to read four numbers off an env we have.
    lcfg.IN_DIM = env.n_obs
    lcfg.ACTION_SIZE = env.n_actions
    lcfg.NUM_AGENTS = env.n_agents
    lcfg.CONTINUOUS_ACTION = not env.discrete
    lcfg.ACTION_SPACE = getattr(env, "individual_action_space", None) if not env.discrete else None
    lcfg.ENV_TYPE = cfg.ENV_TYPE
    lcfg.tokenizer_type = getattr(cfg, "tokenizer_type", "vq")
    # The learner config defaults to cuda. Inference here is tiny and this container has
    # lost GPU access anyway (NVML reports Unknown Error), so pin it to CPU.
    lcfg.DEVICE = "cpu"
    # train.py sets these from CLI flags rather than the config class, and MAWorldModel reads
    # them at construction. Defaults match a plain `train.py --env starcraft` run, which is how
    # the released checkpoints were produced.
    lcfg.use_ce_for_end = False
    lcfg.use_ce_for_r = getattr(lcfg, "use_ce_for_r", False)
    # the released checkpoints were trained with --ce_for_av, which wraps the
    # avail-action head in a `dists` module; leaving it False leaves 6 keys unloaded
    lcfg.use_ce_for_av_action = True
    lcfg.cpc_v2 = getattr(lcfg, "cpc_v2", False)
    lcfg.cpc_temp = getattr(lcfg, "cpc_temp", 1.0)
    lcfg.cpc_mode = getattr(lcfg, "cpc_mode", "per_agent")
    lcfg.action_agg = getattr(lcfg, "action_agg", "mean")
    lcfg.latent_aug_scale = getattr(lcfg, "latent_aug_scale", 0.1)
    lcfg.critic_average_r = getattr(lcfg, "critic_average_r", False)
    apply_per_map_overrides(lcfg, map_name)   # needs NUM_AGENTS, so it runs after the assignments
    learner = lcfg.create_learner()

    params = torch.load(ckpt, map_location="cpu", weights_only=False)
    try:
        missing, unexpected = learner.model.load_state_dict(params["model"], strict=False)
    except RuntimeError as exc:
        raise SystemExit("REC world model does not match the checkpoint:\n%s" % exc)
    n_c = sum(1 for k in params["model"] if "contrastive_network" in k)
    # MARIE has no AC-CPC head, so loading it into MACT's superset legitimately leaves the
    # contrastive tensors unfilled. That is expected absence, not an architecture mismatch, so
    # only count the OTHER missing keys against the guard.
    structural = [k for k in missing if "contrastive_network" not in k]
    print("REC world model loaded: %d missing (%d structural), %d unexpected, "
          "%d contrastive tensors in ckpt (%s)"
          % (len(missing), len(structural), len(unexpected), n_c,
             "MACT" if n_c else "MARIE / no AC-CPC head"), flush=True)
    if len(structural) > 8:
        raise SystemExit("REC world model architecture does not match the checkpoint")
    learner.model.eval()
    return env, controller, cfg, learner.model


def apply_per_map_overrides(cfg, map_name):
    """Mirror train.py's per-map setup.

    Setting HORIZON alone is NOT enough. `trans_config` is a functools.partial that baked
    `max_blocks` when the config object was constructed, so the transformer keeps its default
    sequence length and the checkpoint refuses to load: a HORIZON=15 checkpoint carries a
    270x270 attention mask against the default's 144x144. train.py rebuilds the partial, so
    this does too.
    """
    import functools
    if map_name in ("so_many_baneling", "2s3z"):
        horizon, epochs, k_cpc = 5, 30, 5
    elif getattr(cfg, "NUM_AGENTS", 0) > 5:
        horizon, epochs, k_cpc = 8, 10, 8
    else:
        horizon, epochs, k_cpc = 15, 4, 8
    cfg.HORIZON = horizon
    cfg.SEQ_LENGTH = horizon
    old_tc = cfg.trans_config
    cfg.trans_config = functools.partial(
        old_tc.func, **{**old_tc.keywords, "max_blocks": horizon})
    cfg.EPOCHS = epochs
    cfg.K_cpc = k_cpc


def _patch_kv_refresh(wm_env):
    """Work around a shape bug in MAWorldModelEnv.refresh_keys_values_with_initial_obs_tokens.

    That method does `n, k = obs_tokens.shape`, which holds when
    reset_from_initial_observations calls it (it rearranges to 2-D first) but not when step_ar
    calls it on cache overflow, where self.obs_tokens is still (b, n, k). Any map whose rollout
    exceeds max_tokens therefore dies with "too many values to unpack".

    so_many_baneling is the one that hits it: 7 agents x 32 enemies gives 202-dim observations,
    so the token budget runs out mid-rollout. The failure was silent because the recorder caught
    the exception and wrote NaN, which rendered as a clip with no rings and frozen bars.

    Fixing it here rather than in the training code, which is not ours to change for a
    visualisation.
    """
    from einops import rearrange
    orig_refresh = wm_env.refresh_keys_values_with_initial_obs_tokens

    def refresh(obs_tokens):
        if obs_tokens.dim() == 3:
            obs_tokens = rearrange(obs_tokens, "b n k -> (b n) k")
        return orig_refresh(obs_tokens)

    wm_env.refresh_keys_values_with_initial_obs_tokens = refresh
    return wm_env


@torch.no_grad()
def imagine(wm_env, obs_now, action_seq, dec, n_agents):
    """Roll the frozen world model forward and decode each step back to relative positions.

    Returns a list over horizon of (enemy_rel, ally_rel) arrays, each (n_agents, n_units, 2),
    expressed as offsets from the observing agent.
    """
    obs_t = torch.tensor(np.stack(obs_now), dtype=torch.float32).unsqueeze(0)
    wm_env.reset_from_initial_observations(obs_t)
    out = []
    for acts in action_seq:
        a = torch.tensor(acts, dtype=torch.long).view(1, n_agents, 1)
        wm_env.step_ar(a, should_predict_next_obs=True)
        rec = wm_env.decode_obs_tokens()
        rec = rec.reshape(n_agents, -1).cpu().numpy()
        en_rel, al_rel = [], []
        for i in range(n_agents):
            ex, es, ax, as_ = dec.decode(rec[i], i, (0.0, 0.0))  # anchor at origin -> pure offsets
            en_rel.append(ex)
            al_rel.append(ax)
        out.append((np.array(en_rel), np.array(al_rel)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=432)
    ap.add_argument("--camera-width", type=float, default=22.0)
    ap.add_argument("--pan-y", type=float, default=1.0)
    ap.add_argument("--pick", default="first_win", choices=["first_win", "first"])
    ap.add_argument("--max-tries", type=int, default=2)
    ap.add_argument("--render-timeout", type=float, default=5.0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop the episode after N steps (0 = run to termination)")
    ap.add_argument("--force-target", type=int, default=-1,
                    help="override the policy: every agent attacks this enemy id when able. "
                         "Used to execute a chosen focus-fire plan so two runs actually diverge.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env, controller, cfg, world_model = build(args.repo, args.map, args.ckpt, args.seed,
                                             args.width, args.height, args.camera_width)
    env.reset()
    sc2 = env.env
    dec = ObsDecoder(env)
    print("REC obs layout: %s" % dec.describe(), flush=True)
    for _ in range(12):   # close the distance: nothing is within sight range at reset
        avail = [sc2.get_avail_agent_actions(i) for i in range(env.n_agents)]
        env.step([int(np.argmax(v)) for v in avail])
    errs = validate_obs_decoder(env, dec)
    if not np.isfinite(np.nanmean(errs)) or len(errs) < 2:
        raise SystemExit("REC decoder check saw no visible enemies, so it verified nothing")
    if np.nanmean(errs) > 0.25:
        raise SystemExit("REC observation decoder does not reproduce true positions, refusing to record")
    env.reset()

    from agent.models.world_model_env import MAWorldModelEnv
    wm_env = MAWorldModelEnv(tokenizer=controller.tokenizer, world_model=world_model,
                             device="cpu", env_type=cfg.ENV_TYPE)
    _patch_kv_refresh(wm_env)

    best = None
    for attempt in range(1, args.max_tries + 1):
        ep = run_episode(env, controller, cfg, sc2, dec, wm_env, args)
        print("REC ep%d steps=%d won=%s frames=%d" % (attempt, ep["steps"], ep["won"], len(ep["frames"])),
              flush=True)
        if args.pick == "first" or ep["won"]:
            best = ep
            break
        best = ep
    np.savez_compressed(args.out, **best)
    print("REC wrote %s" % args.out, flush=True)
    os._exit(0)


@torch.no_grad()
def run_episode(env, controller, cfg, sc2, dec, wm_env, args):
    from collections import defaultdict

    def ctrl():
        """Never cache this. SMAC's full_restart() closes the process and builds a new
        RemoteController, and a stale handle raises `called while in state: Status.quit`."""
        return sc2._controller

    state = {i: torch.tensor(o).float() for i, o in env.reset().items()}
    controller.init_rnns()
    controller.init_buffer()

    t_start = time.time()
    frames, cams, allies, enemies, imagined, acts_log, obs_log = [], [], [], [], [], [], []
    done = defaultdict(lambda: False)
    steps, info = 0, {}
    n = env.n_agents

    def snapshot():
        a = np.full((n, 3), np.nan)
        for i in range(n):
            u = sc2.get_unit_by_id(i)
            a[i] = (u.pos.x, u.pos.y, u.health)
        e_ids = sorted(sc2.enemies.keys())
        e = np.full((len(e_ids), 3), np.nan)
        for j, k in enumerate(e_ids):
            u = sc2.enemies[k]
            e[j] = (u.pos.x, u.pos.y, u.health)
        return a, e

    def capture():
        c = np.nanmean(np.concatenate([snapshot()[0][:, :2], snapshot()[1][:, :2]]), axis=0)
        ctrl().act(sc_pb.Action(action_raw=raw_pb2.ActionRaw(camera_move=raw_pb2.ActionRawCameraMove(
            center_world_space=dict(x=float(c[0]), y=float(c[1] + args.pan_y), z=0)))))
        t0 = time.time()
        while time.time() - t0 < args.render_timeout:
            obs = ctrl().observe()
            r = obs.observation.render_data.map
            if r.data:
                img = np.frombuffer(r.data, dtype=np.uint8).reshape(r.size.y, r.size.x, 3).copy()
                cam = obs.observation.raw_data.player.camera
                return img, (cam.x, cam.y)
            time.sleep(0.02)
        return None, None

    while True:
        if args.max_steps and steps >= args.max_steps:
            print("REC   hit --max-steps %d" % args.max_steps, flush=True)
            break
        steps += 1
        avail, obs_list = [], []
        for h in range(n):
            avail.append(torch.tensor(env.get_avail_agent_actions(h)))
            obs_list.append(torch.zeros(1, cfg.IN_DIM) if done[h] == 1 else state[h].unsqueeze(0))
        observations = torch.cat(obs_list).unsqueeze(0)
        av_action = torch.stack(avail).unsqueeze(0)

        action, _ = controller.step(observations, av_action, None)
        a_int = [int(x.argmax()) for x in action]

        if args.force_target >= 0:
            # Execute a specific focus-fire plan rather than the policy's choice, so that two
            # runs with different targets genuinely diverge. Agents that cannot reach the target
            # keep their own action, otherwise they would stand still and the run would stall.
            forced = 6 + args.force_target
            for h in range(n):
                if done[h] != 1 and forced < av_action.shape[-1] and av_action[0, h, forced] > 0:
                    a_int[h] = forced

        img, cam = capture()
        if img is not None:
            al, en = snapshot()
            frames.append(img)
            cams.append(cam)
            allies.append(al)
            enemies.append(en)
            acts_log.append(a_int)
            obs_log.append(np.stack([(state[i].numpy() if done[i] != 1
                                      else np.zeros(cfg.IN_DIM, dtype=np.float32))
                                     for i in range(n)]))
            # imagine forward, repeating the chosen action (the window the agent is committing to)
            try:
                seq = [a_int] * args.horizon
                im = imagine(wm_env, [o.numpy() for o in
                                      [state[i] if done[i] != 1 else torch.zeros(cfg.IN_DIM)
                                       for i in range(n)]], seq, dec, n)
                imagined.append(np.array([x[0] for x in im]))   # (H, n_agents, n_enemies, 2)
            except Exception as exc:
                if len(imagined) == 0:
                    import traceback as _tb
                    print("REC imagination failed: %r" % (exc,), flush=True)
                    _tb.print_exc()
                imagined.append(np.full((args.horizon, n, len(en), 2), np.nan))

        if steps % 5 == 0:
            print("REC   step %3d  frames %3d  %.1fs/step" %
                  (steps, len(frames), (time.time() - t_start) / max(steps, 1)), flush=True)
        next_state, reward, dones, info = env.step(a_int)
        state = {i: torch.tensor(o).float() for i, o in next_state.items()}
        done = {i: float(d) for i, d in dones.items()}
        if all(done[k] == 1 for k in range(n)):
            break

    return dict(frames=np.array(frames), cams=np.array(cams, dtype=float),
                allies=np.array(allies), enemies=np.array(enemies),
                imagined=np.array(imagined), actions=np.array(acts_log),
                observations=np.array(obs_log),
                steps=steps, won=bool(info.get("battle_won", False)),
                horizon=args.horizon)


if __name__ == "__main__":
    main()
