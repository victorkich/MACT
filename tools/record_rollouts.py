"""Roll out a trained MACT / MARIE checkpoint in SMAC and dump per-step unit state to JSON.

At execution time these methods use only the VQ tokenizer + the actor MLP (the world model is
used for imagination during training only -- see DreamerController.receive_params), so this runs
comfortably on CPU.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch


def build_controller(repo, map_name, ckpt_path, seed, temperature=1.0):
    sys.path.insert(0, repo)
    os.chdir(repo)
    from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
    from configs.EnvConfigs import StarCraftConfig

    env_config = StarCraftConfig(map_name, seed)
    env = env_config.create_env()

    cfg = DreamerControllerConfig()
    cfg.IN_DIM = env.n_obs
    cfg.ACTION_SIZE = env.n_actions
    cfg.NUM_AGENTS = env.n_agents
    cfg.CONTINUOUS_ACTION = False
    cfg.ACTION_SPACE = None
    cfg.ENV_TYPE = __import__("environments").Env.STARCRAFT
    cfg.tokenizer_type = "vq"
    cfg.temperature = temperature
    if hasattr(cfg, "determinisitc"):
        cfg.determinisitc = True

    controller = cfg.create_controller()
    params = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    controller.tokenizer.load_state_dict(params["tokenizer"])
    controller.actor.load_state_dict(params["actor"])
    controller.tokenizer.eval()
    controller.actor.eval()
    return env, controller, cfg


def snapshot(sc2, actions=None):
    """Capture positions/health of every unit this step, plus what each ally did."""
    def unit_rec(u, alive):
        return dict(
            x=float(u.pos.x), y=float(u.pos.y),
            hp=float(u.health), hp_max=float(u.health_max),
            sh=float(u.shield), sh_max=float(u.shield_max),
            utype=int(u.unit_type), alive=bool(alive),
        )

    allies, enemies = [], []
    for i in range(sc2.n_agents):
        u = sc2.get_unit_by_id(i)
        allies.append(unit_rec(u, u.health > 0))
    for e_id, e_u in sorted(sc2.enemies.items()):
        enemies.append(unit_rec(e_u, e_u.health > 0))

    rec = dict(allies=allies, enemies=enemies)
    if actions is not None:
        # SMAC action ids: 0 no-op, 1 stop, 2 N, 3 S, 4 E, 5 W, 6+ attack enemy (id = a-6)
        rec["actions"] = [int(a) for a in actions]
        rec["attack"] = [int(a) - 6 if int(a) >= 6 else -1 for a in actions]
    return rec


@torch.no_grad()
def run_episode(env, controller, cfg, record_replay_dir=None):
    sc2 = env.env
    state = {i: torch.tensor(o).float() for i, o in env.reset().items()}
    controller.init_rnns()
    controller.init_buffer()

    done = defaultdict(lambda: False)
    frames = []
    frames.append(snapshot(sc2))
    steps = 0
    info = {}

    while True:
        steps += 1
        avail, obs_list = [], []
        for h in range(env.n_agents):
            avail.append(torch.tensor(env.get_avail_agent_actions(h)))
            if done[h] == 1:
                obs_list.append(torch.zeros(1, cfg.IN_DIM))
            else:
                obs_list.append(state[h].unsqueeze(0))
        observations = torch.cat(obs_list).unsqueeze(0)
        av_action = torch.stack(avail).unsqueeze(0)

        actions, _ = controller.step(observations, av_action, None)
        # controller.step returns action.squeeze(0) -> (n_agents, n_actions)
        acts = [int(a.argmax()) for a in actions]

        next_state, reward, dones, info = env.step(acts)
        frames.append(snapshot(sc2, actions=acts))

        state = {i: torch.tensor(o).float() for i, o in next_state.items()}
        done = {i: float(d) for i, d in dones.items()}
        if all(done[k] == 1 for k in range(env.n_agents)):
            break

    won = bool(info.get("battle_won", False))
    if record_replay_dir:
        try:
            sc2.save_replay()
        except Exception as e:
            print(f"  (replay save failed: {e})")
    return dict(frames=frames, steps=steps, won=won,
                map_x=int(sc2.map_x), map_y=int(sc2.map_y),
                n_agents=int(env.n_agents), n_enemies=int(len(sc2.enemies)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--only-wins", action="store_true")
    ap.add_argument("--max-tries", type=int, default=24)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env, controller, cfg = build_controller(args.repo, args.map, args.ckpt, args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def flush(eps):
        """Write after every episode: this box is heavily contended and a job can be
        killed mid-run, so partial results must survive."""
        out = dict(label=args.label or os.path.basename(args.ckpt),
                   map=args.map, episodes=eps,
                   n_won=sum(e["won"] for e in eps), n_total=len(eps))
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, args.out)
        return out

    eps, tries = [], 0
    while len(eps) < args.episodes and tries < args.max_tries:
        tries += 1
        ep = run_episode(env, controller, cfg)
        print(f"  ep{tries}: steps={ep['steps']} won={ep['won']}", flush=True)
        if args.only_wins and not ep["won"]:
            continue
        eps.append(ep)
        out = flush(eps)
        print(f"  saved {len(eps)} eps ({out['n_won']} won) -> {args.out}", flush=True)

    env.close()
    out = flush(eps)
    print(f"wrote {args.out}: {len(eps)} eps, {out['n_won']} won")


if __name__ == "__main__":
    main()
