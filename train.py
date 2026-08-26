import argparse
import os
import shutil
import datetime
from pathlib import Path

from agent.runners.DreamerRunner import DreamerRunner
from configs import Experiment
from configs.EnvConfigs import StarCraftConfig


from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
# for MPE

# for GRF

# for MAMuJoCo


from environments import Env
from utils import generate_group_name, format_numel_str_deci

import torch
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="starcraft", help='environment (starcraft)')
    parser.add_argument('--env_name', type=str, default="5_agents", help='Specific setting')

    # specialized arg for MAMujoco
    parser.add_argument('--agent_conf', type=str, default=None)

    parser.add_argument('--n_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--seed', type=int, default=1, help='Number of workers')
    parser.add_argument('--steps', type=int, default=1e6, help='Number of workers')
    parser.add_argument('--mode', type=str, default='disabled')
    parser.add_argument('--tokenizer', type=str, default='vq')
    parser.add_argument('--decay', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1.)  # for controller sampling data

    parser.add_argument('--sample_temp', type=float, default='inf')

    parser.add_argument('--average_r', action='store_true')
    parser.add_argument('--ce_for_r', action='store_true')
    parser.add_argument('--ce_for_av', action='store_true')
    parser.add_argument('--ce_for_end', action='store_true')
    parser.add_argument('--cpc_v2', action='store_true',
                        help='AC-CPC v2: latent-space augmentation')
    parser.add_argument('--cpc_he', action='store_true',
                        help='v2 augmentation + raised entropy coefficient (0.01)')
    # --- ablation knobs (single-seed studies) ---
    parser.add_argument('--k_cpc', type=int, default=None,
                        help='override K_cpc (CPC horizon); ablation. Default: per-map value.')
    parser.add_argument('--noise', type=float, default=None,
                        help='override latent_aug_scale (v2 noise level); ablation. Default 0.1.')
    parser.add_argument('--cpc_mode', type=str, default='per_agent', choices=['per_agent', 'team'],
                        help="CPC conditioning: 'per_agent' (default) or 'team' (aggregated).")
    parser.add_argument('--run_tag', type=str, default='',
                        help='no-op marker for unique process identification + wandb run naming (ablations).')

    return parser.parse_args()


def train_dreamer(exp, n_workers): 
    runner = DreamerRunner(exp.env_config, exp.learner_config, exp.controller_config, n_workers)
    runner.run(exp.steps, exp.episodes, save_interval = 200000, save_mode = 'interval')


def get_env_info(configs, env):
    if not env.discrete:
        assert hasattr(env, 'individual_action_space')
        individual_action_space = env.individual_action_space
    else:
        individual_action_space = None

    for config in configs:
        config.IN_DIM = env.n_obs
        config.ACTION_SIZE = env.n_actions
        config.NUM_AGENTS = env.n_agents
        config.CONTINUOUS_ACTION = not env.discrete
        config.ACTION_SPACE = individual_action_space
    
    print(f'Observation dims: {env.n_obs}')
    print(f'Action dims: {env.n_actions}')
    print(f'Num agents: {env.n_agents}')
    print(f'Continuous action for control? -> {not env.discrete}')
    
    if hasattr(env, 'individual_action_space'):
        print(f'Individual action space: {env.individual_action_space}')

    env.close()


def prepare_starcraft_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = StarCraftConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 2000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

if __name__ == "__main__":
    RANDOM_SEED = 23
    args = parse_args()
    RANDOM_SEED += args.seed * 100
    if args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name)
    else:
        raise Exception("This release ships the StarCraft II setup used for the paper.")
    
    # seed everywhere
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # --------------------

    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)

    configs["learner_config"].seed = RANDOM_SEED

    configs["learner_config"].tokenizer_type = args.tokenizer
    configs["controller_config"].tokenizer_type = args.tokenizer
    configs["learner_config"].ema_decay = args.decay
    configs["controller_config"].ema_decay = args.decay

    configs["controller_config"].temperature = args.temperature

    configs["learner_config"].critic_average_r = args.average_r

    configs["learner_config"].use_ce_for_r = args.ce_for_r
    configs["learner_config"].use_ce_for_end = False
    configs["learner_config"].use_ce_for_av_action = args.ce_for_av

    # AC-CPC v2 (latent-space augmentation). --cpc_he = v2 + raised entropy.
    configs["learner_config"].cpc_v2 = args.cpc_v2 or args.cpc_he
    if configs["learner_config"].cpc_v2:
        configs["learner_config"].cpc_temp = 1.0   # TWISTER uses NO temperature
        configs["learner_config"].latent_aug_scale = 0.1
    # ablations: conditioning mode + noise-level override (applied after the v2 default)
    configs["learner_config"].cpc_mode = args.cpc_mode
    if args.noise is not None:
        configs["learner_config"].latent_aug_scale = args.noise

    rewards_prediction_config = configs["learner_config"].rewards_prediction_config

    if args.sample_temp == float('inf'):
        configs["learner_config"].sample_temperature = str(args.sample_temp)
    else:
        configs["learner_config"].sample_temperature = args.sample_temp

    # ── Per-map hyperparameter overrides (MARIE paper Table 15 / Appendix A.2) ──
    # MACT inherits the same agent-count scaling as MARIE:
    import functools as _functools
    n_agents = configs["learner_config"].NUM_AGENTS

    _H5_MAPS = {"so_many_baneling", "2s3z"}
    if args.env_name in _H5_MAPS:
        horizon = 5
        epochs  = 30
        k_cpc   = 5
    elif n_agents > 5:
        horizon = 8
        epochs  = 10
        k_cpc   = 8
    else:   # n_agents ≤ 5
        horizon = 15
        epochs  = 4
        k_cpc   = 8

    for _cfg in [configs["learner_config"], configs["controller_config"]]:
        _cfg.HORIZON    = horizon
        _cfg.SEQ_LENGTH = horizon
        _old_tc = _cfg.trans_config
        _cfg.trans_config = _functools.partial(
            _old_tc.func,
            **{**_old_tc.keywords, 'max_blocks': horizon},
        )

    configs["learner_config"].EPOCHS = epochs
    configs["learner_config"].K_cpc  = k_cpc

    # N_SAMPLES: 200 for 3s_vs_5z (more transitions before each update)
    if args.env_name == "3s_vs_5z":
        configs["learner_config"].N_SAMPLES = 200

    # Batch size: reduce for large-agent maps (MMM = 10 agents)
    if n_agents >= 9:
        configs["learner_config"].MODEL_BATCH_SIZE = 20
        configs["learner_config"].BATCH_SIZE       = 20

    # 2m_vs_1z: higher entropy + distributional reward loss stabilizes learning (per MARIE)
    if args.env_name == "2m_vs_1z":
        configs["learner_config"].ENTROPY      = 0.01
        configs["learner_config"].use_ce_for_r = True

    # 3s_vs_5z: more exploration needed (per MARIE README)
    if args.env_name == "3s_vs_5z":
        configs["controller_config"].temperature = 2.0

    # --cpc_he: raise the entropy coefficient (overrides the per-map default) to
    # counter the late entropy-collapse seen on aggressive-config maps (e.g. 2s3z).
    if args.cpc_he:
        configs["learner_config"].ENTROPY = 0.01

    # ablation: override K_cpc (after the per-map default is set above)
    if args.k_cpc is not None:
        configs["learner_config"].K_cpc = args.k_cpc

    print(f"[Config] map={args.env_name}  n_agents={n_agents}  "
          f"H={horizon}  EPOCHS={epochs}  K_cpc={k_cpc}  "
          f"N_SAMPLES={configs['learner_config'].N_SAMPLES}  "
          f"ENTROPY={configs['learner_config'].ENTROPY}")

    current_date = datetime.datetime.now()
    current_date_string = current_date.strftime("%m%d")
    # current_date_string = "extreme_partial"

    # make run directory
    dir_prefix = args.env_name + '-'+ args.agent_conf if args.agent_conf is not None else args.env_name

    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f"/{current_date_string}_results") / args.env / (dir_prefix + f"-{args.tokenizer}")
    # curr_run = f"run{random.randint(1000, 9999)}"
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                            str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        os.makedirs(str(run_dir / "ckpt"))

    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "agent"), dst=run_dir / "agent")
    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "configs"), dst=run_dir / "configs")
    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "networks"), dst=run_dir / "networks")
    shutil.copyfile(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "train.py"), dst=run_dir / "train.py")
    
    print(f"Run files are saved at {str(run_dir)}\n")
    # -------------------

    configs["learner_config"].RUN_DIR = str(run_dir)
    configs["learner_config"].map_name = args.env_name

    group_name = generate_group_name(args, configs["learner_config"])
    postfix = "_reward-average" if args.average_r else ""
    postfix += f'_sample_temp={args.sample_temp}' if not configs["learner_config"].CONTINUOUS_ACTION else ""

    if configs["learner_config"].use_ce_for_r:
        run_name = f'(t_embed={configs["learner_config"].EMBED_DIM}) MACT_{args.env_name}_{args.agent_conf}_seed_{RANDOM_SEED}_' + format_numel_str_deci(args.steps) + f'_interval={configs["learner_config"].N_SAMPLES}_{rewards_prediction_config["loss_type"]}_bins{rewards_prediction_config["bins"]}' + postfix
    else:
        run_name = f'(t_embed={configs["learner_config"].EMBED_DIM}) MACT_{args.env_name}_{args.agent_conf}_seed_{RANDOM_SEED}_' + format_numel_str_deci(args.steps) + f'_interval={configs["learner_config"].N_SAMPLES}' + postfix

    prefix = f"({current_date_string}_T={args.temperature}_eval-T=1.0)" if not configs["learner_config"].CONTINUOUS_ACTION else f"({current_date_string})"

    _grp = "MACT_v2_abl" if args.run_tag else ("MACT_v2he" if args.cpc_he else ("MACT_v2" if args.cpc_v2 else "MACT"))
    if args.run_tag:
        run_name = f"{args.run_tag}_" + run_name
    global wandb
    import wandb
    wandb.init(
        project="mact_baselines",
        group=_grp,
        mode=args.mode,
        name=f"{_grp}_{args.env_name}_seed_{RANDOM_SEED}_{int(args.steps)//1000}K",
        config=configs["learner_config"].to_dict(),
        notes="",
    )

    exp = Experiment(steps=args.steps,
                     episodes=50000,
                     random_seed=RANDOM_SEED,
                     env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env),
                                                    obs_builder_config=configs["obs_builder_config"],
                                                    reward_config=configs["reward_config"]),
                     controller_config=configs["controller_config"],
                     learner_config=configs["learner_config"])

    train_dreamer(exp, n_workers=args.n_workers)
