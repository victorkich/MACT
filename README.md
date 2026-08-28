# MACT

Action-Conditioned Transformers for Decentralized Multi-Agent World Models.
Published in *Transactions on Machine Learning Research*, 2026.

[Paper](https://openreview.net/forum?id=99nyrFfTJf) · [Project page](https://victorkich.github.io/MACT/)

MACT is a decentralized transformer world model for cooperative MARL. Each agent processes
discretized observation-action tokens with a shared transformer, one cross-agent Perceiver step
supplies global context under CTDE, and an action-conditioned contrastive objective (AC-CPC)
predicts future latent representations over a short horizon given the actions the agent intends
to take.

## Install

Python 3.7 and a StarCraft II installation are required.

```bash
pip install -r requirements.txt
export SC2PATH=/path/to/StarCraftII
```

Results in the paper use StarCraft II build **SC2.4.1.2.60604** (Base60321) with the SMAC maps.
SMAC results are not comparable across game builds, so pin this one to reproduce the numbers.

## Train

```bash
python train.py --env starcraft --env_name 3s_vs_3z --seed 1 --steps 50000
```

Imagination horizon, policy epochs and the AC-CPC horizon are set per map inside `train.py`,
following the paper's appendix, so the command above is the same for every map:

| maps | H | epochs | K_cpc |
|---|---|---|---|
| `so_many_baneling`, `2s3z` | 5 | 30 | 5 |
| more than 5 agents | 8 | 10 | 8 |
| 5 agents or fewer | 15 | 4 | 8 |

Two maps take additional overrides that the script also applies: `2m_vs_1z` raises the entropy
coefficient and uses a distributional reward loss, and `3s_vs_5z` samples more transitions and
raises the exploration temperature.

Budgets are 50k environment steps, except `3s_vs_5z`, `corridor` and `6h_vs_8z` at 200k.

Useful flags: `--mode disabled` turns off Weights & Biases logging, `--k_cpc` overrides the
AC-CPC horizon for the ablation, and `--cpc_v2` selects latent-space augmentation.

## Layout

```
train.py            entry point, per-map hyperparameters
agent/
  models/           world model, tokenizer, transformer, Perceiver, AC-CPC
  learners/         world-model and actor-critic updates
  controllers/      decentralized execution (tokenizer + actor only)
  runners/          training loop
  workers/          environment workers
networks/dreamer/   actor and critic
configs/            model and environment configuration
env/starcraft/      SMAC wrapper
```

At execution time only the tokenizer and each agent's own actor are used. The world model is
needed for training and not for acting.

## Baselines

The baselines are the authors' own implementations, run unmodified under one harness with
matched budgets, a pinned game build and the same 100-episode greedy evaluation:

| method | source |
|---|---|
| MARIE | https://github.com/breez3young/MARIE |
| MAMBA | https://github.com/jbr-ai-labs/mamba |
| MATWM | https://github.com/tomdeihim/MATWM |
| MAPPO | https://github.com/marlbenchmark/on-policy |
| MAT | https://github.com/PKU-MARL/Multi-Agent-Transformer |
| MBVD | https://github.com/HarryYangMLR/MBVD |

They are not vendored here. Clone each one, install its own requirements, and run it against the
same SC2 build and step budgets.

## Citation

```bibtex
@article{kich2026mact,
  title   = {Action-Conditioned Transformers for Decentralized Multi-Agent World Models},
  author  = {Kich, Victor A. and Yamamori, Satoshi and de Jesus, Junior C. and Morimoto, Jun},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  issn    = {2835-8856},
  url     = {https://openreview.net/forum?id=99nyrFfTJf}
}
```
