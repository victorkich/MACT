# MACT — Project Page

This branch holds the GitHub Pages site. The method code lives on `main`.

Project page for **Action-Conditioned Transformers for Decentralized Multi-Agent World Models**
(MACT), accepted at *Transactions on Machine Learning Research* (TMLR).

Live page: _(enable GitHub Pages on this repository to publish)_

OpenReview: <https://openreview.net/forum?id=99nyrFfTJf>

## Layout

```
index.html              single-file page (inline CSS)
assets/diagrams/        the paper's method figures — SVG (vector) with PNG companions
assets/plots/           the paper's result figures — SVG (vector) with PNG companions
assets/videos/          rollout videos captured from the StarCraft II engine, H.264 MP4
data/rollouts/          raw per-step rollout records
tools/                  scripts that produced the rollouts, videos and figures
```

## Assets

Every figure on the page is the paper's own, converted from the source PDF with
`pdftocairo -svg` so the vector text stays crisp, with a PNG companion alongside:

| File | Paper source |
|---|---|
| `diagrams/ac_cpc.svg` | `images/world_model.pdf` — world-model training (Fig. 2) |
| `diagrams/imagination.svg` | `images/behavior_learning.pdf` — behaviour learning (Fig. 3) |
| `plots/fig_headline.svg` | `charts/overall_bars.pdf` — overall mean/median win rate |
| `plots/fig_rliable.svg` | `charts/rliable_iqm.pdf` — IQM with 95% bootstrap CIs |
| `plots/fig_permap.svg` | `charts/_all_envs.pdf` — learning curves, all methods, all maps |
| `plots/abl_k/noise/cond.svg` | `plots/ablation_study/*.pdf` — the three ablation panels |

> Take figures from the paper's own PDFs, never by cropping a page out of the compiled
> manuscript. An earlier version of this page extracted the ablations as whole US-Letter pages, so
> each `viewBox` was `0 0 612 792` and the chart rendered tiny inside a tall empty box.

Videos are H.264 (yuv420p, `+faststart`) so they play inline everywhere.

## Rollout videos

The clips are captured from the **StarCraft II engine itself** — the same 3D view the paper's
figures show — not from a schematic re-drawing of unit positions.

```bash
python tools/render_sc2_true.py \
    --repo /path/to/MACT --map 3s_vs_3z \
    --ckpt /path/to/model_final.pth \
    --out assets/videos/mact_3s_vs_3z.mp4 \
    --seed 1 --pick first_win --substeps 3
```

Rollouts run through the same evaluation path the paper uses — the VQ tokenizer plus each agent's
decentralized actor. The world model is used for imagination during *training* only, so inference
needs no GPU.

Requires `torch`, `smac`, `pysc2`, `imageio-ffmpeg`, and a StarCraft II installation (`SC2PATH`).
All results use the pinned build **SC2.4.1.2.60604** (Base60321).

### Getting the engine to render at all

SC2's Linux build *can* render, contrary to the usual claim — but two things must be right, and
`render_sc2_true.py` handles both:

1. **Loader environment.** SC2 prepends its own `Libs/` to the loader path, and the single file
   there is a 2018 `libstdc++.so.6` (max `GLIBCXX_3.4.21`). Mesa's `swrast_dri.so` needs
   `3.4.29`, so the driver fails to load, EGL never initialises, and the game falls back to a stub
   renderer — reporting only `Rendered interface was requested but not available`. The script
   re-execs itself with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` and
   `LIBGL_ALWAYS_SOFTWARE=1`. A working launch logs `Configure: render interface enabled`.
2. **Surviving a restart.** SMAC hardcodes `want_rgb=False` and an interface with no render
   section, so `_launch` is patched to ask for one. The subtler part: `env.reset()` goes through
   `full_restart()`, which closes the process and builds a **new** `RemoteController` — orphaning
   any capture wrapper bound to the old one. Nothing errors; the episode runs to completion and
   wins, just with 9 frames instead of 119. The grabber therefore re-attaches inside `_launch`,
   which every relaunch path goes through.
3. **Waiting for the frame.** Under software rasterization on a loaded machine the renderer is
   often not ready when the observation returns, and `render_data` comes back empty — again with
   no error. Capture polls until the frame appears rather than skipping it.

Frames are captured on sub-steps of the environment's `step_mul`, which raises the frame rate
without changing a single action the policy takes: the same total game loops elapse either way.

**Framing.** SC2 clamps its own camera to the playable area, so a policy that kites into a map
corner stays in the corner of frame no matter what camera target is requested — asking to pan by
4.5 world units (~157 px) moved the units 8 px. `composite_frames.py --follow` therefore re-frames
after the fact: it tracks the units, smooths the path, and crops a window around them, clamped per
frame to where the map is actually drawn so the view never wanders into off-map black.

> The clips are samples from a single checkpoint, recorded to illustrate behaviour. They are not
> the paper's evaluation, which is 100-episode greedy evaluation over seeds. Captions on the page
> state the measured outcome alongside the published value.

## Citation

```bibtex
@article{kich2026mact,
  title   = {Action-Conditioned Transformers for Decentralized Multi-Agent World Models},
  author  = {Kich, Victor A. and Yamamori, Satoshi and de Jesus, Junior C. and Morimoto, Jun},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2026},
  issn    = {2835-8856},
  url     = {https://openreview.net/forum?id=99nyrFfTJf}
}
```
