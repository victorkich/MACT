"""Render recorded SMAC rollouts into web-ready H.264 videos.

Draws a clean top-down tactical view: allies vs enemies, health bars, and attack lines
(which is what makes focus-fire legible). Multiple rollouts are composited side by side
into a single MP4, matching the project-page pattern.
"""
import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow
import matplotlib.patheffects as pe
import imageio_ffmpeg

# Palette aligned with the project page
ALLY = "#1d4ed8"
ALLY_L = "#93b4fd"
ENEMY = "#dc2626"
ENEMY_L = "#fca5a5"
DEAD = "#d4d4d8"
INK = "#111111"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BG = "#ffffff"

SUBSTEPS = 4  # interpolated frames between env steps -> smooth motion

# labels share the unit colours, so they need a halo to stay legible over a marker
HALO = [pe.withStroke(linewidth=3.0, foreground="white")]


def bounds_of(episodes, pad=1.6):
    xs, ys = [], []
    for ep in episodes:
        for fr in ep["frames"]:
            for u in fr["allies"] + fr["enemies"]:
                if u["alive"]:
                    xs.append(u["x"]); ys.append(u["y"])
    if not xs:
        return (0, 32, 0, 32)
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    # square aspect so units never look stretched
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2
    return (cx - half, cx + half, cy - half, cy + half)


def lerp_unit(u0, u1, t):
    if u0 is None:
        return u1
    return dict(
        x=u0["x"] + (u1["x"] - u0["x"]) * t,
        y=u0["y"] + (u1["y"] - u0["y"]) * t,
        hp=u0["hp"] + (u1["hp"] - u0["hp"]) * t,
        hp_max=u1["hp_max"], sh=u0["sh"] + (u1["sh"] - u0["sh"]) * t,
        sh_max=u1["sh_max"], utype=u1["utype"],
        alive=u1["alive"] if t > 0.5 else u0["alive"],
    )


def radius_for(u, span):
    base = 0.022 + 0.013 * min(1.0, u["hp_max"] / 150.0)
    return base * span


def diamond(cx, cy, r):
    return [(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)]


def draw_panel(ax, frame, prev, t, bnds, title, subtitle, outcome=None, step_label=""):
    x0, x1, y0, y1 = bnds
    span = x1 - x0
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # subtle grid for spatial reference
    for gx in np.arange(math.floor(x0), math.ceil(x1) + 1, 2):
        ax.axvline(gx, color=GRID, lw=0.6, zorder=0)
    for gy in np.arange(math.floor(y0), math.ceil(y1) + 1, 2):
        ax.axhline(gy, color=GRID, lw=0.6, zorder=0)

    allies = [lerp_unit(prev["allies"][i] if prev else None, u, t)
              for i, u in enumerate(frame["allies"])]
    enemies = [lerp_unit(prev["enemies"][i] if prev else None, u, t)
               for i, u in enumerate(frame["enemies"])]

    # attack lines first (under units); count focus to highlight concentrated fire
    atk = frame.get("attack", [])
    focus = {}
    for i, tgt in enumerate(atk):
        if tgt is None or tgt < 0 or tgt >= len(enemies):
            continue
        a, e = allies[i], enemies[tgt]
        if not a["alive"] or not e["alive"]:
            continue
        focus[tgt] = focus.get(tgt, 0) + 1
        ax.plot([a["x"], e["x"]], [a["y"], e["y"]], color=ENEMY,
                lw=1.0, alpha=0.32, zorder=1, solid_capstyle="round")

    # ring the enemy under concentrated fire -- the paper's motivating "focus fire" case
    for tgt, cnt in focus.items():
        if cnt < 2:
            continue
        e = enemies[tgt]
        r = radius_for(e, span)
        ax.add_patch(Circle((e["x"], e["y"]), r * 2.1, facecolor="none",
                            edgecolor=ENEMY, lw=1.4, alpha=0.55, zorder=2,
                            linestyle=(0, (2.5, 2))))

    def blit(units, col, col_l, is_ally):
        for u in units:
            r = radius_for(u, span)
            if not u["alive"]:
                ax.add_patch(Circle((u["x"], u["y"]), r * 0.5, facecolor="none",
                                    edgecolor=DEAD, lw=0.9, zorder=2, alpha=0.8))
                continue
            if is_ally:
                ax.add_patch(Circle((u["x"], u["y"]), r, facecolor=col,
                                    edgecolor="white", lw=1.0, zorder=4))
            else:
                ax.add_patch(plt.Polygon(diamond(u["x"], u["y"], r * 1.18),
                                         facecolor=col, edgecolor="white",
                                         lw=1.0, zorder=4))
            # slim health bar
            frac = max(0.0, min(1.0, u["hp"] / max(u["hp_max"], 1e-6)))
            bw, bh = r * 1.8, r * 0.26
            bx, by = u["x"] - bw / 2, u["y"] + r * 1.45
            ax.add_patch(Rectangle((bx, by), bw, bh, facecolor="#0000001a",
                                   edgecolor="none", zorder=5))
            ax.add_patch(Rectangle((bx, by), bw * frac, bh, facecolor=col_l,
                                   edgecolor="none", zorder=6))

    blit(enemies, ENEMY, ENEMY_L, False)
    blit(allies, ALLY, ALLY_L, True)

    n_a = sum(u["alive"] for u in allies)
    n_e = sum(u["alive"] for u in enemies)

    ax.text(0.5, 1.075, title, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=13, fontweight="600", color=INK, zorder=10)
    ax.text(0.5, 1.020, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color=MUTED, zorder=10)
    ax.text(0.015, 0.015, f"● allies {n_a}", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=10, color=ALLY, fontweight="600", zorder=10, path_effects=HALO)
    ax.text(0.985, 0.015, f"◆ enemies {n_e}", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=10, color=ENEMY, fontweight="600", zorder=10, path_effects=HALO)
    if outcome:
        col = "#15803d" if outcome == "WIN" else "#b91c1c"
        ax.text(0.5, 0.015, outcome, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=11, fontweight="700", color=col, zorder=10, path_effects=HALO)
    elif step_label:
        ax.text(0.5, 0.015, step_label, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9.5, color=MUTED, zorder=10, path_effects=HALO)


def render(panels, out_path, fps=24, width=None, height=520, hold_end=18):
    """panels: list of dicts {episode, title, subtitle}"""
    n = len(panels)
    per_w = 520
    W = width or per_w * n
    H = height
    dpi = 100
    figw, figh = W / dpi, H / dpi

    # Every panel gets the SAME span -- so neither method looks zoomed relative to the
    # other -- but is centred on its own trajectory. A shared union box would leave both
    # episodes as specks in opposite corners when they occupy different map regions.
    raw = [bounds_of([p["episode"]]) for p in panels]
    span = max(b[1] - b[0] for b in raw)
    bnds = []
    for x0, x1, y0, y1 in raw:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bnds.append((cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2))
    max_len = max(len(p["episode"]["frames"]) for p in panels)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        out_path, (W, H), fps=fps, quality=None,
        macro_block_size=1,
        output_params=["-crf", "20", "-pix_fmt", "yuv420p",
                       "-profile:v", "high", "-movflags", "+faststart"],
    )
    writer.send(None)

    total = (max_len - 1) * SUBSTEPS + hold_end
    for gi in range(total):
        step = min(gi // SUBSTEPS, max_len - 2) if max_len > 1 else 0
        t = (gi % SUBSTEPS) / SUBSTEPS
        if gi >= (max_len - 1) * SUBSTEPS:
            step, t = max_len - 2 if max_len > 1 else 0, 1.0

        fig, axes = plt.subplots(1, n, figsize=(figw, figh), dpi=dpi)
        if n == 1:
            axes = [axes]
        fig.patch.set_facecolor(BG)

        for ax, p, b in zip(axes, panels, bnds):
            frames = p["episode"]["frames"]
            si = min(step + 1, len(frames) - 1)
            done = si >= len(frames) - 1
            outcome = ("WIN" if p["episode"]["won"] else "LOSS") if done else None
            draw_panel(ax, frames[si], frames[si - 1] if si > 0 else None, t, b,
                       p["title"], p["subtitle"], outcome,
                       step_label=f"step {si}/{len(frames) - 1}")

        fig.subplots_adjust(left=0.012, right=0.988, top=0.86, bottom=0.03, wspace=0.06)
        for k in range(1, n):  # divider between panels
            fig.add_artist(plt.Line2D([k / n, k / n], [0.035, 0.90],
                                      color="#e5e7eb", lw=1.1))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.send(np.ascontiguousarray(buf))
        plt.close(fig)

    writer.close()
    print(f"wrote {out_path} ({total} frames, {total/fps:.1f}s)")


def load(path, pick="first_win"):
    with open(path) as f:
        d = json.load(f)
    eps = d["episodes"]
    if pick == "first_win":
        for e in eps:
            if e["won"]:
                return d, e
    elif pick == "first_loss":
        for e in eps:
            if not e["won"]:
                return d, e
    elif pick == "longest":
        return d, max(eps, key=lambda e: e["steps"])
    return d, eps[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="json:title:subtitle:pick")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--height", type=int, default=520)
    args = ap.parse_args()

    panels = []
    for spec in args.panel:
        parts = spec.split("::")
        path, title = parts[0], parts[1]
        subtitle = parts[2] if len(parts) > 2 else ""
        pick = parts[3] if len(parts) > 3 else "first_win"
        _, ep = load(path, pick)
        panels.append(dict(episode=ep, title=title, subtitle=subtitle))

    render(panels, args.out, fps=args.fps, height=args.height)


if __name__ == "__main__":
    main()
