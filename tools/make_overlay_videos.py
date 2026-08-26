"""Draw MACT's internals onto the recorded StarCraft II frames.

Four overlays, all from recordings produced by record_overlay_data.py:

  ghosts     where the world model thinks each enemy will be, 1..H steps ahead
  strip      the ghosts plus a per-horizon fidelity bar, measured against a static baseline
  drift      MACT and MARIE side by side, so the difference in prediction quality is visible
  fork       one frozen state rolled forward under two different action windows

The prediction is relative (SMAC gives an agent no absolute self-position), so a prediction
made at t for horizon k is placed at the observing agent's TRUE position at t+k. That shows
what the model got right about the relative geometry without charging it for an absolute
coordinate it was never given. Ghosts are averaged over the agents that can see each enemy.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mact_introspect import project

BG = (13, 16, 21)
FG = (238, 241, 245)
MUTED = (150, 160, 172)
# horizon colours: near predictions bright, far ones cooler and fainter
HCOL = [(255, 209, 102), (255, 155, 84), (247, 106, 106), (200, 92, 168), (140, 108, 220)]


def _cv():
    import cv2
    return cv2


def text(img, xy, s, colour=FG, scale=0.5, thick=1):
    cv2 = _cv()
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def predicted_enemy_world(rec, t, k):
    """Absolute predicted enemy positions for horizon k, from the prediction made at step t.

    Averaged over agents: each agent predicts the enemy relative to itself, and anchoring at
    that agent's true future position turns each into an absolute estimate. Agents that
    predicted nothing (all-zero offsets, meaning the enemy was not visible) are skipped.
    """
    allies, imagined = rec["allies"], rec["imagined"]
    T = len(allies)
    tk = min(t + k + 1, T - 1)
    n_agents, n_en = imagined.shape[2], imagined.shape[3]
    out = np.full((n_en, 2), np.nan)
    for j in range(n_en):
        est = []
        for i in range(n_agents):
            if allies[tk, i, 2] <= 0 or not np.isfinite(allies[tk, i, 0]):
                continue
            rel = imagined[t, k, i, j]
            if not np.isfinite(rel).all() or np.allclose(rel, 0):
                continue
            est.append(allies[tk, i, :2] + rel)
        if est:
            out[j] = np.mean(est, axis=0)
    return out


def draw_ghosts(img, rec, t, H, homo, horizons=None, scale=None):
    cv2 = _cv()
    cam = rec["cams"][t]
    if scale is None:
        scale = img.shape[1] / 768.0
    horizons = horizons if horizons is not None else range(H)
    for k in horizons:
        pred = predicted_enemy_world(rec, t, k)
        good = np.isfinite(pred).all(axis=1)
        if not good.any():
            continue
        pts = project(homo, pred[good], cam)
        col = HCOL[min(k, len(HCOL) - 1)]
        r = int(round(max(3, 9 - k) * scale))
        thick = max(2, int(round(2 * scale)))
        for p in pts:
            x, y = int(round(p[0])), int(round(p[1]))
            if -40 < x < img.shape[1] + 40 and -40 < y < img.shape[0] + 40:
                overlay = img.copy()
                cv2.circle(overlay, (x, y), r, col, thick, cv2.LINE_AA)
                cv2.addWeighted(overlay, max(0.25, 0.85 - 0.13 * k), img, 1 - max(0.25, 0.85 - 0.13 * k), 0, img)
    return img


def bar_panel(w, h, rec, t, H, title, scale=None):
    """Per-horizon fidelity: how far the prediction made at t landed from the truth,
    against what a 'nothing moves' guess would have scored."""
    cv2 = _cv()
    if scale is None:
        scale = w / 768.0
    panel = np.zeros((h, w, 3), np.uint8)
    panel[:] = BG
    text(panel, (int(12 * scale), int(22 * scale)), title, FG, 0.5 * scale, max(1, int(scale)))
    enemies = rec["enemies"]
    T = len(enemies)
    x0 = int(14 * scale)
    y0 = int(38 * scale)
    gap = int(8 * scale)
    bw = (w - 2 * x0 - (H - 1) * gap) // H
    foot = int(26 * scale)
    for k in range(H):
        tk = min(t + k + 1, T - 1)
        pred = predicted_enemy_world(rec, t, k)
        errs, base = [], []
        for j in range(enemies.shape[1]):
            if enemies[tk, j, 2] <= 0 or not np.isfinite(pred[j]).all():
                continue
            errs.append(np.hypot(*(pred[j] - enemies[tk, j, :2])))
            base.append(np.hypot(*(enemies[t, j, :2] - enemies[tk, j, :2])))
        x = x0 + k * (bw + gap)
        hh = h - y0 - foot
        cv2.rectangle(panel, (x, y0), (x + bw, y0 + hh), (40, 46, 54), -1)
        if errs:
            e, b = float(np.mean(errs)), float(np.mean(base))
            frac = float(np.clip(1.0 - e / 3.0, 0.03, 1.0))
            col = HCOL[min(k, len(HCOL) - 1)]
            cv2.rectangle(panel, (x, y0 + int(hh * (1 - frac))), (x + bw, y0 + hh), col, -1)
            if b > 0:
                mark = y0 + int(hh * (1 - float(np.clip(1.0 - b / 3.0, 0.03, 1.0))))
                cv2.line(panel, (x - 2, mark), (x + bw + 2, mark), (235, 238, 242), 2, cv2.LINE_AA)
        text(panel, (x + 2, h - int(8 * scale)), "k=%d" % (k + 1), MUTED, 0.4 * scale, max(1, int(scale)))
    if w > 600:
        cap = "white line = 'nothing moves' baseline"
        cw_ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX, 0.38 * scale, max(1, int(scale)))[0][0]
        text(panel, (w - cw_ - int(12 * scale), int(22 * scale)), cap,
             MUTED, 0.38 * scale, max(1, int(scale)))
    return panel


def encode(frames, out_path, fps=12, crf=20):
    import imageio_ffmpeg
    h, w = frames[0].shape[:2]
    w -= w % 16
    h -= h % 16
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    wr = imageio_ffmpeg.write_frames(out_path, (w, h), fps=fps, quality=None, codec="libx264",
                                     pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                     output_params=["-crf", str(crf), "-preset", "slow",
                                                    "-movflags", "+faststart"])
    wr.send(None)
    for f in frames:
        wr.send(np.ascontiguousarray(f[:h, :w]))
    wr.close()
    print("wrote %s  (%d frames, %dx%d)" % (out_path, len(frames), w, h))


def banner(w, title, sub, hgt=34, scale=None):
    if scale is None:
        scale = w / 768.0
    hgt = int(round(hgt * scale))
    b = np.zeros((hgt, w, 3), np.uint8)
    b[:] = BG
    th = max(1, int(round(2 * scale)))
    text(b, (int(12 * scale), int(23 * scale)), title, FG, 0.6 * scale, th)
    cv2 = _cv()
    tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, th)[0][0]
    text(b, (int(20 * scale) + tw, int(23 * scale)), sub, MUTED, 0.45 * scale, max(1, int(scale)))
    return b


# ---------------------------------------------------------------------------
# the four videos
# ---------------------------------------------------------------------------
def video_ghosts(rec, homo, out, fps=12):
    """1. Where the world model thinks everyone will be."""
    frames, H = rec["frames"], int(rec["horizon"])
    out_frames = []
    for t in range(len(frames)):
        img = frames[t].copy()
        draw_ghosts(img, rec, t, H, homo)
        b = banner(img.shape[1], "MACT", "imagined enemy positions, 1 to %d steps ahead" % H)
        legend = np.zeros((30, img.shape[1], 3), np.uint8); legend[:] = BG
        cv2 = _cv()
        for k in range(H):
            x = 14 + k * 92
            cv2.circle(legend, (x, 13), max(3, 9 - k), HCOL[min(k, len(HCOL) - 1)], 2, cv2.LINE_AA)
            text(legend, (x + 12, 18), "k=%d" % (k + 1), MUTED, 0.42, 1)
        out_frames.append(np.vstack([b, img, legend]))
    encode(out_frames, out, fps=fps)


def video_strip(rec, homo, out, fps=12):
    """3. The same ghosts with a per-horizon fidelity bar underneath."""
    frames, H = rec["frames"], int(rec["horizon"])
    out_frames = []
    for t in range(len(frames)):
        img = frames[t].copy()
        draw_ghosts(img, rec, t, H, homo)
        b = banner(img.shape[1], "MACT", "prediction fidelity per horizon, live")
        sc_ = img.shape[1] / 768.0
        panel = bar_panel(img.shape[1], int(118 * sc_), rec, t, H,
                          "how close each horizon's prediction landed")
        out_frames.append(np.vstack([b, img, panel]))
    encode(out_frames, out, fps=fps)


def video_drift(rec_a, rec_b, homo, out, fps=12, labels=("MACT", "MARIE")):
    """4. Two models' predictions side by side."""
    Ha, Hb = int(rec_a["horizon"]), int(rec_b["horizon"])
    n = min(len(rec_a["frames"]), len(rec_b["frames"]))
    h = min(rec_a["frames"].shape[1], rec_b["frames"].shape[1])
    w = min(rec_a["frames"].shape[2], rec_b["frames"].shape[2])
    gap = 16
    out_frames = []
    for t in range(n):
        panes = []
        for rec, H, lab in ((rec_a, Ha, labels[0]), (rec_b, Hb, labels[1])):
            img = rec["frames"][t][:h, :w].copy()
            draw_ghosts(img, rec, t, H, homo)
            panes.append(np.vstack([banner(w, lab, "imagined vs actual"), img]))
        canvas = np.zeros((panes[0].shape[0], w * 2 + gap, 3), np.uint8)
        canvas[:] = BG
        canvas[:, :w] = panes[0]
        canvas[:, w + gap:] = panes[1]
        out_frames.append(canvas)
    encode(out_frames, out, fps=fps)


def video_fork(rec, homo, out, forks, freeze_at, fps=12, hold=90):
    """2. One frozen state, two different action windows, two different predicted futures."""
    cv2 = _cv()
    img0 = rec["frames"][freeze_at]
    cam = rec["cams"][freeze_at]
    h, w = img0.shape[:2]
    gap = 16
    out_frames = []
    for i in range(hold):
        panes = []
        reveal = min(1.0, i / float(hold * 0.45))
        for name, pred_by_k in forks:
            img = img0.copy()
            kmax = int(round(reveal * len(pred_by_k)))
            for k in range(kmax):
                pred = pred_by_k[k]
                good = np.isfinite(pred).all(axis=1)
                if not good.any():
                    continue
                pts = project(homo, pred[good], cam)
                col = HCOL[min(k, len(HCOL) - 1)]
                for p in pts:
                    x, y = int(round(p[0])), int(round(p[1]))
                    if -40 < x < w + 40 and -40 < y < h + 40:
                        ov = img.copy()
                        cv2.circle(ov, (x, y), max(3, 9 - k), col, 2, cv2.LINE_AA)
                        a = max(0.25, 0.85 - 0.13 * k)
                        cv2.addWeighted(ov, a, img, 1 - a, 0, img)
            panes.append(np.vstack([banner(w, name, "same state, different plan"), img]))
        canvas = np.zeros((panes[0].shape[0], w * 2 + gap, 3), np.uint8)
        canvas[:] = BG
        canvas[:, :w] = panes[0]
        canvas[:, w + gap:] = panes[1]
        out_frames.append(canvas)
    encode(out_frames, out, fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--rec-b", default="")
    ap.add_argument("--homography", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--which", default="ghosts,strip,drift")
    ap.add_argument("--fork", default="", help="fork npz from make_fork.py")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--sequence", default="",
                    help="comma list of name=path.npz, played in order")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    homo = np.load(args.homography)
    rec = np.load(args.rec, allow_pickle=True)
    os.makedirs(args.outdir, exist_ok=True)
    which = [x.strip() for x in args.which.split(",") if x.strip()]

    if "ghosts" in which:
        video_ghosts(rec, homo, os.path.join(args.outdir, "1_ghost_futures.mp4"), args.fps)
    if "strip" in which:
        video_strip(rec, homo, os.path.join(args.outdir, "3_horizon_fidelity.mp4"), args.fps)
    if "fork" in which and args.fork:
        f = np.load(args.fork, allow_pickle=True)
        t = int(f["freeze_at"])
        names = [str(x) for x in f["names"]]
        allies = rec["allies"]
        forks = []
        for bi, nm in enumerate(names):
            rel = f["branch_%d" % bi]                     # (H, n_agents, n_enemies, 2)
            per_k = []
            for k in range(rel.shape[0]):
                tk = min(t + k + 1, len(allies) - 1)
                n_en = rel.shape[2]
                abs_pred = np.full((n_en, 2), np.nan)
                for j in range(n_en):
                    est = [allies[tk, i, :2] + rel[k, i, j]
                           for i in range(rel.shape[1])
                           if allies[tk, i, 2] > 0 and np.isfinite(rel[k, i, j]).all()
                           and not np.allclose(rel[k, i, j], 0)]
                    if est:
                        abs_pred[j] = np.mean(est, axis=0)
                per_k.append(abs_pred)
            forks.append((nm, per_k))
        video_fork(rec, homo, os.path.join(args.outdir, "2_action_conditioned_fork.mp4"),
                   forks, t, args.fps)

    if "sequence" in which and args.sequence:
        recs = []
        for spec in args.sequence.split(","):
            name, path = spec.split("=", 1)
            recs.append((name, np.load(path, allow_pickle=True)))
        video_sequence(recs, os.path.join(args.outdir, "9_sequence_five_maps.mp4"), args.fps)

    if "planexec" in which and args.fork and args.rec_b:
        video_plan_vs_execution(rec, np.load(args.rec_b, allow_pickle=True),
                                np.load(args.fork, allow_pickle=True), homo,
                                os.path.join(args.outdir, "7_plan_vs_execution_8m.mp4"),
                                labels=[args.label_a, args.label_b], fps=args.fps)

    if "matrix" in which and args.fork:
        video_matrix(rec, np.load(args.fork, allow_pickle=True), homo,
                     os.path.join(args.outdir, "5_matrix_plans_and_fidelity.mp4"), args.fps)

    if "drift" in which and args.rec_b:
        rec_b = np.load(args.rec_b, allow_pickle=True)
        video_drift(rec, rec_b, homo, os.path.join(args.outdir, "4_mact_vs_marie_drift.mp4"), args.fps)



def video_matrix(rec, fork, homo, out, fps=12):
    """A 2x2 combining the two overlays Victor picked.

    top row     the same state rolled forward under two different plans, live at every step,
                so the action-conditioning is visible as the episode plays rather than frozen
    bottom left the on-policy ghosts, what the model actually predicts as the agents act
    bottom right per-horizon fidelity, how close each of those predictions landed

    Reading across the top shows what changes when the plan changes. Reading down the left
    shows what the model believes and, underneath, whether it was right.
    """
    cv2 = _cv()
    frames, H = rec["frames"], int(rec["horizon"])
    allies = rec["allies"]
    names = [str(x) for x in fork["names"]]
    branches = [fork["branch_0"], fork["branch_1"]]
    n = min(len(frames), branches[0].shape[0])
    h, w = frames[0].shape[:2]
    cw, ch = w // 2, h // 2
    gap = 12

    def abs_from_rel(rel_k, tk):
        n_en = rel_k.shape[1]
        out_p = np.full((n_en, 2), np.nan)
        for j in range(n_en):
            est = [allies[tk, i, :2] + rel_k[i, j] for i in range(rel_k.shape[0])
                   if allies[tk, i, 2] > 0 and np.isfinite(rel_k[i, j]).all()
                   and not np.allclose(rel_k[i, j], 0)]
            if est:
                out_p[j] = np.mean(est, axis=0)
        return out_p

    out_frames = []
    for t in range(n):
        cam = rec["cams"][t]
        cells = []
        # --- top row: the two plans ---
        for bi, nm in enumerate(names):
            img = cv2.resize(frames[t], (cw, ch), interpolation=cv2.INTER_AREA)
            sx, sy = cw / float(w), ch / float(h)
            for k in range(H):
                pred = abs_from_rel(branches[bi][t, k], min(t + k + 1, len(allies) - 1))
                good = np.isfinite(pred).all(axis=1)
                if not good.any():
                    continue
                for p in project(homo, pred[good], cam):
                    x, y = int(round(p[0] * sx)), int(round(p[1] * sy))
                    if -30 < x < cw + 30 and -30 < y < ch + 30:
                        ov = img.copy()
                        cv2.circle(ov, (x, y), max(2, 6 - k), HCOL[min(k, len(HCOL) - 1)], 2, cv2.LINE_AA)
                        a = max(0.3, 0.9 - 0.13 * k)
                        cv2.addWeighted(ov, a, img, 1 - a, 0, img)
            cells.append(np.vstack([banner(cw, "PLAN %d" % (bi + 1), nm, 26), img]))
        # --- bottom left: on-policy ghosts ---
        g = frames[t].copy()
        draw_ghosts(g, rec, t, H, homo)
        g = cv2.resize(g, (cw, ch), interpolation=cv2.INTER_AREA)
        cells.append(np.vstack([banner(cw, "ACTUAL", "what the agents really do", 26), g]))
        # --- bottom right: fidelity ---
        panel = bar_panel(cw, ch, rec, t, H, "how close each horizon landed")
        cells.append(np.vstack([banner(cw, "FIDELITY", "vs 'nothing moves'", 26), panel]))

        cellh = cells[0].shape[0]
        canvas = np.zeros((cellh * 2 + gap, cw * 2 + gap, 3), np.uint8)
        canvas[:] = BG
        canvas[:cellh, :cw] = cells[0]
        canvas[:cellh, cw + gap:] = cells[1]
        canvas[cellh + gap:, :cw] = cells[2]
        canvas[cellh + gap:, cw + gap:] = cells[3]
        out_frames.append(canvas)
    encode(out_frames, out, fps=fps)


def video_plan_vs_execution(rec_a, rec_b, fork, homo, out, labels, fps=12):
    """Top row static: the PLAN the world model predicts for each focus target, frozen.
    Bottom row dynamic: that same focus actually executed, with per-horizon error bars.

    The two bottom panes are different episodes, because the agents really were forced onto
    different targets, so they diverge in the way the plans said they would.
    """
    cv2 = _cv()
    n = max(len(rec_a["frames"]), len(rec_b["frames"]))
    h, w = rec_a["frames"][0].shape[:2]
    cw, ch = w // 2, h // 2
    gap, barh = 12, 74
    names = [str(x) for x in fork["names"]]
    branches = [fork["branch_0"], fork["branch_1"]]
    plan_t = min(6, branches[0].shape[0] - 1)     # an early step: the plan before it plays out

    def abs_from_rel(rel_k, allies, tk):
        n_en = rel_k.shape[1]
        o = np.full((n_en, 2), np.nan)
        for j in range(n_en):
            est = [allies[tk, i, :2] + rel_k[i, j] for i in range(rel_k.shape[0])
                   if allies[tk, i, 2] > 0 and np.isfinite(rel_k[i, j]).all()
                   and not np.allclose(rel_k[i, j], 0)]
            if est:
                o[j] = np.mean(est, axis=0)
        return o

    # --- top row is STATIC: render each plan once and reuse it every frame ---
    static = []
    ref = rec_a
    for bi in range(2):
        img = cv2.resize(ref["frames"][plan_t], (cw, ch), interpolation=cv2.INTER_AREA)
        sx, sy = cw / float(w), ch / float(h)
        H = branches[bi].shape[1]
        for k in range(H):
            pred = abs_from_rel(branches[bi][plan_t, k], ref["allies"],
                                min(plan_t + k + 1, len(ref["allies"]) - 1))
            good = np.isfinite(pred).all(axis=1)
            if not good.any():
                continue
            for p in project(homo, pred[good], ref["cams"][plan_t]):
                x, y = int(round(p[0] * sx)), int(round(p[1] * sy))
                if -30 < x < cw + 30 and -30 < y < ch + 30:
                    ov = img.copy()
                    cv2.circle(ov, (x, y), max(2, 6 - k), HCOL[min(k, len(HCOL) - 1)], 2, cv2.LINE_AA)
                    a = max(0.3, 0.9 - 0.13 * k)
                    cv2.addWeighted(ov, a, img, 1 - a, 0, img)
        static.append(np.vstack([banner(cw, "PLAN", names[bi] + "  (predicted)", 26), img]))

    out_frames = []
    for t in range(n):
        cells = list(static)
        for rec, lab in ((rec_a, labels[0]), (rec_b, labels[1])):
            ti = min(t, len(rec["frames"]) - 1)
            g = rec["frames"][ti].copy()
            draw_ghosts(g, rec, ti, int(rec["horizon"]), homo)
            g = cv2.resize(g, (cw, ch - barh), interpolation=cv2.INTER_AREA)
            bars = bar_panel(cw, barh, rec, ti, int(rec["horizon"]), "error by horizon")
            won = " won" if bool(rec["won"]) and ti >= len(rec["frames"]) - 2 else ""
            cells.append(np.vstack([banner(cw, "EXECUTED", lab + won, 26), g, bars]))
        cellh = max(c.shape[0] for c in cells)
        cells = [np.vstack([c, np.zeros((cellh - c.shape[0], cw, 3), np.uint8)]) if c.shape[0] < cellh else c
                 for c in cells]
        canvas = np.zeros((cellh * 2 + gap, cw * 2 + gap, 3), np.uint8)
        canvas[:] = BG
        canvas[:cellh, :cw] = cells[0]
        canvas[:cellh, cw + gap:] = cells[1]
        canvas[cellh + gap:, :cw] = cells[2]
        canvas[cellh + gap:, cw + gap:] = cells[3]
        out_frames.append(canvas)
    encode(out_frames, out, fps=fps)


MAP_BLURB = {
    "8m": "8 Marines a side. Focus fire only pays off if several agents commit to one target.",
    "MMM": "Marines, Marauders and a healing Medivac. Roles differ, so the plan has to as well.",
    "so_many_baneling": "Banelings detonate on contact. A clumped formation loses instantly.",
    "3s_vs_3z": "Stalkers against Zealots. Kiting: stay out of melee range while firing.",
    "corridor": "SuperHard. Six Zealots against 24 Zerglings in a choke point.",
    "3s_vs_4z": "Outnumbered. Every method struggles here.",
    "2s_vs_1sc": "Two Stalkers against a Spine Crawler.",
}


def title_card(w, h, map_name, n_frames, scale=None):
    cv2 = _cv()
    if scale is None:
        scale = w / 768.0
    card = np.zeros((h, w, 3), np.uint8)
    card[:] = BG
    text(card, (int(40 * scale), int(h * 0.42)), map_name, FG, 1.5 * scale, max(2, int(3 * scale)))
    blurb = MAP_BLURB.get(map_name, "")
    if blurb:
        text(card, (int(42 * scale), int(h * 0.42) + int(38 * scale)), blurb, MUTED,
             0.52 * scale, max(1, int(scale)))
    return [card] * n_frames


def video_sequence(recs, out, fps=12, card_frames=14):
    """Chain the fidelity view across several maps, with a title card before each.

    Every map is drawn with the analytic projection, so no per-map calibration is involved and
    the geometry is identical throughout.
    """
    from mact_introspect import analytic_projection
    cv2 = _cv()
    out_frames = []
    W = H = None
    for map_name, rec in recs:
        frames = rec["frames"]
        Hh = int(rec["horizon"])
        fw, fh = frames.shape[2], frames.shape[1]
        sc_ = fw / 768.0
        homo = analytic_projection(fw, fh, 22.0)
        panel_h = int(118 * sc_)
        bh = int(34 * sc_)
        if W is None:
            W, H = fw, bh + fh + panel_h
        for t in range(len(frames)):
            img = frames[t].copy()
            draw_ghosts(img, rec, t, Hh, homo)
            b = banner(fw, "MACT", "%s  |  prediction fidelity per horizon" % map_name)
            panel = bar_panel(fw, panel_h, rec, t, Hh, "how close each horizon's prediction landed")
            out_frames.append(np.vstack([b, img, panel]))
    encode(out_frames, out, fps=fps)

if __name__ == "__main__":
    main()
