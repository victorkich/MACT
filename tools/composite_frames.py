"""Stack recorded engine-frame arrays side by side into one labelled comparison video.

`render_sc2_true.py --npz` writes the raw frames for a single checkpoint. This joins two or
more of those into the side-by-side clips the page uses, with a caption bar naming each panel.

Panels are held on their last frame once they end, so a short episode does not truncate a long
one, and every panel is drawn at the same pixel scale -- neither method can appear zoomed
relative to the other.
"""
import argparse
import os

import numpy as np

BAR_H = 32
BG = (14, 17, 22)
FG = (236, 239, 243)
SUB = (150, 160, 172)


def _text(img, xy, s, color, scale=0.6, thick=1):
    import cv2
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color[::-1], thick, cv2.LINE_AA)


def unit_mask(frame):
    """Units against SMAC's grass: anything not green-dominant, and not off-map black."""
    a = frame.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (~((g > r + 8) & (g > b + 8))) & (a.sum(2) > 60)


def follow_crop(frames, cw=512, ch=288, smooth=9):
    """Re-frame a wide render onto the action.

    SC2 clamps its own camera to the playable area, so a policy that kites into a map corner
    ends up parked in the corner of frame no matter what camera target we request. Cropping
    after the fact has no such limit: track the units' centroid, smooth it so the view does not
    jitter, and cut a fixed window around it.
    """
    H, W = frames[0].shape[:2]
    cw, ch = min(cw, W), min(ch, H)
    cx, cy = [], []
    lx, ly = W / 2.0, H / 2.0
    for f in frames:
        m = unit_mask(f)
        ys, xs = np.nonzero(m)
        if len(xs) >= 50:
            lx, ly = xs.mean(), ys.mean()
        cx.append(lx); cy.append(ly)

    k = max(1, smooth | 1)
    pad = k // 2
    def smooth1(v):
        v = np.array(v, dtype=float)
        v = np.concatenate([np.repeat(v[:1], pad), v, np.repeat(v[-1:], pad)])
        return np.convolve(v, np.ones(k) / k, mode="valid")
    cx, cy = smooth1(cx), smooth1(cy)

    # Anything outside the playable area renders as black, and a crop that wanders into it shows
    # an ugly band. Clamp per frame to where the map is actually drawn -- a union over the clip
    # is useless here, because a moving camera lights every pixel at some point.
    out = []
    for f, x, y in zip(frames, cx, cy):
        lit = f.astype(np.int16).sum(2) > 60
        rows = np.nonzero(lit.any(axis=1))[0]
        cols = np.nonzero(lit.any(axis=0))[0]
        my0, my1 = (rows[0], rows[-1] + 1) if len(rows) else (0, H)
        mx0, mx1 = (cols[0], cols[-1] + 1) if len(cols) else (0, W)
        if my1 - my0 < ch:
            my0, my1 = 0, H
        if mx1 - mx0 < cw:
            mx0, mx1 = 0, W
        x0 = int(round(min(max(x - cw / 2.0, mx0), mx1 - cw)))
        y0 = int(round(min(max(y - ch / 2.0, my0), my1 - ch)))
        x0 = min(max(x0, 0), W - cw)
        y0 = min(max(y0, 0), H - ch)
        out.append(f[y0:y0 + ch, x0:x0 + cw])
    return np.stack(out)


def load(spec):
    """spec = path.npz::Title::Subtitle"""
    parts = spec.split("::")
    path = parts[0]
    title = parts[1] if len(parts) > 1 else os.path.basename(path)
    sub = parts[2] if len(parts) > 2 else ""
    d = np.load(path, allow_pickle=True)
    frames = d["frames"]
    won = bool(d["won"]) if "won" in d else None
    return dict(frames=frames, title=title, sub=sub, won=won)


def build(panels, gap=16, bar=BAR_H, max_frames=0):
    import cv2
    n = max(len(p["frames"]) for p in panels)
    # A panel that ends early is held on its last frame. That is honest -- one method finished
    # the fight and the other did not -- but many seconds of a frozen panel is dead air, so the
    # clip can be capped.
    if max_frames:
        n = min(n, max_frames)
    h = min(p["frames"].shape[1] for p in panels)
    w = min(p["frames"].shape[2] for p in panels)
    out_w = w * len(panels) + gap * (len(panels) - 1)
    out_h = h + bar
    out = []
    for i in range(n):
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:] = BG
        for j, p in enumerate(panels):
            f = p["frames"][min(i, len(p["frames"]) - 1)][:h, :w]
            x = j * (w + gap)
            canvas[bar:bar + h, x:x + w] = f
            if not bar:
                continue
            _text(canvas, (x + 10, 23), p["title"], FG, 0.62, 2)
            if p["sub"]:
                tw = cv2.getTextSize(p["title"], cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0]
                _text(canvas, (x + 18 + tw, 23), p["sub"], SUB, 0.5, 1)
        out.append(canvas)
    return out


def encode(frames, out_path, fps=18, crf=20):
    import imageio_ffmpeg
    h, w = frames[0].shape[:2]
    w -= w % 16
    h -= h % 16
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        out_path, (w, h), fps=fps, quality=None, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
        output_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"],
    )
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(f[:h, :w]))
    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="frames.npz::Title::Subtitle (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--follow", action="store_true",
                    help="crop-follow the action instead of using the raw wide frames")
    ap.add_argument("--crop", default="512x288")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap the composite length (0 = longest panel)")
    ap.add_argument("--no-labels", action="store_true",
                    help="omit the caption bar (for a single hero clip)")
    args = ap.parse_args()

    panels = [load(s) for s in args.panel]
    if args.follow:
        cw, ch = (int(v) for v in args.crop.lower().split("x"))
        for p in panels:
            p["frames"] = follow_crop(p["frames"], cw, ch)
    for p in panels:
        print("  %-28s %4d frames  won=%s" % (p["title"], len(p["frames"]), p["won"]))
    frames = build(panels, bar=0 if args.no_labels else BAR_H,
                   max_frames=args.max_frames)
    encode(frames, args.out, fps=args.fps)
    print("wrote %s (%d frames, %dx%d)"
          % (args.out, len(frames), frames[0].shape[1], frames[0].shape[0]))


if __name__ == "__main__":
    main()
