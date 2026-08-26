"""Is the world model's imagined future accurate enough to draw?

For every step t and horizon k, the prediction made at t is compared with what actually
happened at t+k. The prediction is relative (SMAC observations carry no absolute self
position), so it is anchored at the observing agent's TRUE position at t+k. That charges the
model for the relative geometry it was asked to predict and nothing else.

The number that matters is not the raw error but the comparison against a static baseline that
assumes nothing moves. A world model that merely reports "everything stays put" can look
accurate on a slow map while carrying no predictive content, and ghosts drawn from it would be
visually convincing and scientifically empty.
"""
import argparse

import numpy as np


def evaluate(rec):
    allies = rec["allies"]          # (T, n_agents, 3) x, y, health
    enemies = rec["enemies"]        # (T, n_enemies, 3)
    imagined = rec["imagined"]      # (T, H, n_agents, n_enemies, 2) relative offsets
    T, H = imagined.shape[0], imagined.shape[1]

    per_k_model, per_k_static = [[] for _ in range(H)], [[] for _ in range(H)]
    for t in range(T):
        for k in range(H):
            tk = t + k + 1
            if tk >= T:
                continue
            for i in range(allies.shape[1]):
                if not np.isfinite(allies[tk, i, 0]) or allies[tk, i, 2] <= 0:
                    continue
                own_future = allies[tk, i, :2]
                for j in range(enemies.shape[1]):
                    if enemies[tk, j, 2] <= 0 or not np.isfinite(enemies[tk, j, 0]):
                        continue
                    rel = imagined[t, k, i, j]
                    if not np.isfinite(rel).all() or np.allclose(rel, 0):
                        continue
                    pred = own_future + rel
                    truth = enemies[tk, j, :2]
                    per_k_model[k].append(float(np.hypot(*(pred - truth))))
                    # static baseline: the enemy never moves from where it was at t
                    per_k_static[k].append(float(np.hypot(*(enemies[t, j, :2] - truth))))
    return per_k_model, per_k_static


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    args = ap.parse_args()
    rec = np.load(args.npz, allow_pickle=True)
    print("GHOST %d steps, horizon %s, won=%s"
          % (rec["imagined"].shape[0], rec["horizon"], bool(rec["won"])))
    m, s = evaluate(rec)
    print("GHOST  k   n      model err   static err   verdict")
    beats = 0
    for k in range(len(m)):
        if not m[k]:
            print("GHOST %2d   no comparable samples" % (k + 1))
            continue
        mm, ss = float(np.mean(m[k])), float(np.mean(s[k]))
        good = mm < ss
        beats += int(good)
        print("GHOST %2d %4d   %8.2f    %8.2f   %s"
              % (k + 1, len(m[k]), mm, ss, "beats static" if good else "no better than static"))
    usable = beats >= max(1, len(m) // 2)
    print("GHOST verdict:", "WORTH DRAWING" if usable
          else "NOT WORTH DRAWING (predictions carry little content beyond 'nothing moves')")


if __name__ == "__main__":
    main()
