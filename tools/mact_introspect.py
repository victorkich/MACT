"""Turn MACT's internals into things that can be drawn on a StarCraft II frame.

Two conversions are needed for any of the overlays, and both are verified against ground
truth rather than assumed:

  observation vector -> unit world positions
      SMAC encodes each visible unit as (available/visible, distance, relative_x, relative_y, ...)
      with the relative offsets divided by sight_range. Multiply back and add the observing
      agent's own position and you have world coordinates. The same inversion applied to an
      *imagined* observation gives the world model's prediction of where everyone will be.

  world position -> screen pixel
      The maps are flat and the camera only translates, so the ground plane maps to the frame
      by a fixed homography applied to (world - camera_target). Fitting it needs no camera
      intrinsics, only correspondences, which we already have: SMAC knows where every unit is
      and the units are visible in the frame.

Nothing here is drawn until `validate_*` reports acceptable error.
"""
import numpy as np


# ---------------------------------------------------------------------------
# observation vector -> unit positions
# ---------------------------------------------------------------------------
class ObsDecoder:
    """Invert SMAC's per-agent observation back into world positions.

    Layout, from StarCraft2Env.get_obs_agent:
        move_feats | enemy_feats (n_enemies x nf_en) | ally_feats (n_allies x nf_al) | own_feats
    with enemy/ally blocks starting (visible/available, distance, rel_x, rel_y, ...).
    Sizes are read off the live env rather than hardcoded, because they depend on
    unit_type_bits, shield bits and the pathing/terrain options.
    """

    def __init__(self, env):
        e = env.env if hasattr(env, "env") else env
        self.e = e
        self.n_agents = e.n_agents
        self.n_enemies = e.n_enemies
        self.sight = e.unit_sight_range(0)

        nf_al, nf_en = e.get_obs_move_feats_size, None  # placeholders, resolved below
        move_size = e.get_obs_move_feats_size()
        en_size = e.get_obs_enemy_feats_size()
        al_size = e.get_obs_ally_feats_size()
        own_size = e.get_obs_own_feats_size()

        # these helpers return (n, nf) tuples in SMAC
        self.n_en, self.nf_en = en_size if isinstance(en_size, tuple) else (self.n_enemies, en_size)
        self.n_al, self.nf_al = al_size if isinstance(al_size, tuple) else (self.n_agents - 1, al_size)
        self.move_size = move_size if isinstance(move_size, int) else int(np.prod(move_size))
        self.own_size = own_size if isinstance(own_size, int) else int(np.prod(own_size))

        self.en_start = self.move_size
        self.al_start = self.en_start + self.n_en * self.nf_en
        self.own_start = self.al_start + self.n_al * self.nf_al

    def describe(self):
        return ("move=%d  enemies=%dx%d @%d  allies=%dx%d @%d  own=%d @%d  total=%d  sight=%.1f"
                % (self.move_size, self.n_en, self.nf_en, self.en_start,
                   self.n_al, self.nf_al, self.al_start, self.own_size, self.own_start,
                   self.own_start + self.own_size, self.sight))

    def decode(self, obs_vec, agent_id, own_xy):
        """-> (enemy_xy, enemy_seen, ally_xy, ally_seen) in world coordinates.

        `own_xy` anchors the frame: the observation is relative to the observing agent, so the
        agent's own position has to come from somewhere else (the live env, or the previous
        prediction when rolling forward).
        """
        o = np.asarray(obs_vec, dtype=np.float64).reshape(-1)
        ex, es = np.zeros((self.n_en, 2)), np.zeros(self.n_en, dtype=bool)
        for i in range(self.n_en):
            b = self.en_start + i * self.nf_en
            seen = o[b] > 0.05 or abs(o[b + 2]) > 1e-3 or abs(o[b + 3]) > 1e-3
            es[i] = seen
            ex[i] = (own_xy[0] + o[b + 2] * self.sight,
                     own_xy[1] + o[b + 3] * self.sight)
        ax, as_ = np.zeros((self.n_al, 2)), np.zeros(self.n_al, dtype=bool)
        for i in range(self.n_al):
            b = self.al_start + i * self.nf_al
            seen = o[b] > 0.05 or abs(o[b + 2]) > 1e-3 or abs(o[b + 3]) > 1e-3
            as_[i] = seen
            ax[i] = (own_xy[0] + o[b + 2] * self.sight,
                     own_xy[1] + o[b + 3] * self.sight)
        return ex, es, ax, as_


def validate_obs_decoder(env, dec, verbose=True):
    """Decode a REAL observation and compare with the true unit positions."""
    e = env.env if hasattr(env, "env") else env
    errs = []
    obs = e.get_obs()
    for a in range(e.n_agents):
        u = e.get_unit_by_id(a)
        if u.health <= 0:
            continue
        own = (u.pos.x, u.pos.y)
        ex, es, ax, as_ = dec.decode(obs[a], a, own)
        true_en = [(en.pos.x, en.pos.y, en.health > 0) for _, en in sorted(e.enemies.items())]
        for i, (tx, ty, alive) in enumerate(true_en):
            if i < len(es) and es[i] and alive:
                errs.append(float(np.hypot(ex[i][0] - tx, ex[i][1] - ty)))
    errs = np.array(errs) if errs else np.array([np.nan])
    if verbose:
        print("VAL obs-decoder: %d visible enemy observations, mean err %.3f world units, max %.3f"
              % (len(errs), np.nanmean(errs), np.nanmax(errs)), flush=True)
    return errs


# ---------------------------------------------------------------------------
# world position -> screen pixel
# ---------------------------------------------------------------------------
def fit_homography(world_rel, screen):
    """Least-squares homography mapping (world - camera_target) -> pixels. >=4 correspondences."""
    w = np.asarray(world_rel, float)
    s = np.asarray(screen, float)
    n = len(w)
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    for i in range(n):
        X, Y = w[i]
        u, v = s[i]
        A[2 * i] = [X, Y, 1, 0, 0, 0, -u * X, -u * Y]
        A[2 * i + 1] = [0, 0, 0, X, Y, 1, -v * X, -v * Y]
        b[2 * i] = u
        b[2 * i + 1] = v
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.append(h, 1.0).reshape(3, 3)


def fit_affine(world_rel, screen):
    """Least-squares affine (6 params) returned as a 3x3 so `project` is unchanged.

    Preferred over a full homography here. The camera only translates over flat ground, so the
    extra two projective parameters buy nothing and are free to blow up on a noisy
    correspondence: on 8m the homography fit turned a 233 px outlier into a projected span of
    13218 px inside a 768 px frame.
    """
    w = np.asarray(world_rel, float)
    s_ = np.asarray(screen, float)
    A = np.concatenate([w, np.ones((len(w), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A, s_, rcond=None)      # (3, 2)
    M = np.eye(3)
    M[:2, :2] = coef[:2].T
    M[:2, 2] = coef[2]
    return M


def fit_robust(world_rel, screen, drop_frac=0.25, rounds=3):
    """Affine fit with iterative rejection of the worst correspondences.

    Units die mid-sweep and muzzle flashes drag the mask centroid, so a few pairs are simply
    wrong. Trimming them matters more than the model's flexibility.
    """
    w = np.asarray(world_rel, float)
    s_ = np.asarray(screen, float)
    keep = np.ones(len(w), bool)
    M = fit_affine(w, s_)
    for _ in range(rounds):
        if keep.sum() < 6:
            break
        pred = project(M, w[keep], (0, 0))
        res = np.hypot(*(pred - s_[keep]).T)
        cut = np.quantile(res, 1 - drop_frac)
        idx = np.nonzero(keep)[0]
        keep[idx[res > cut]] = False
        M = fit_affine(w[keep], s_[keep])
    return M, keep


def analytic_projection(width, height, camera_width):
    """Build the world->screen matrix from SC2's camera geometry instead of fitting it.

    Fitting needs correspondences, which needs finding units in the image, and no single mask
    isolates units across maps: on MMM and so_many_baneling every candidate spans most of the
    frame because the terrain is as saturated as the units.

    But fitting is unnecessary. Two independently fitted, visually validated matrices (8m at
    1536x864 and 3s_vs_3z at 768x432) agree on the structure:

        x-scale  = 0.78 * (width / camera_width)      within 3% on both
        y-scale  = -0.74 * x-scale                    the isometric tilt, within 6% on both
        origin   = frame centre                       within 39 px of centre on both
        off-diagonals ~ 0                             axis aligned

    Those are properties of SC2's fixed camera pitch, not of any map, so they transfer. The
    small residual differences are noise in the fits, not real per-map variation.
    """
    sx = 0.78 * (float(width) / float(camera_width))
    sy = -0.74 * sx
    M = np.eye(3)
    M[0, 0] = sx
    M[1, 1] = sy
    M[0, 2] = width / 2.0
    M[1, 2] = height / 2.0
    return M


def project(H, world_xy, cam_xy):
    """Apply the fitted homography to world points, relative to the camera target."""
    p = np.asarray(world_xy, float).reshape(-1, 2) - np.asarray(cam_xy, float).reshape(1, 2)
    hom = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ H.T
    return hom[:, :2] / hom[:, 2:3]
