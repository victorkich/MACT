"""Record a trained checkpoint in SMAC and capture TRUE StarCraft II engine frames.

The earlier renderer drew a synthetic top-down schematic. This one takes the pixels the game
itself produces: SC2 is launched with the render interface enabled and each frame is read back
from `observation.render_data.map`, so what you see is the same 3D engine the paper's figures show.

Two things make that work:

* SC2 prepends its own `Libs/` to the loader path, and the single file in there is a 2018
  `libstdc++.so.6` (max GLIBCXX_3.4.21). Mesa's `swrast_dri.so` needs 3.4.29, so the driver fails
  to load, EGL never initialises, and the game quietly falls back to a stub renderer -- reporting
  only "Rendered interface was requested but not available". Preloading the system libstdc++ fixes
  it. This module re-execs itself with the right environment, so callers need not care.

* SMAC hardcodes `want_rgb=False` and an interface with no render section, so `_launch` is
  patched below to ask for one.

Frames are captured on sub-steps of the environment's `step_mul`, which multiplies the frame rate
without changing a single action the policy takes: the same total game loops elapse either way.
"""
import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Re-exec with a loader environment that lets Mesa's software rasteriser load.
# Without this SC2 silently launches with no graphics and render_data comes back empty.
# ---------------------------------------------------------------------------
_SYS_LIBSTDCPP = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

if os.environ.get("_SC2_RENDER_READY") != "1":
    env = dict(os.environ)
    env["_SC2_RENDER_READY"] = "1"
    preload = env.get("LD_PRELOAD", "")
    if _SYS_LIBSTDCPP not in preload:
        env["LD_PRELOAD"] = (preload + ":" if preload else "") + _SYS_LIBSTDCPP
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env.setdefault("SC2PATH", "/root/StarCraftII")
    # argv[0] may be relative and something downstream chdirs, so resolve it now.
    argv = list(sys.argv)
    if argv and os.path.exists(argv[0]):
        argv[0] = os.path.abspath(argv[0])
    os.execve(sys.executable, [sys.executable] + argv, env)

import numpy as np
import torch
from absl import flags

flags.FLAGS([sys.argv[0]])

from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_rollouts import build_controller  # noqa: E402  (needs the re-exec first)


# ---------------------------------------------------------------------------
# Render-enabled launch
# ---------------------------------------------------------------------------
# SMAC calls full_restart() on any protocol hiccup, which relaunches SC2 and builds a NEW
# RemoteController -- orphaning a grabber wrapped around the old one. It fails silently: the
# episode finishes normally, just with almost no frames. The active grabber re-attaches here.
_ACTIVE_GRABBER = []

def patch_smac_launch(width, height, camera_width=0.0):
    """Make StarCraft2Env._launch request the render interface.

    Mirrors SMAC's own _launch, adding only want_rgb and the render section; everything
    after join_game is left to the original so map geometry is set up identically.
    """
    from smac.env.starcraft2 import starcraft2 as sc2mod

    original = sc2mod.StarCraft2Env._launch

    def _launch(self):
        from pysc2 import maps, run_configs

        self._run_config = run_configs.get(version=self.game_version)
        _map = maps.get(self.map_name)

        interface_options = sc_pb.InterfaceOptions(raw=True, score=False)
        interface_options.render.resolution.x = width
        interface_options.render.resolution.y = height
        # A minimap is required alongside the main view or the game rejects the request.
        interface_options.render.minimap_resolution.x = 128
        interface_options.render.minimap_resolution.y = 128
        if camera_width:
            # World units across the viewport: smaller = tighter on the fight.
            interface_options.render.width = camera_width

        self._sc2_proc = self._run_config.start(
            window_size=self.window_size, want_rgb=True
        )
        self._controller = self._sc2_proc.controller

        create = sc_pb.RequestCreateGame(
            local_map=sc_pb.LocalMap(
                map_path=_map.path,
                map_data=self._run_config.map_data(_map.path),
            ),
            realtime=False,
            random_seed=self._seed,
        )
        create.player_setup.add(type=sc_pb.Participant)
        create.player_setup.add(
            type=sc_pb.Computer,
            race=sc2mod.races[self._bot_race],
            difficulty=sc2mod.difficulties[self.difficulty],
        )
        self._controller.create_game(create)

        join = sc_pb.RequestJoinGame(
            race=sc2mod.races[self._agent_race], options=interface_options
        )
        self._controller.join_game(join)

        game_info = self._controller.game_info()
        if not game_info.options.HasField("render"):
            raise RuntimeError(
                "SC2 launched without the render interface. The GL framework failed to "
                "initialise -- check that LD_PRELOAD points at a libstdc++ new enough for "
                "Mesa (GLIBCXX_3.4.29+)."
            )

        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y

        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(
                self.map_x, int(self.map_y / 8)
            )
            self.pathing_grid = np.transpose(
                np.array([[(b >> i) & 1 for b in row for i in range(7, -1, -1)]
                          for row in vals], dtype=bool)
            )
        else:
            self.pathing_grid = np.invert(
                np.flip(
                    np.transpose(
                        np.array(list(map_info.pathing_grid.data), dtype=np.bool_).reshape(
                            self.map_x, self.map_y
                        )
                    ),
                    axis=1,
                )
            )

        self.terrain_height = (
            np.flip(
                np.transpose(
                    np.array(list(map_info.terrain_height.data)).reshape(
                        self.map_x, self.map_y
                    )
                ),
                1,
            )
            / 255
        )

        for g in _ACTIVE_GRABBER:
            g.attach(self._controller)

    sc2mod.StarCraft2Env._launch = _launch
    return original


class FrameGrabber:
    """Captures engine frames on sub-steps, keeping the camera on the fight."""

    def __init__(self, sc2, substeps=4, zoom_pad=1.0, pan_x=0.0, render_timeout=4.0):
        self.sc2 = sc2
        self.ctrl = sc2._controller
        self.substeps = substeps
        self.zoom_pad = zoom_pad
        self.pan_x = pan_x
        self.render_timeout = render_timeout
        self.frames = []
        self.misses = 0
        self.restarts = 0
        self._orig_step = None
        self.attach(sc2._controller)
        _ACTIVE_GRABBER.append(self)

    def attach(self, ctrl):
        """Wrap a (possibly brand-new) controller's step method."""
        if getattr(ctrl, "step", None) is self._stepping:
            return  # already ours; re-wrapping would recurse into itself
        if self._orig_step is not None and ctrl is not self.ctrl:
            self.restarts += 1
        self.ctrl = ctrl
        self._orig_step = ctrl.step
        ctrl.step = self._stepping

    # -- camera ------------------------------------------------------------
    def _live_centroid(self):
        xs, ys = [], []
        for i in range(self.sc2.n_agents):
            u = self.sc2.get_unit_by_id(i)
            if u.health > 0:
                xs.append(u.pos.x); ys.append(u.pos.y)
        for _, e in self.sc2.enemies.items():
            if e.health > 0:
                xs.append(e.pos.x); ys.append(e.pos.y)
        if not xs:
            return None
        return float(np.mean(xs)), float(np.mean(ys))

    def _move_camera(self):
        c = self._live_centroid()
        if c is None:
            return
        # The camera looks down-forward, so its target sits "below" the action on screen;
        # biasing south keeps the units centred in frame rather than riding the top edge.
        # SC2 CLAMPS the camera so the viewport cannot leave the playable area, so an x
        # correction is simply discarded once the army reaches a map edge -- measured: pan_x=-4.5
        # moved the units 8 px, not the 157 px it asks for. On 3s_vs_3z the Stalkers kite
        # backwards into the edge, so they start centred (x=362) and end pinned left (x=127).
        # The lever that works is camera_width: a narrower viewport lets the clamped centre sit
        # closer to the edge, and it enlarges the units. Keep x on the centroid and correct only y.
        point = raw_pb2.ActionRawCameraMove(
            center_world_space=dict(x=c[0] + self.pan_x, y=c[1] + self.zoom_pad, z=0)
        )
        # controller.act() takes a single Action and wraps it in the RequestAction itself.
        # Passing a RequestAction raises TypeError -- which an over-broad `except` will hide,
        # leaving the camera parked at its spawn default for the whole episode.
        self.ctrl.act(sc_pb.Action(action_raw=raw_pb2.ActionRaw(camera_move=point)))

    # -- capture -----------------------------------------------------------
    def capture(self):
        """Grab one engine frame, waiting for the renderer if it is behind.

        Under software rasterization on a loaded box the renderer frequently has no frame
        ready when the observation returns, and `render_data` comes back empty with no error
        -- which silently drops ~95% of frames. Poll until it appears.
        """
        deadline = time.time() + self.render_timeout
        while True:
            obs = self.ctrl.observe()
            r = obs.observation.render_data.map
            if r.data:
                a = np.frombuffer(r.data, dtype=np.uint8).reshape(r.size.y, r.size.x, 3)
                self.frames.append(a.copy())
                return True
            if time.time() >= deadline:
                self.misses += 1
                return False
            time.sleep(0.02)

    def _stepping(self, count=1):
        """Split one env step into `substeps` engine steps, grabbing a frame after each.

        Total game loops are unchanged, so the policy sees exactly the trajectory it would
        have seen otherwise -- only the sampling of pixels in between is finer.
        """
        n = max(1, count // self.substeps) if count > 1 else 1
        remaining = count
        while remaining > 0:
            chunk = min(n, remaining)
            self._orig_step(chunk)
            remaining -= chunk
            # Observe FIRST: an action request between step() and observe() leaves
            # render_data empty (measured -- it dropped capture from 12/12 to ~1/12).
            # The camera move is queued afterwards and takes effect next chunk.
            self.capture()
            self._move_camera()

    def detach(self):
        self.ctrl.step = self._orig_step


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_episode(env, controller, cfg, grabber):
    from collections import defaultdict

    grabber.frames = []
    state = {i: torch.tensor(o).float() for i, o in env.reset().items()}
    controller.init_rnns()
    controller.init_buffer()
    grabber._move_camera()
    grabber.capture()

    done = defaultdict(lambda: False)
    steps, info = 0, {}
    while True:
        steps += 1
        avail, obs_list = [], []
        for h in range(env.n_agents):
            avail.append(torch.tensor(env.get_avail_agent_actions(h)))
            obs_list.append(torch.zeros(1, cfg.IN_DIM) if done[h] == 1
                            else state[h].unsqueeze(0))
        observations = torch.cat(obs_list).unsqueeze(0)
        av_action = torch.stack(avail).unsqueeze(0)

        actions, _ = controller.step(observations, av_action, None)
        acts = [int(a.argmax()) for a in actions]

        next_state, reward, dones, info = env.step(acts)
        state = {i: torch.tensor(o).float() for i, o in next_state.items()}
        done = {i: float(d) for i, d in dones.items()}
        if all(done[k] == 1 for k in range(env.n_agents)):
            break

    # Hold on the final frame so the outcome is readable rather than flashing past.
    if grabber.frames:
        grabber.frames.extend([grabber.frames[-1]] * 8)
    return dict(frames=list(grabber.frames), steps=steps,
                won=bool(info.get("battle_won", False)))


def encode(frames, out_path, fps=18, crf=20):
    """Write H.264 that plays inline everywhere: yuv420p, even dimensions, +faststart."""
    import imageio_ffmpeg
    if not frames:
        raise RuntimeError("no frames captured")
    h, w = frames[0].shape[:2]
    w -= w % 2
    h -= h % 2
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
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--npz", default="", help="also dump raw frames here for compositing")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=432)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--pan-x", type=float, default=0.0,
                    help="world-unit x correction applied to the camera target")
    ap.add_argument("--pan-y", type=float, default=1.0,
                    help="world-unit y correction applied to the camera target")
    ap.add_argument("--camera-width", type=float, default=15.0,
                    help="world units across the viewport; smaller zooms in")
    ap.add_argument("--render-timeout", type=float, default=4.0,
                    help="seconds to wait for the renderer to produce each frame")
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--pick", default="first_win",
                    choices=["first_win", "first_loss", "longest", "first"])
    ap.add_argument("--max-tries", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    patch_smac_launch(args.width, args.height, camera_width=args.camera_width)
    env, controller, cfg = build_controller(args.repo, args.map, args.ckpt, args.seed)
    # SMAC launches the game lazily on the first reset, so the controller the grabber
    # needs to wrap does not exist until after it.
    env.reset()
    grabber = FrameGrabber(env.env, substeps=args.substeps, zoom_pad=args.pan_y,
                           pan_x=args.pan_x, render_timeout=args.render_timeout)

    best = None
    last = None
    for attempt in range(1, args.max_tries + 1):
        ep = run_episode(env, controller, cfg, grabber)
        last = ep
        print("  ep%d: steps=%d won=%s frames=%d (missed %d, sc2 restarts %d)"
              % (attempt, ep["steps"], ep["won"], len(ep["frames"]),
                 grabber.misses, grabber.restarts), flush=True)
        grabber.misses = 0
        grabber.restarts = 0
        if args.pick == "first" \
           or (args.pick == "first_win" and ep["won"]) \
           or (args.pick == "first_loss" and not ep["won"]):
            best = ep
            break
        if args.pick == "longest" and (best is None or ep["steps"] > best["steps"]):
            best = ep
    if best is None:
        if last is None:
            raise SystemExit("no episode completed at all")
        print("  no episode matched --pick %s in %d tries; keeping the last one "
              "(won=%s) rather than discarding the render"
              % (args.pick, args.max_tries, last["won"]), flush=True)
        best = last

    encode(best["frames"], args.out, fps=args.fps)
    print("wrote %s  (%d frames, won=%s)" % (args.out, len(best["frames"]), best["won"]))
    if args.npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.npz)), exist_ok=True)
        np.savez_compressed(args.npz, frames=np.stack(best["frames"]),
                            won=best["won"], steps=best["steps"])
        print("wrote %s" % args.npz)
    env.close()
    os._exit(0)  # SC2 cleanup can hang; results are already on disk


if __name__ == "__main__":
    main()
