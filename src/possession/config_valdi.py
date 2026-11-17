"""Every tunable number in the pipeline, in one place.

Evidence is in "frames": a weight of 1 means one frame of full-strength evidence, so a switch cost
of 40 needs roughly 1.6 s of uncontested evidence for the other state before the decoder moves.
"""

import json
from dataclasses import asdict, dataclass, fields


@dataclass
class Params:
    # ball interpolation
    max_gap: int = 25                # fill missing ball across gaps up to this many frames

    # who is closest to the ball
    w_ball: float = 1.0              # weight of the nearest-player evidence on observed frames
    w_filled: float = 0.5            # multiplier when the ball position was interpolated
    control_radius: float = 2.5      # metres; evidence fades as exp(-nearest / radius)
    margin_scale: float = 2.0        # metres of distance gap for full-strength evidence

    # Smart Tagging
    w_tag: float = 0.3               # per-frame weight of the attacking-team prior
    w_sp_dead: float = 0.6           # dead-ball evidence at the start of a tagged set piece
    sp_dead_frames: int = 50         # how long into a set piece interval the ball is still dead
    w_sp_team: float = 0.6           # restarting-team evidence after the restart

    # dead ball
    w_outside: float = 2.0           # ball tracked outside the pitch
    dead_gap: int = 75               # missing-ball runs longer than this count as dead evidence
    w_missing: float = 0.3           # long missing run on its own
    slow_speed: float = 1.5          # m/s, mean player speed below which play looks stopped
    w_slow: float = 0.6              # slow players
    w_missing_slow: float = 0.8      # long missing run and slow players together
    alive_ball_speed: float = 4.0    # m/s, an observed ball moving this fast is in play
    w_alive: float = 0.5             # evidence against D from a moving ball
    d_bias: float = 0.0              # constant added to D everywhere

    # decoder
    switch_cost: float = 40.0        # H <-> A
    dead_cost: float = 30.0          # alive <-> D

    @classmethod
    def load(cls, path=None, **overrides):
        values = {}
        if path:
            with open(path) as handle:
                values.update(json.load(handle))
        values.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise KeyError("unknown params: {}".format(sorted(unknown)))
        return cls(**values)

    def save(self, path):
        with open(path, "w") as handle:
            json.dump(asdict(self), handle, indent=2)
            handle.write("\n")


@dataclass
class ParamsV2(Params):
    """v1 plus camera (homography), per-half normalised speeds and a dead-ball memory."""

    # homography
    err_scale: float = 1.0           # quality = exp(-reprojection_error / err_scale)
    pan_window: int = 25             # frames, smoothing of camera pan speed

    # normalised slowness: slow = sigmoid(-(z_player_speed - z_slow) / z_temp)
    z_slow: float = -0.3
    z_temp: float = 0.3
    w_slow_z: float = 0.4            # D evidence from normalised slowness
    team_gate: float = 0.5           # team evidence shrinks by team_gate * slow
    w_cam_still: float = 0.2         # extra D when slow players and the camera is still
    cam_still_speed: float = 0.5     # m/s of camera aim below which the camera is "still"

    # dead memory: after the ball leaves play, stay dead until a fast restart kick
    w_memory: float = 0.6
    edge_margin: float = 2.0         # m; a ball lost within this of a line counts as leaving
    restart_speed: float = 8.0       # m/s; observed ball faster than this restarts play
    restart_min_delay: int = 50      # frames after the exit before a restart can count
    memory_max: int = 1500           # frames; memory gives up after this long

    # Smart Tagging set-piece window around the tagged start (restart is near the tag start)
    sp_pre_frames: int = 50
