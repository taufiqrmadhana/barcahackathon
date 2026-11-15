import json
from dataclasses import asdict, dataclass, fields


class _Configurable:
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
class Params(_Configurable):
    player_margin: float = 5.0
    max_player_speed: float = 12.0
    vel_k: int = 2
    ball_spike_speed: float = 45.0
    ball_island_len: int = 3
    ball_island_gap: int = 50
    max_gap: int = 12

    ctrl_radius: float = 1.3
    ctrl_softness: float = 0.35
    sync_free: float = 2.5
    sync_scale: float = 3.0
    filled_trust: float = 0.6
    corner_zone: float = 3.0
    still_speed: float = 1.2
    slow_window: int = 50
    ball_smooth: int = 9

    touch_k: int = 3
    touch_dv: float = 3.5
    touch_spacing: int = 6
    touch_radius: float = 2.0
    touch_margin: float = 1.5
    duel_radius: float = 1.5
    seq_max_gap: int = 100
    pass_min_travel: float = 5.0
    shot_speed: float = 18.0
    shot_range: float = 35.0
    out_margin: float = 0.5
    out_min_frames: int = 3
    vanish_min_frames: int = 50
    vanish_horizon: float = 0.4
    decel_window: int = 25
    decel_drop: float = 1.2
    decel_to: float = 1.6
    decel_lead: int = 20
    rest_speed: float = 1.8
    placed_min_frames: int = 25
    placed_calm_speed: float = 1.8
    restart_kick_speed: float = 5.0
    kick_within: int = 20
    reappear_min_frames: int = 75
    dead_min_gap: int = 50
    dead_max_gap: int = 2250
    hold_kinds: str = "goal_line,touchline,vanish"
    hold_goal_line: int = 200
    hold_touchline: int = 75

    shape_folds: int = 8
    shape_guard: int = 250
    shape_stride: int = 3
    shape_iter: int = 150
    shape_eps: float = 0.05
    shape_rounds: int = 2
    shape_conf: float = 0.0

    ctrl_eps: float = 0.05
    w_ball: float = 1.0
    w_shape: float = 1.0
    w_shape_known: float = 0.0
    w_tag: float = 0.0
    w_dead: float = 1.0
    d_bias: float = 0.0
    w_out: float = 6.0
    w_out_carry: float = 3.0
    slow_speed: float = 1.3
    slow_scale: float = 0.3
    w_slow: float = 1.0
    dead_gap: int = 75
    w_missing: float = 0.0
    w_missing_slow: float = 0.5
    w_still: float = 2.0
    w_shape_dead: float = 0.5
    alive_ball_speed: float = 6.0
    w_alive: float = 3.0
    w_hold: float = 5.0
    w_post_restart: float = 2.0
    post_restart_frames: int = 50
    w_tag_dead: float = 1.0
    tag_dead_frames: int = 25
    z_max: float = 4.0

    loose_cost: float = 1.0
    switch_cost: float = 16.0
    tackle_extra: float = 5.0
    dead_entry_cost: float = 20.0
    dead_exit_cost: float = 30.0
    dead_min_frames: int = 50
    w_turnover: float = 8.0
    w_touch: float = 4.0
    w_duel: float = 2.0
    touch_before: int = 2
    touch_after: int = 10
    w_stoppage: float = 8.0
    stop_before: int = 10
    stop_after: int = 25
    w_shot_dead: float = 3.0
    shot_window: int = 75
    w_restart: float = 8.0
    restart_before: int = 10
    restart_after: int = 10


@dataclass
class CameraParams(_Configurable):
    max_gap: int = 25
    w_ball: float = 1.0
    w_filled: float = 0.5
    control_radius: float = 2.5
    margin_scale: float = 2.0
    w_tag: float = 0.0
    w_sp_dead: float = 0.6
    sp_dead_frames: int = 50
    w_sp_team: float = 0.6
    w_outside: float = 2.0
    dead_gap: int = 75
    w_missing: float = 0.3
    slow_speed: float = 1.5
    w_slow: float = 0.6
    w_missing_slow: float = 0.8
    alive_ball_speed: float = 4.0
    w_alive: float = 0.5
    d_bias: float = -0.1
    switch_cost: float = 20.0
    dead_cost: float = 20.0
    err_scale: float = 1.0
    pan_window: int = 25
    z_slow: float = -0.3
    z_temp: float = 0.3
    w_slow_z: float = 0.4
    team_gate: float = 0.5
    w_cam_still: float = 0.2
    cam_still_speed: float = 0.5
    w_memory: float = 0.6
    edge_margin: float = 2.0
    restart_speed: float = 8.0
    restart_min_delay: int = 50
    memory_max: int = 1500
    sp_pre_frames: int = 50
