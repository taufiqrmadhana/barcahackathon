"""Every tunable number in the pipeline, in one place. src/params.json is the submitted config.

Scores are in "frames of evidence": a per-frame log-likelihood of about 1, so a switch cost of 20
needs roughly 20 frames' worth of support for the other state before the decoder moves.
"""

import json
from dataclasses import asdict, dataclass, fields


@dataclass
class Params:
    # step 1: cleaning
    player_margin: float = 5.0       # m outside the pitch beyond which a player detection is dropped
    max_player_speed: float = 12.0   # m/s; faster slot motion is an identity switch, not a run
    vel_k: int = 2                   # frames either side for central-difference velocities
    ball_spike_speed: float = 45.0   # m/s; a ball point needing more than this both ways is noise
    ball_island_len: int = 3         # observed ball runs shorter than this ...
    ball_island_gap: int = 50        # ... with gaps this long either side are dropped
    max_gap: int = 12                # interpolate the ball across gaps up to this many frames
    ball_source: str = ""            # "" = ATD ball; else a fused ATD+video track in cache/cv/<name>.npz

    # step 2: control evidence
    ctrl_radius: float = 1.3         # m; control probability is 0.5 at this distance
    ctrl_softness: float = 0.35      # m; how sharply it falls off
    sync_free: float = 2.5           # m/s ball-vs-player speed difference allowed under control
    sync_scale: float = 3.0          # m/s; decay beyond that
    filled_trust: float = 0.6        # multiplier on control for interpolated ball frames
    corner_zone: float = 3.0         # m from both lines
    still_speed: float = 1.2         # m/s; the ball is resting
    slow_window: int = 50            # frames over which player speed is averaged
    ball_smooth: int = 9             # frames over which ball speed is averaged for rest / kick

    # step 3: events
    touch_k: int = 3
    touch_dv: float = 3.5            # m/s velocity change that counts as a touch
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
    decel_window: int = 25           # frames either side when looking for a collective slowdown
    decel_drop: float = 1.2          # m/s drop in mean player speed that looks like a whistle
    decel_to: float = 1.6            # ... ending below this
    decel_lead: int = 20             # the whistle comes this many frames before the slowdown peaks
    rest_speed: float = 1.8          # m/s smoothed ball speed of a resting ball (tracking jitter)
    placed_min_frames: int = 25
    placed_calm_speed: float = 1.8
    restart_kick_speed: float = 5.0
    kick_within: int = 20
    reappear_min_frames: int = 75
    dead_min_gap: int = 50           # a stoppage lasts at least this long before its restart
    dead_max_gap: int = 2250         # and is paired with a restart no later than this
    hold_kinds: str = "goal_line,touchline,vanish"
    hold_goal_line: int = 200        # goal kicks and corners take a while to set up
    hold_touchline: int = 75

    # shape model
    shape_folds: int = 8
    shape_guard: int = 250
    shape_stride: int = 3
    shape_iter: int = 150
    shape_eps: float = 0.05
    pretrained_shape: str = ""      # checkpoint of src/pretrain/train.py; empty keeps the self-trained model only
    pretrained_mix: float = 1.0      # weight of the pretrained probabilities against the self-trained ones
    shape_rounds: int = 1            # refit the shape model on its own decode this many times
    shape_conf: float = 0.0          # > 0: train only on frames the decoder is this sure of

    # Optional external supervised evidence, applied only in the final decode.
    ownership_model: str = ""       # local joblib checkpoint; empty preserves the baseline
    supervised_shape_model: str = "" # replace final H/A shape ratio; preserve baseline P(D)
    w_owner: float = 1.0
    w_owner_switch: float = 12.0

    # Optional second H/A decode using Neo's independent ATD contact ownership.
    w_contact: float = 0.0
    contact_alpha: float = 0.5
    contact_window: int = 50         # abstain if no observed ball within this many frames
    contact_lock_dead: int = 1      # retain all D frames from the first final decode
    contact_switch_cost: float = -1.0 # < 0 uses switch_cost
    contact_min_frames: int = 0    # merge shorter H/A islands surrounded by the other team
    contact_local_switch: int = 0  # lower cost only toward a team supported by contact
    hybrid_model: str = ''         # optional local Valdi video/context classifier checkpoint
    hybrid_weight: float = .5
    hybrid_lock_dead: int = 1
    hybrid_prune_max_frames: int = 0  # > 0: merge short team islands unsupported by the video head
    hybrid_prune_min_confidence: float = .5
    hybrid_loose_threshold: float = 0.0  # > 0: confident video can switch into a loose-ball state
    hybrid_dead_margin: int = 0    # allow video to adjust locked D edges within this many frames
    hybrid_dead_weight: float = -1.0  # < 0: use hybrid_weight for both team and dead evidence

    # step 4: emissions
    ctrl_eps: float = 0.05
    near_radius: float = 1.3         # m; proximity-only ownership is 0.5 at this distance
    near_margin: float = 0.5         # m; the other team's nearest must be this much farther for full weight
    w_near: float = 0.0              # control = max(ctrl, w_near * nearest-team proximity); 0 keeps ctrl only
    near_rounds: int = 0             # 1: also in the decodes that make shape pseudo-labels; 0: final decode only
    w_ball: float = 1.0
    w_shape: float = 1.0
    w_shape_known: float = 0.0
    w_tag: float = 0.0
    w_dead: float = 1.0
    d_bias: float = 0.0
    w_out: float = 6.0
    w_out_carry: float = 3.0
    slow_speed: float = 1.3          # m/s mean player speed (2 s average) at which D and alive tie
    slow_scale: float = 0.3
    w_slow: float = 1.0
    dead_gap: int = 75
    w_missing: float = 0.0
    w_missing_slow: float = 0.5
    w_still: float = 2.0
    w_shape_dead: float = 0.5
    alive_ball_speed: float = 6.0
    w_alive: float = 1.0
    w_hold: float = 5.0              # inside a stoppage-to-restart interval
    w_post_restart: float = 2.0      # just after a restart
    inner_restarts: int = 1          # 0: a "restart" inside a stoppage-to-restart interval, before the
                                     # restart it is paired with, is the ball being set up: no alive evidence
    post_restart_frames: int = 50
    w_tag_dead: float = 1.0          # around a tagged set piece start
    tag_dead_frames: int = 25
    z_max: float = 4.0
    pre_kickoff_dead: int = 1        # D on graded frames before a documented kickoff (FAQ.md)
    goal_dead: int = 1               # D from the on-screen score change to the tagged kick-off
    w_whistle: float = 0.0           # cut of the cost into D around a referee whistle (audio, cv/whistle.py)
    w_whistle_dead: float = 0.0      # D evidence for whistle_hold frames after a whistle
    whistle_thresh: float = 26.0     # dB of the 3.8 kHz tone over the neighbouring bands
    whistle_before: int = 5
    whistle_after: int = 20
    whistle_hold: int = 50
    w_whistle_confirmed: float = 0.0 # D evidence from a whistle until the players' slowdown that confirms it
    whistle_confirm_speed: float = 1.5   # m/s smoothed player speed that confirms a stoppage ...
    whistle_confirm_window: int = 125    # ... within this many frames after the whistle
    whistle_lag: int = 5                 # D starts this many frames after the blast begins
    whistle_min_speed: float = 0.0       # players must be at least this fast at the blast (a real
                                         # deceleration follows); 0 = any blast counts

    # step 4: transitions
    loose_cost: float = 1.0
    switch_cost: float = 20.0
    tackle_extra: float = 5.0
    dead_entry_cost: float = 20.0
    dead_exit_cost: float = 20.0
    dead_min_frames: int = 50
    w_turnover: float = 8.0
    w_touch: float = 4.0
    w_duel: float = 2.0
    w_deflection: float = 0.0        # a one-touch of the other team inside a sequence (GT counts last touch)
    touch_before: int = 2
    touch_after: int = 10
    w_near_switch: float = 0.0       # switch-cost cut on tracked-ball frames where that team is nearest
                                     # (scaled by its proximity); missing-ball frames keep the full cost
    w_stoppage: float = 8.0
    stop_before: int = 10
    stop_after: int = 25
    w_shot_dead: float = 3.0
    shot_window: int = 75
    w_restart: float = 8.0
    restart_before: int = 10
    restart_after: int = 10

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
