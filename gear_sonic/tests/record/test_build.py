"""Offline grid builder. No hardware and no lerobot."""

import math

import pytest

from gear_sonic.data.record.build import build_rows
from gear_sonic.data.record.schema import (
    ACTION_MISSING,
    BODY_DIM,
    CAMERA_STALE,
    CAMERA_STALE_NS,
    FPS,
    HAND_DIM,
    PICO_INTERP,
    PICO_INTERP_MAX_NS,
    PICO_NAMES,
    PICO_STALE,
    STATE_INTERP,
    STATE_INTERP_MAX_NS,
    TOKEN_DIM,
    ImageRef,
    StateSample,
    TrackingSample,
    action_max_age_ns,
    load_layout,
    period_ns,
)

LAYOUT = load_layout()
NAMES = list(LAYOUT["joint_names"])
_SQRT2_OVER_2 = math.sqrt(2.0) / 2.0
# 90 degree roll about +X, wxyz. Its inverse takes world (0, 0, -1) to (0, -1, 0).
ROLL_90 = (_SQRT2_OVER_2, _SQRT2_OVER_2, 0.0, 0.0)


def _joint(name: str) -> int:
    return NAMES.index(name)


def _pico(name: str) -> int:
    return PICO_NAMES.index(name)


def _body(length: int, **at_index: float) -> tuple[float, ...]:
    values = [0.0] * length
    for index, value in at_index.items():
        values[int(index)] = value
    return tuple(values)


def make_state(
    stamp_ns: int,
    *,
    index: int = 0,
    body_q=None,
    body_dq=None,
    left_hand_q=None,
    left_hand_dq=None,
    right_hand_q=None,
    right_hand_dq=None,
    last_action=None,
    last_left_hand_action=None,
    last_right_hand_action=None,
    token=(1.0,) * TOKEN_DIM,
    base_quat=(1.0, 0.0, 0.0, 0.0),
    base_ang_vel=(0.0, 0.0, 0.0),
) -> StateSample:
    return StateSample(
        stamp_ns=stamp_ns,
        index=index,
        body_q=(0.0,) * BODY_DIM if body_q is None else body_q,
        body_dq=(0.0,) * BODY_DIM if body_dq is None else body_dq,
        left_hand_q=(0.0,) * HAND_DIM if left_hand_q is None else left_hand_q,
        left_hand_dq=(0.0,) * HAND_DIM if left_hand_dq is None else left_hand_dq,
        right_hand_q=(0.0,) * HAND_DIM if right_hand_q is None else right_hand_q,
        right_hand_dq=(0.0,) * HAND_DIM if right_hand_dq is None else right_hand_dq,
        last_action=(0.0,) * BODY_DIM if last_action is None else last_action,
        last_left_hand_action=(
            (0.0,) * HAND_DIM if last_left_hand_action is None else last_left_hand_action
        ),
        last_right_hand_action=(
            (0.0,) * HAND_DIM if last_right_hand_action is None else last_right_hand_action
        ),
        token=token,
        base_quat=base_quat,
        base_ang_vel=base_ang_vel,
    )


def make_tracking(
    stamp_ns: int, *, sequence: int = 0, values=None, valid: bool = True
) -> TrackingSample:
    vector = [0.0] * len(PICO_NAMES)
    for index, name in enumerate(PICO_NAMES):
        if name.endswith("_qw"):
            vector[index] = 1.0
    if values:
        for name, value in values.items():
            vector[_pico(name)] = value
    return TrackingSample(
        stamp_ns=stamp_ns,
        sequence=sequence,
        body_update_sequence=0,
        vector=vector,
        valid=valid,
    )


def make_image(stamp_ns: int, *, sequence: int = 0, valid: bool = True) -> ImageRef:
    return ImageRef(stamp_ns=stamp_ns, sequence=sequence, valid=valid)


def build(state, tracking, images, t0_ns: int, fps: int = FPS):
    return build_rows(state, tracking, images, fps=fps, t0_ns=t0_ns, layout=LAYOUT)


def test_empty_streams_emit_nothing():
    assert build([], [], [], 0) == []
    assert build([], [make_tracking(1_000_000_000)], [], 0) == []


def test_identity_quaternion_and_hand_actuator_permutation():
    left_q = (11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0)
    right_q = (21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0)
    body_q = tuple(100.0 + index for index in range(BODY_DIM))
    body_dq = tuple(float(index) for index in range(BODY_DIM))
    left_dq = (31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0)
    right_dq = (41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0)
    action = tuple(200.0 + index for index in range(BODY_DIM))
    left_action = (51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0)
    right_action = (61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0)
    token = tuple(float(index + 1) for index in range(TOKEN_DIM))
    sample = make_state(
        0,
        index=42,
        body_q=body_q,
        body_dq=body_dq,
        left_hand_q=left_q,
        left_hand_dq=left_dq,
        right_hand_q=right_q,
        right_hand_dq=right_dq,
        last_action=action,
        last_left_hand_action=left_action,
        last_right_hand_action=right_action,
        token=token,
        base_quat=(1.0, 0.0, 0.0, 0.0),
    )
    rows = build([sample], [make_tracking(0, sequence=8)], [make_image(0)], 0)
    assert len(rows) == 1
    row = rows[0]

    assert row.valid == 1
    assert row.flags == 0
    assert row.projected_gravity == (0.0, 0.0, -1.0)
    assert row.base_quat == (1.0, 0.0, 0.0, 0.0)
    assert row.token == token
    assert row.action == action
    assert row.left_hand_action == left_action
    assert row.right_hand_action == right_action
    assert row.joint_velocity == body_dq
    assert row.left_hand_dq == left_dq
    assert row.right_hand_dq == right_dq
    assert len(row.joint_velocity) == BODY_DIM
    assert len(row.state) == LAYOUT["state_dim"] == 43
    assert row.g1_index == 42
    assert row.pico_sequence == 8
    assert row.image_index == 0
    assert row.ages == (0.0, 0.0, 0.0)
    assert row.t_ns == 0

    thumb = _joint("left_hand_thumb_0_joint")
    index = _joint("left_hand_index_0_joint")
    assert row.state[thumb] == left_q[0]
    assert row.state[index] == left_q[3]
    assert row.state[index] != left_q[0]
    assert row.state[_joint("right_hand_thumb_0_joint")] == right_q[0]

    body_names = list(LAYOUT["body_joint_names"])
    shoulder_body = body_names.index("right_shoulder_pitch_joint")
    shoulder_state = _joint("right_shoulder_pitch_joint")
    assert shoulder_state != shoulder_body
    assert row.state[shoulder_state] == body_q[shoulder_body]
    assert row.state[shoulder_body] != body_q[shoulder_body]


def test_projected_gravity_is_inverse_of_base_rotation():
    sample = make_state(0, base_quat=ROLL_90, token=(2.0,) * TOKEN_DIM)
    row = build([sample], [make_tracking(0)], [make_image(0)], 0)[0]
    assert row.valid == 1
    assert row.projected_gravity == pytest.approx((0.0, -1.0, 0.0))


def test_state_interp_at_60ms_does_not_blend_token():
    gap = STATE_INTERP_MAX_NS
    t_k = gap // 2
    previous = make_state(
        0,
        index=4,
        body_q=_body(BODY_DIM, **{"0": 0.0}),
        body_dq=_body(BODY_DIM, **{"0": 0.0}),
        left_hand_q=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        left_hand_dq=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        last_action=_body(BODY_DIM, **{"0": 3.0}),
        last_left_hand_action=(3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        token=(1.0,) * TOKEN_DIM,
        base_quat=(1.0, 0.0, 0.0, 0.0),
        base_ang_vel=(0.0, 0.0, 0.0),
    )
    following = make_state(
        gap,
        index=9,
        body_q=_body(BODY_DIM, **{"0": 10.0}),
        body_dq=_body(BODY_DIM, **{"0": 10.0}),
        left_hand_q=(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        left_hand_dq=(10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        last_action=_body(BODY_DIM, **{"0": 7.0}),
        last_left_hand_action=(7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        token=(9.0,) * TOKEN_DIM,
        base_quat=ROLL_90,
        base_ang_vel=(2.0, 0.0, 0.0),
    )
    # Deliver the later sample first to show the builder orders by stamp.
    rows = build(
        [following, previous],
        [make_tracking(t_k, sequence=3)],
        [make_image(t_k)],
        t_k,
    )
    row = rows[0]
    assert row.t_ns == t_k
    assert t_k - previous.stamp_ns == action_max_age_ns(FPS)
    assert row.flags & STATE_INTERP
    assert row.flags & ACTION_MISSING == 0
    assert row.valid == 1
    assert row.token == previous.token
    assert row.token != following.token
    hip = _joint("left_hip_pitch_joint")
    thumb = _joint("left_hand_thumb_0_joint")
    assert row.state[hip] == 5.0
    assert row.state[thumb] == 5.0
    assert row.state[_joint("left_hand_index_0_joint")] == 0.0
    assert row.joint_velocity[0] == 5.0
    assert len(row.joint_velocity) == BODY_DIM
    assert row.left_hand_dq[0] == 5.0
    assert row.action == previous.last_action
    assert row.left_hand_action == previous.last_left_hand_action
    assert row.action[0] != 5.0
    assert row.g1_index == previous.index
    half = math.pi / 8.0
    assert row.base_quat == pytest.approx((math.cos(half), math.sin(half), 0.0, 0.0))
    assert row.base_ang_vel == pytest.approx((1.0, 0.0, 0.0))
    assert row.projected_gravity == pytest.approx((0.0, -_SQRT2_OVER_2, -_SQRT2_OVER_2))


def test_state_gap_past_60ms_is_not_interpolated():
    gap = STATE_INTERP_MAX_NS + 1
    t_k = 30_000_000
    previous = make_state(
        0,
        index=4,
        body_q=_body(BODY_DIM, **{"0": 4.0}),
        body_dq=_body(BODY_DIM, **{"0": 4.0}),
        left_hand_q=(4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        token=(1.0,) * TOKEN_DIM,
        last_action=_body(BODY_DIM, **{"0": 4.0}),
    )
    following = make_state(
        gap,
        index=9,
        body_q=_body(BODY_DIM, **{"0": 10.0}),
        left_hand_q=(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        token=(9.0,) * TOKEN_DIM,
        last_action=_body(BODY_DIM, **{"0": 10.0}),
    )
    row = build([previous, following], [make_tracking(t_k)], [make_image(t_k)], t_k)[0]
    hip = _joint("left_hip_pitch_joint")
    thumb = _joint("left_hand_thumb_0_joint")
    assert row.flags & STATE_INTERP == 0
    assert row.state[hip] == previous.body_q[0]
    assert row.state[thumb] == previous.left_hand_q[0]
    assert row.joint_velocity == previous.body_dq
    assert row.state[hip] != pytest.approx(7.0)
    assert row.valid == 0
    assert row.token == previous.token
    assert row.action == previous.last_action


def test_action_one_nanosecond_past_the_age_limit_is_not_a_token_blend():
    limit = action_max_age_ns(FPS)
    t0 = 5_000_000_000
    fresh = make_state(
        t0 - limit,
        index=2,
        body_q=_body(BODY_DIM, **{"0": 1.0}),
        last_action=_body(BODY_DIM, **{"0": 6.0}),
        last_left_hand_action=(6.0,) * HAND_DIM,
        token=(4.0,) * TOKEN_DIM,
    )
    later = make_state(
        t0 + 5_000_000,
        index=3,
        body_q=_body(BODY_DIM, **{"0": 99.0}),
        last_action=_body(BODY_DIM, **{"0": 8.0}),
        token=(8.0,) * TOKEN_DIM,
    )
    held = build([fresh, later], [make_tracking(t0)], [make_image(t0)], t0)[0]
    assert held.flags & ACTION_MISSING == 0
    assert held.token == fresh.token
    assert held.action == fresh.last_action
    assert held.flags & STATE_INTERP
    alpha = limit / (limit + 5_000_000)
    assert held.state[_joint("left_hip_pitch_joint")] == pytest.approx(1.0 + 98.0 * alpha)
    assert held.valid == 1

    stale = make_state(
        t0 - limit - 1,
        index=2,
        body_q=_body(BODY_DIM, **{"0": 1.0}),
        left_hand_q=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        last_action=_body(BODY_DIM, **{"0": 6.0}),
        last_left_hand_action=(6.0,) * HAND_DIM,
        last_right_hand_action=(6.5,) * HAND_DIM,
        token=(4.0,) * TOKEN_DIM,
    )
    nxt = make_state(
        stale.stamp_ns + 40_000_000,
        index=3,
        body_q=_body(BODY_DIM, **{"0": 11.0}),
        left_hand_q=(11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        last_action=_body(BODY_DIM, **{"0": 8.0}),
        last_left_hand_action=(8.0,) * HAND_DIM,
        token=(8.0,) * TOKEN_DIM,
    )
    row = build([stale, nxt], [make_tracking(t0)], [make_image(t0)], t0)[0]
    assert row.t_ns == t0
    assert row.flags & ACTION_MISSING
    assert row.token == (0.0,) * TOKEN_DIM
    assert row.token != stale.token
    assert row.token != nxt.token
    assert row.action == stale.last_action
    assert row.left_hand_action == stale.last_left_hand_action
    assert row.right_hand_action == stale.last_right_hand_action
    assert row.action[0] != 7.0
    assert row.valid == 0
    # The next sample is inside the state window, so joints may move, but the token does not.
    assert stale.stamp_ns + STATE_INTERP_MAX_NS >= nxt.stamp_ns
    assert row.flags & STATE_INTERP
    assert row.state[_joint("left_hand_thumb_0_joint")] != stale.left_hand_q[0]
    assert row.state[_joint("left_hand_thumb_0_joint")] != nxt.left_hand_q[0]


def test_camera_is_causal_and_stale_after_100ms():
    t0 = 1_000_000_000
    state = [make_state(t0, token=(3.0,) * TOKEN_DIM)]
    tracking = [make_tracking(t0)]
    kept = build(
        state,
        tracking,
        [make_image(t0 - CAMERA_STALE_NS, sequence=1), make_image(t0 + 1, sequence=2)],
        t0,
    )[0]
    assert kept.image_index == 0
    assert kept.flags & CAMERA_STALE == 0
    assert kept.valid == 1
    assert kept.ages[2] == pytest.approx(CAMERA_STALE_NS / 1_000_000_000)

    stale = build(
        state,
        tracking,
        [make_image(t0 - CAMERA_STALE_NS - 1, sequence=4)],
        t0,
    )[0]
    assert stale.image_index == 0
    assert stale.flags & CAMERA_STALE
    assert stale.valid == 0

    future_only = build(state, tracking, [make_image(t0 + 1_000, sequence=5)], t0)[0]
    assert future_only.image_index == -1
    assert future_only.flags & CAMERA_STALE
    assert future_only.valid == 0


def test_invalid_image_is_not_chosen():
    t0 = 2_000_000_000
    images = [
        make_image(t0, sequence=1, valid=False),
        make_image(t0 + 50_000_000, sequence=2, valid=True),
        make_image(t0 - 1_000_000, sequence=3, valid=True),
    ]
    row = build([make_state(t0)], [make_tracking(t0)], images, t0)[0]
    assert row.image_index == 2
    assert row.flags & CAMERA_STALE == 0
    assert row.valid == 1

    rejected = build([make_state(t0)], [make_tracking(t0)], [make_image(t0, valid=False)], t0)[0]
    assert rejected.image_index == -1
    assert rejected.flags & CAMERA_STALE
    assert rejected.valid == 0


def test_pico_interp_changes_position_and_keeps_valid():
    t0 = 0
    previous = make_tracking(
        -10_000_000,
        sequence=4,
        values={
            "head_x": 0.0,
            "head_qx": 0.0,
            "head_qy": 0.0,
            "head_qz": 0.0,
            "head_qw": 1.0,
            "left_controller_stick_x": 0.2,
            "left_controller_trigger": 1.0,
            "left_controller_tracked": 1.0,
            "body_pos_0_y": 10.0,
        },
    )
    following = make_tracking(
        10_000_000,
        sequence=11,
        values={
            "head_x": 4.0,
            "head_qx": _SQRT2_OVER_2,
            "head_qy": 0.0,
            "head_qz": 0.0,
            "head_qw": _SQRT2_OVER_2,
            "left_controller_stick_x": 0.8,
            "left_controller_trigger": 0.0,
            "left_controller_tracked": 0.0,
            "body_pos_0_y": 30.0,
        },
    )
    rows = build(
        [make_state(t0, token=(5.0,) * TOKEN_DIM)],
        [previous, following],
        [make_image(t0)],
        t0,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row.flags == PICO_INTERP
    assert row.valid == 1
    assert row.pico[_pico("head_x")] == pytest.approx(2.0)
    assert row.pico[_pico("body_pos_0_y")] == pytest.approx(20.0)
    assert row.pico[_pico("left_controller_stick_x")] == pytest.approx(0.2)
    assert row.pico[_pico("left_controller_trigger")] == pytest.approx(1.0)
    assert row.pico[_pico("left_controller_tracked")] == pytest.approx(1.0)
    assert row.pico_sequence == previous.sequence
    half = math.pi / 8.0
    assert row.pico[_pico("head_qx")] == pytest.approx(math.sin(half))
    assert row.pico[_pico("head_qw")] == pytest.approx(math.cos(half))
    assert row.ages[1] == pytest.approx(0.01)


def test_pico_gap_above_150ms_stays_valid():
    t0 = 0
    held = make_tracking(
        t0 - PICO_INTERP_MAX_NS - 1,
        sequence=6,
        values={"head_x": 3.5, "left_controller_trigger": 1.0},
    )
    row = build(
        [make_state(t0, token=(5.0,) * TOKEN_DIM)],
        [held],
        [make_image(t0)],
        t0,
    )[0]
    assert row.flags == PICO_STALE
    assert row.valid == 1
    assert row.pico[_pico("head_x")] == pytest.approx(3.5)
    assert row.pico[_pico("left_controller_trigger")] == pytest.approx(1.0)
    assert row.flags & PICO_INTERP == 0
    assert row.flags & ACTION_MISSING == 0
    assert row.flags & CAMERA_STALE == 0


def test_pico_wide_bracket_is_stale_without_clearing_valid():
    t0 = 1_000_000
    previous = make_tracking(t0 - 1_000_000, sequence=1, values={"head_x": 1.0})
    following = make_tracking(
        previous.stamp_ns + PICO_INTERP_MAX_NS + 1, sequence=2, values={"head_x": 9.0}
    )
    row = build(
        [make_state(t0, token=(5.0,) * TOKEN_DIM)],
        [previous, following],
        [make_image(t0)],
        t0,
    )[0]
    assert row.flags & PICO_STALE
    assert row.flags & PICO_INTERP == 0
    assert row.pico[_pico("head_x")] == pytest.approx(1.0)
    assert row.valid == 1


def test_nan_tracking_vector_does_not_raise():
    nan = float("nan")
    missing = {
        "head_tracked": 0.0,
        "head_x": nan,
        "head_y": nan,
        "head_z": nan,
        "head_qx": nan,
        "head_qy": nan,
        "head_qz": nan,
        "head_qw": nan,
    }
    previous = make_tracking(-5_000_000, sequence=1, values=missing)
    following = make_tracking(5_000_000, sequence=2, values={**missing, "body_pos_1_x": 2.0})
    row = build(
        [make_state(0, token=(5.0,) * TOKEN_DIM)],
        [previous, following],
        [make_image(0)],
        0,
    )[0]
    assert math.isnan(row.pico[_pico("head_x")])
    assert math.isnan(row.pico[_pico("head_qw")])
    assert row.pico[_pico("body_pos_1_x")] == pytest.approx(1.0)
    assert row.valid == 1
    assert row.flags & PICO_INTERP

    single = make_tracking(0, values=missing)
    held = build([make_state(0)], [single], [make_image(0)], 0)[0]
    assert math.isnan(held.pico[_pico("head_y")])
    assert held.valid == 1


def test_grid_includes_the_last_stamp_and_not_the_next_step():
    period = period_ns(FPS)
    on_grid = build([make_state(0), make_state(period)], [], [make_image(0)], 0)
    assert [row.t_ns for row in on_grid] == [0, period]

    short = build([make_state(0)], [], [make_image(period - 1)], 0)
    assert [row.t_ns for row in short] == [0]

    image_extends = build([make_state(0)], [], [make_image(period)], 0)
    assert [row.t_ns for row in image_extends] == [0, period]
