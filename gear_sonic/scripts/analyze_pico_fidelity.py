"""Where does teleoperated motion lose fidelity: PICO tracking, SMPL mapping, or timing?

    # Record while teleoperating (POSE mode for the SMPL checks), bridge and manager running:
    python gear_sonic/scripts/analyze_pico_fidelity.py record capture.msgpack --seconds 30
    python gear_sonic/scripts/analyze_pico_fidelity.py report capture.msgpack

Captures from measure_pico_sim_latency.py have the same format and can be reported too.
Controls' tracking.jsonl observer files can also be reported. Those files do not
contain standalone robot feedback or the original device-clock fields.

The policy never sees PICO's joint positions. It sees a standard SMPL skeleton
posed with PICO's joint *rotations*. PICO also reports where it believes each
joint is, so every frame offers an independent check of that mapping:

  * Skeleton agreement: compare PICO positions with the skeleton reconstructed
    from its rotations. Differences can come from mapping, proportions, rest
    axes or inconsistencies between the source's positions and rotations.
  * Bone consistency: check segment lengths and their direction in the parent
    joint's frame, without SMPL. Variable lengths cannot be reproduced by a
    fixed-shape skeleton; this check does not establish physical accuracy.
  * Body timing: how unevenly body packets arrive, and whether the headset sends
    a timestamp that lets the manager undo it.
  * Wrist following: how closely and how late the robot's wrist joints follow
    their reference. Only the wrists are commanded as joint angles in POSE mode.
"""

import argparse
import json
from pathlib import Path
import time

import msgpack
import numpy as np

from gear_sonic.utils.teleop.input_readers import NUM_BODY_JOINTS, to_openxr

BODY_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hand", "right_hand",
]  # fmt: skip
# Hardware-order indices of the six wrist joints in g1_debug (roll, pitch, yaw per arm).
WRIST_JOINTS = {
    "left_wrist_roll": 19, "left_wrist_pitch": 20, "left_wrist_yaw": 21,
    "right_wrist_roll": 26, "right_wrist_pitch": 27, "right_wrist_yaw": 28,
}  # fmt: skip
# Bones shorter than this in either skeleton have no reliable direction.
MIN_BONE_M = 0.02
POLICY_PERIOD_S = 0.02


def _percentiles(values) -> dict | None:
    if len(values) == 0:
        return None
    p50, p95, top = np.percentile(values, [50, 95, 100])
    return {"p50": round(float(p50), 3), "p95": round(float(p95), 3), "max": round(float(top), 3)}


def body_poses(frame: dict) -> np.ndarray | None:
    """``(24, 7)`` OpenXR body poses of a tracking frame, as the manager reads them."""
    body = frame.get("body")
    if body is None:
        return None
    positions = np.asarray(body["positions"], float).reshape(NUM_BODY_JOINTS, 3)
    rotations = np.asarray(body["rotations"], float).reshape(NUM_BODY_JOINTS, 4)
    return to_openxr(positions, rotations).astype(np.float32)


def bone_vectors(body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(pico, smpl)`` bone vectors, each ``(23, 3)`` in the Z-up world frame.

    ``pico`` joins PICO's reported joint positions; ``smpl`` joins the joints of
    the SMPL skeleton the manager poses with PICO's rotations.
    """
    import torch

    from gear_sonic.scripts.pico_manager_thread_server import (
        SMPL_PARENT_INDICES,
        compute_from_body_poses,
    )

    parents = np.array(SMPL_PARENT_INDICES[1:])
    smpl = compute_from_body_poses(SMPL_PARENT_INDICES, torch.device("cpu"), body)["joints"]
    smpl = smpl[0].numpy().astype(float)
    # OpenXR is Y-up; the SMPL joints were turned Z-up by a quarter turn about X.
    pico = body[:, :3].astype(float)[:, [0, 2, 1]] * [1.0, -1.0, 1.0]
    return pico[1:] - pico[parents], smpl[1:] - smpl[parents]


def skeleton_agreement(bodies: list[np.ndarray]) -> dict:
    """Per bone: angle between PICO's and the rebuilt skeleton's bone direction."""
    angles = np.full((len(bodies), NUM_BODY_JOINTS - 1), np.nan)
    ratios = np.full_like(angles, np.nan)
    for row, body in enumerate(bodies):
        pico, smpl = bone_vectors(body)
        pico_len, smpl_len = np.linalg.norm(pico, axis=1), np.linalg.norm(smpl, axis=1)
        valid = (pico_len > MIN_BONE_M) & (smpl_len > MIN_BONE_M)
        cosine = np.einsum("ij,ij->i", pico[valid], smpl[valid]) / (pico_len[valid] * smpl_len[valid])
        angles[row, valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        ratios[row, valid] = pico_len[valid] / smpl_len[valid]
    bones = {}
    for joint in range(1, NUM_BODY_JOINTS):
        column = angles[:, joint - 1]
        column = column[np.isfinite(column)]
        if len(column):
            bones[BODY_JOINT_NAMES[joint]] = {
                "direction_error_deg": _percentiles(column),
                "length_pico_over_smpl": round(float(np.nanmedian(ratios[:, joint - 1])), 3),
            }
    return {"frames": len(bodies), "bones": bones}


def bone_consistency(bodies: list[np.ndarray]) -> dict:
    """PICO segment geometry before SMPL: length and parent-local direction.

    A rigid segment should have a fixed length and direction in its parent's
    bone frame. Some vendor joints do not follow that frame convention, so a
    changing direction alone is not proof of a tracking fault.
    """
    from scipy.spatial.transform import Rotation

    from gear_sonic.scripts.pico_manager_thread_server import SMPL_PARENT_INDICES

    bones = {}
    if not bodies:
        return {"frames": 0, "bones": bones}
    poses = np.asarray(bodies, dtype=float)
    parents = np.asarray(SMPL_PARENT_INDICES[1:])
    delta = poses[:, 1:, :3] - poses[:, parents, :3]
    local = Rotation.from_quat(poses[:, parents, 3:].reshape(-1, 4)).inv().apply(
        delta.reshape(-1, 3)
    ).reshape(len(poses), NUM_BODY_JOINTS - 1, 3)
    for index, name in enumerate(BODY_JOINT_NAMES[1:]):
        lengths = np.linalg.norm(local[:, index], axis=1)
        valid = np.isfinite(lengths) & (lengths > MIN_BONE_M)
        if not valid.any():
            continue
        lengths = lengths[valid]
        directions = local[valid, index] / lengths[:, None]
        centre = np.median(directions, axis=0)
        norm = np.linalg.norm(centre)
        direction_error = None
        if norm > np.finfo(float).eps:
            cosine = directions @ (centre / norm)
            direction_error = _percentiles(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        median_length = float(np.median(lengths))
        bones[name] = {
            "valid_frames": int(valid.sum()),
            "length_m": _percentiles(lengths),
            "length_deviation_pct": _percentiles(abs(lengths / median_length - 1) * 100),
            "parent_local_direction_deviation_deg": direction_error,
        }
    return {"frames": len(bodies), "bones": bones}


def body_timing(frames: list[dict]) -> dict:
    """Arrival regularity of fresh bodies and what the headset timestamps would undo."""
    fresh, last_sequence = [], None
    for frame in frames:
        sequence = frame.get("body_update_sequence")
        if frame.get("body") is None or (sequence is not None and sequence == last_sequence):
            continue
        last_sequence = sequence
        fresh.append(frame)
    arrival = np.array([f["sample_time_ns"] for f in fresh], dtype=np.int64)
    receipts = bool(fresh) and all(f.get("body_update_sequence") is not None for f in fresh)
    report = {"body_frames": len(fresh), "body_receipt_counters_present": receipts}
    if receipts:
        report["fresh_bodies"] = len(fresh)
    if len(fresh) < 3:
        return report
    span = (arrival[-1] - arrival[0]) * 1e-9
    rate_key = "rate_hz" if receipts else "poll_rate_hz"
    report[rate_key] = round((len(fresh) - 1) / span, 2) if span > 0 else None
    report["arrival_gap_ms"] = _percentiles(np.diff(arrival) * 1e-6)
    for index, name in enumerate(("body", "packet")):
        stamps = [(f.get("device_time_ns") or [0, 0])[index] for f in fresh]
        stamps = np.array([s if isinstance(s, int) else 0 for s in stamps], dtype=np.int64)
        usable = stamps > 0
        field = {"present": bool(usable.any())}
        if usable.sum() >= 3:
            a, s = arrival[usable], stamps[usable]
            elapsed = a[-1] - a[0]
            field["advancing"] = bool((np.diff(s) > 0).mean() > 0.9)
            field["rate_vs_pc_clock"] = round(float((s[-1] - s[0]) / elapsed), 4) if elapsed else None
            # Arrival minus headset time is a constant offset plus each packet's
            # transport delay; above its minimum is the jitter the manager removes.
            field["arrival_jitter_ms"] = _percentiles(((a - s) - (a - s).min()) * 1e-6)
        report[f"headset_{name}_timestamp"] = field
    return report


def wrist_following(feedback: list[tuple[float, dict]]) -> dict:
    """Per wrist joint: tracking error and the delay that best aligns robot and reference."""
    rows = [(t, f["body_q_target"], f["body_q_measured"]) for t, f in feedback
            if "body_q_target" in f and "body_q_measured" in f]  # fmt: skip
    if len(rows) < 100:
        return {"feedback_frames": len(rows)}
    times = np.array([r[0] for r in rows])
    target, measured = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    # Feedback is one frame per policy tick; put both on that grid for shifting.
    grid = np.arange(times[0], times[-1], POLICY_PERIOD_S)
    report = {"feedback_frames": len(rows), "joints": {}}
    for name, joint in WRIST_JOINTS.items():
        ref, got = np.interp(grid, times, target[:, joint]), np.interp(grid, times, measured[:, joint])
        if np.ptp(ref) < np.radians(5):
            report["joints"][name] = {"note": "reference moved less than 5 degrees"}
            continue
        errors = [np.sqrt(np.mean((got[lag:] - ref[: len(ref) - lag]) ** 2)) for lag in range(26)]
        best = int(np.argmin(errors))
        report["joints"][name] = {
            "rms_error_deg": round(float(np.degrees(errors[0])), 2),
            "best_delay_ms": round(best * POLICY_PERIOD_S * 1e3),
            "rms_error_at_best_delay_deg": round(float(np.degrees(errors[best])), 2),
            "reference_range_deg": round(float(np.degrees(np.ptp(ref))), 1),
        }
    return report


def load(path: Path) -> tuple[list[dict], list[tuple[float, dict]]]:
    """``(tracking frames, [(receive time, g1_debug fields)])`` of a capture."""
    if path.suffix == ".jsonl":
        tracking = []
        with path.open() as source:
            for line in source:
                row = json.loads(line)
                frame = row.get("response", {}).get("frame")
                if row.get("kind") != "sample" or not frame:
                    continue
                joints = frame.get("body", {})
                if not joints.get("tracked"):
                    continue
                body = {"positions": joints["positions_m"], "rotations": joints["rotations"]}
                converted = {
                    "sequence": int(frame["sequence"]),
                    "sample_time_ns": int(frame["sample_time_ns"]), "body": body,
                }
                if int(joints.get("update_sequence", 0)):
                    # Receipt counters restart with the backend run.
                    converted["body_update_sequence"] = (
                        int(frame.get("run", 0)), int(joints["update_sequence"])
                    )
                tracking.append(converted)
        return tracking, []
    rows = msgpack.unpackb(path.read_bytes(), raw=False)
    tracking, feedback = [], []
    for received, raw in rows:
        if raw.startswith(b"tracking"):
            tracking.append(msgpack.unpackb(raw[len(b"tracking") :], raw=False))
        elif raw.startswith(b"g1_debug"):
            feedback.append((received, msgpack.unpackb(raw[len(b"g1_debug") :], raw=False)))
    return tracking, feedback


def report(path: Path, max_bodies: int = 2000) -> dict:
    if max_bodies < 1:
        raise ValueError("max_bodies must be positive")
    tracking, feedback = load(path)
    bodies = [b for b in map(body_poses, tracking) if b is not None]
    stride = max(1, (len(bodies) + max_bodies - 1) // max_bodies)
    return {
        "capture": str(path),
        "recorded_body_frames": len(bodies),
        "skeleton_agreement": skeleton_agreement(bodies[::stride]),
        "bone_consistency": bone_consistency(bodies[::stride]),
        "body_timing": body_timing(tracking),
        "wrist_following": wrist_following(feedback),
    }


def record(path: Path, seconds: float, ports: list[int]) -> None:
    """Save every message of the manager and bridge sockets for ``seconds``."""
    import zmq

    if path.exists():
        raise SystemExit(f"{path} already exists; choose another name to keep it.")
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    for port in ports:
        socket.connect(f"tcp://127.0.0.1:{port}")
    rows, start = [], None
    print("Waiting for tracking...", flush=True)
    try:
        while start is None or time.monotonic() - start < seconds:
            if not socket.poll(100):
                continue
            raw = socket.recv()
            if start is None:
                if not raw.startswith(b"tracking"):
                    continue
                start = time.monotonic()
                print(f"Recording {seconds:g} s. Move every limb through its range.", flush=True)
            rows.append((time.monotonic(), raw))
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(linger=0)
        context.term()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        msgpack.pack(rows, file, use_bin_type=True)
    print(f"Saved {len(rows)} messages to {path}")


def print_report(result: dict) -> None:
    agreement = result["skeleton_agreement"]
    print(f"\nSkeleton agreement over {agreement['frames']} body frames")
    print("  bone (child joint)   direction error deg: p50 / p95 / max    PICO/SMPL length")
    ranked = sorted(
        agreement["bones"].items(), key=lambda item: -item[1]["direction_error_deg"]["p50"]
    )
    for name, bone in ranked:
        error = bone["direction_error_deg"]
        print(f"  {name:18s} {error['p50']:8.1f} / {error['p95']:6.1f} / {error['max']:6.1f}"
              f" {bone['length_pico_over_smpl']:18.2f}")  # fmt: skip
    print("  Differences can reflect rest axes, proportions, mapping or inconsistent source poses.")
    print("\nPICO bone consistency (before SMPL)")
    print(json.dumps(result["bone_consistency"], indent=2))
    print("\nBody timing")
    print(json.dumps(result["body_timing"], indent=2))
    print("\nWrist following (robot vs reference)")
    print(json.dumps(result["wrist_following"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record", help="save live manager and bridge messages")
    rec.add_argument("capture", type=Path)
    rec.add_argument("--seconds", type=float, default=30.0)
    rec.add_argument("--ports", type=int, nargs="+", default=[5556, 5558],
                     help="manager and bridge ports (default: 5556 5558)")  # fmt: skip
    rep = commands.add_parser("report", help="analyze a capture")
    rep.add_argument("capture", type=Path)
    rep.add_argument("--json", type=Path, help="also write the report as JSON")
    args = parser.parse_args()
    if args.command == "record":
        record(args.capture, args.seconds, args.ports)
        return
    result = report(args.capture)
    print_report(result)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
