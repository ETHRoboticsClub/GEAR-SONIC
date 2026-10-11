"""Write ethrc/g1/<variant>/sonic.yaml: the sonic/v1 manifests that run NVIDIA's
released G1 networks, unchanged, on the ETHRC controls stack.

The release on Hugging Face carries the networks and the observation config but not
the joint data the deploy binary compiles in. That data is read here from
policy_parameters.hpp; the planner settings are cited from the deploy sources. The
manifest names NVIDIA's files by their paths in the release and binds them by SHA-256,
so the robot installs the release as published and places this file beside it.

    python ethrc/g1_sonic_bundle.py --urdf-sha256 <hex>     # regenerate
    python ethrc/g1_sonic_bundle.py --urdf-sha256 <hex> --check

--urdf-sha256 is the controls repo's model identity for its G1 description
(export_unitree_rl_gym.model_urdf_sha256 there).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import operator
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"
OUTPUT_DIR = ROOT / "ethrc/g1"

HF_REPOSITORY = "nvidia/GEAR-SONIC"
HF_REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
# The release's encoder/decoder variants, by their directory on the Hub (README
# "Model variants"). They share the planner, joints and control rate.
VARIANTS = ("sonic_v1_1", "low_latency")
# deploy.sh's default planner (planner/target_vel/V2/), published at the release root.
PLANNER = "planner_sonic.onnx"
PLANNER_VERSION = 2

# g1_deploy_onnx_ref.cpp: policy 50 Hz, planner_dt_ 0.1 s, replan_interval_ 1.0 s and
# replan_interval_running_ 0.1 s.
CONTROL_PERIOD_S = 0.02
PLANNER_PERIOD_S = 0.1
REPLAN_PERIOD_S = 1.0
RUN_REPLAN_PERIOD_S = 0.1
# localmotion_kplanner.hpp, PlannerConfig::default_height.
DEFAULT_ROOT_HEIGHT_M = 0.78874
# LocomotionMode codes (localmotion_kplanner.hpp) with the operator speed bounds
# the gamepad input clamps them to (input_interface/gamepad.hpp:354-363; slow walk
# and run match keyboard_handler.hpp:271-276). The planner's nominal walk band is
# 0.8-2.5 m/s (localmotion_kplanner.hpp:81); the operator bound stops at 1.5.
PLANNER_MODES = {
    "idle": {"code": 0},
    "slow_walk": {"code": 1, "speed_m_s": [0.2, 0.8]},
    "walk": {"code": 2, "speed_m_s": [0.8, 1.5]},
    "run": {"code": 3, "speed_m_s": [1.5, 3.0], "replan_period_s": RUN_REPLAN_PERIOD_S},
    "squat": {"code": 4, "height_m": [0.2, 0.8]},
}
# observation_config.yaml's encoder mode names.
ENCODER_MODES = {"robot_joints": "g1", "vr_three_point": "teleop", "smpl": "smpl"}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow}


def _eval(expr: str, constants: dict[str, float]) -> float:
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            return constants[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError(f"unsupported expression {expr!r}")

    return float(walk(ast.parse(expr.strip(), mode="eval").body))


def _strip_comment(line: str) -> tuple[str, str]:
    code, _, comment = line.partition("//")
    return code, comment


def parse_header(text: str) -> dict:
    """The header's scalar constants and 29-entry arrays, each array entry with the
    joint name its trailing comment gives (empty for the index arrays)."""
    constants: dict[str, float] = {}
    for match in re.finditer(r"^const double (\w+) = ([^;]+);", text, re.MULTILINE):
        constants[match.group(1)] = _eval(_strip_comment(match.group(2))[0], constants)
    arrays: dict[str, list[tuple[float, str]]] = {}
    pattern = r"std::array<\w+, 29> (\w+) = \{(.*?)\};"
    for match in re.finditer(pattern, text, re.DOTALL):
        entries: list[tuple[float, str]] = []
        for line in match.group(2).splitlines():
            code, comment = _strip_comment(line)
            name = comment.split()[0] if comment.split() else ""
            values = [v for v in code.split(",") if v.strip()]
            if len(values) > 1:
                entries += [(_eval(v, constants), "") for v in values]
            elif values:
                entries.append((_eval(values[0], constants), name))
        if len(entries) != 29:
            raise ValueError(f"{match.group(1)} has {len(entries)} entries, not 29")
        arrays[match.group(1)] = entries
    vectors: dict[str, list[int]] = {}
    for match in re.finditer(r"std::vector<int> (\w+) = \{([^}]*)\};", text):
        vectors[match.group(1)] = [int(v) for v in match.group(2).split(",")]
    return {"arrays": arrays, "vectors": vectors}


def joints(header: dict) -> tuple[list[dict], list[str], dict[str, list[str]]]:
    """Joints in the network's (IsaacLab) order, the MuJoCo order the planner uses,
    and the encoder's joint subsets."""
    arrays, vectors = header["arrays"], header["vectors"]
    mujoco_names = [name for _, name in arrays["default_angles"]]
    for key in ("g1_action_scale", "kps", "kds"):
        names = [name for _, name in arrays[key]]
        if names != mujoco_names:
            raise ValueError(f"{key} is not in default_angles' joint order")
    # mujoco_to_isaaclab[i] is the MuJoCo index of network joint i.
    policy_of = [int(v) for v, _ in arrays["mujoco_to_isaaclab"]]
    if sorted(policy_of) != list(range(29)):
        raise ValueError("mujoco_to_isaaclab is not a permutation")
    entries = []
    for m in policy_of:
        entries.append({
            "name": mujoco_names[m],
            "default_position_rad": arrays["default_angles"][m][0],
            "action_scale_rad": arrays["g1_action_scale"][m][0],
            "stiffness_nm_per_rad": arrays["kps"][m][0],
            "damping_nm_s_per_rad": arrays["kds"][m][0],
        })
    policy_names = [e["name"] for e in entries]
    subsets = {
        "lowerbody": [policy_names[i]
                      for i in vectors["lower_body_joint_isaaclab_order_in_isaaclab_index"]],
        "wrists": [policy_names[i]
                   for i in vectors["wrist_joint_isaaclab_order_in_isaaclab_index"]],
    }
    return entries, mujoco_names, subsets


def files(variant: str) -> tuple[str, str, str]:
    """The variant's encoder, decoder and observation config paths in the release."""
    return (f"{variant}/model_encoder.onnx", f"{variant}/model_decoder.onnx",
            f"{variant}/observation_config.yaml")


def release_digests() -> dict[str, str]:
    """SHA-256 of each release file at HF_REVISION, as the Hub reports it."""
    paths = [PLANNER] + [path for variant in VARIANTS for path in files(variant)]
    body = urllib.parse.urlencode([("paths", p) for p in paths] + [("expand", "true")])
    url = f"https://huggingface.co/api/models/{HF_REPOSITORY}/paths-info/{HF_REVISION}"
    with urllib.request.urlopen(url, data=body.encode(), timeout=60) as response:
        info = {entry["path"]: entry for entry in json.load(response)}
    digests = {}
    for path in paths:
        lfs = info[path].get("lfs")
        if lfs:
            digests[path] = lfs["oid"]
        else:
            # A small file is stored in git; hash its bytes.
            raw = f"https://huggingface.co/{HF_REPOSITORY}/resolve/{HF_REVISION}/{path}"
            with urllib.request.urlopen(raw, timeout=60) as response:
                digests[path] = hashlib.sha256(response.read()).hexdigest()
    return digests


def manifest(variant: str, urdf_sha256: str, digests: dict[str, str]) -> dict:
    encoder, decoder, observation = files(variant)
    entries, planner_order, subsets = joints(parse_header(HEADER.read_text()))
    return {
        "schema": "sonic/v1",
        "robot_identity": {"urdf_sha256": urdf_sha256},
        "source": {"repository": f"hf:{HF_REPOSITORY}", "revision": HF_REVISION},
        "command_frame_id": "pelvis",
        "control_period_s": CONTROL_PERIOD_S,
        "joints": entries,
        "networks": {
            "encoder": {"file": encoder, "sha256": digests[encoder]},
            "decoder": {"file": decoder, "sha256": digests[decoder]},
            "planner": {
                "file": PLANNER,
                "sha256": digests[PLANNER],
                "version": PLANNER_VERSION,
                "period_s": PLANNER_PERIOD_S,
                "replan_period_s": REPLAN_PERIOD_S,
                "default_root_height_m": DEFAULT_ROOT_HEIGHT_M,
                "joint_order": planner_order,
                "modes": PLANNER_MODES,
            },
        },
        "observation": {"file": observation, "sha256": digests[observation]},
        "encoder_modes": ENCODER_MODES,
        "joint_subsets": subsets,
    }


def render(document: dict) -> str:
    source = Path(__file__).relative_to(ROOT)
    head = f"# Generated by {source} from {HEADER.relative_to(ROOT)}; do not edit.\n"
    return head + yaml.safe_dump(document, sort_keys=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--urdf-sha256", required=True,
                        help="the controls repo's model identity for its G1 description")
    parser.add_argument("--check", action="store_true",
                        help="fail if a committed manifest differs from a fresh one")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{64}", args.urdf_sha256):
        parser.error("--urdf-sha256 must be 64 lowercase hex digits")
    digests = release_digests()
    stale = 0
    for variant in VARIANTS:
        output = OUTPUT_DIR / variant / "sonic.yaml"
        text = render(manifest(variant, args.urdf_sha256, digests))
        if args.check:
            if not output.is_file() or output.read_text() != text:
                print(f"{output.relative_to(ROOT)} is stale; regenerate it", file=sys.stderr)
                stale = 1
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        print(f"wrote {output.relative_to(ROOT)}")
    return stale


if __name__ == "__main__":
    sys.exit(main())
