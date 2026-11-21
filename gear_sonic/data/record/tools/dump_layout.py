"""Write ``g1_sonic_layout.json`` from the Sonic G1 robot model.

Uses the same constructor as ``gear_sonic.data.features_sonic_vla.get_g1_robot_model``
(``waist_location="lower_and_upper_body"``, ``high_elbow_pose=False``).
``observation.state`` follows ``RobotModel.joint_names``. On this model that
list is also pinocchio ``idx_q`` order, so a slice of ``joint_names`` is a
slice of the configuration vector.

Groups are the seven supplemental leaf groups. A group is recorded only when
its joints occupy one contiguous half-open range of ``joint_names``. This
script exits non-zero and writes nothing if a group is split, if two groups
overlap, or if an actuated joint would be left out.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

GROUP_ORDER = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
)

_LAYOUT_PATH = Path(__file__).resolve().parent.parent / "g1_sonic_layout.json"


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _model_data_dir():
    from gear_sonic.data.robot_model.instantiation import g1 as g1_factory

    return Path(g1_factory.__file__).resolve().parent.parent / "model_data" / "g1"


def _urdf_joints(urdf_text: str) -> list[tuple[str, str]]:
    return re.findall(r'<joint name="([^"]+)" type="([^"]+)"', urdf_text)


def _mujoco_body_joint_names(xml_text: str) -> list[str]:
    names = re.findall(r'<joint name="([^"]+)"', xml_text)
    return [name for name in names if name != "floating_base_joint" and "hand" not in name]


def build_layout(model) -> dict:
    """Return the layout dict, or raise ValueError if a group is not one slice."""
    joint_names = list(model.joint_names)
    if len(joint_names) != model.num_joints or len(joint_names) != model.num_dofs:
        raise ValueError(
            f"joint_names ({len(joint_names)}), num_joints ({model.num_joints}), "
            f"and nq ({model.num_dofs}) disagree; refusing to pick a state vector"
        )
    for index, name in enumerate(joint_names):
        if model.dof_index(name) != index:
            raise ValueError(
                f"{name} is joint_names[{index}] but idx_q={model.dof_index(name)}; "
                "observation.state order is ambiguous"
            )

    index_of = {name: index for index, name in enumerate(joint_names)}
    groups = {}
    occupied: dict[int, str] = {}
    for group_name in GROUP_ORDER:
        info = model.supplemental_info.joint_groups[group_name]
        if info["groups"]:
            raise ValueError(f"{group_name} has subgroups; refusing to invent a flat slice")
        declared = list(info["joints"])
        if len(declared) != len(set(declared)):
            raise ValueError(f"{group_name} lists a joint more than once")
        missing = [name for name in declared if name not in index_of]
        if missing:
            raise ValueError(f"{group_name} names absent from the model: {missing}")
        positions = [index_of[name] for name in declared]
        start, end = min(positions), max(positions) + 1
        if sorted(positions) != list(range(start, end)):
            raise ValueError(
                f"{group_name} is not one contiguous observation.state slice: "
                f"indices {sorted(positions)}"
            )
        for position in range(start, end):
            previous = occupied.get(position)
            if previous is not None:
                raise ValueError(f"{joint_names[position]} is in both {previous} and {group_name}")
            occupied[position] = group_name
        groups[group_name] = {
            "start": start,
            "end": end,
            "names": joint_names[start:end],
        }

    assigned = {name for group in groups.values() for name in group["names"]}
    actuated = list(model.supplemental_info.body_actuated_joints)
    actuated += list(model.supplemental_info.left_hand_actuated_joints)
    actuated += list(model.supplemental_info.right_hand_actuated_joints)
    left_out = [name for name in actuated if name not in assigned]
    if left_out:
        raise ValueError(f"actuated joints not assigned to a contiguous group: {left_out}")

    model_data = _model_data_dir()
    xml_body = _mujoco_body_joint_names((model_data / "g1_29dof_with_hand.xml").read_text())
    supplemental_body = list(model.supplemental_info.body_actuated_joints)
    hands = set(groups["left_hand"]["names"] + groups["right_hand"]["names"])
    non_hand = [name for name in joint_names if name not in hands]
    if supplemental_body == xml_body == non_hand and len(supplemental_body) == 29:
        body_joint_names = supplemental_body
        body_source = (
            "The 29 MuJoCo body joints. "
            "g1_29dof_with_hand.xml (floating base and hand joints removed) "
            "matches G1SupplementalInfo.body_actuated_joints and also matches "
            "joint_names with both hand groups removed. "
            "This is g1_debug body_q / body_dq / last_action order "
            "(left leg, right leg, waist, left arm, right arm). "
            "It is not one observation.state slice: the left hand sits between the arms."
        )
    elif len(supplemental_body) == 29 and set(supplemental_body) == set(non_hand):
        body_joint_names = supplemental_body
        body_source = (
            "G1SupplementalInfo.body_actuated_joints (29 names). "
            "The MuJoCo xml order did not match this list exactly, so this is the "
            "supplemental body order rather than a raw xml scrape. "
            f"xml_body={xml_body}"
        )
    else:
        body_joint_names = non_hand
        body_source = (
            "Fallback: joint_names that are not in left_hand or right_hand. "
            "The 29 MuJoCo body joints could not be identified as that same set. "
            f"supplemental_len={len(supplemental_body)} xml_len={len(xml_body)} "
            f"non_hand_len={len(non_hand)}"
        )

    urdf_joints = _urdf_joints((model_data / "g1_29dof_with_hand.urdf").read_text())
    omitted = [
        {"name": name, "type": joint_type}
        for name, joint_type in urdf_joints
        if name not in index_of
    ]
    unassigned = [name for name in joint_names if name not in assigned]
    if unassigned:
        unassigned_note = (
            "These joint_names are outside the seven groups. "
            "They are listed rather than forced into a neighboring slice. " + ", ".join(unassigned)
        )
    else:
        unassigned_note = (
            "Empty: every joint in joint_names is an actuated joint in exactly one "
            "of the seven groups. Fixed URDF joints and the floating base are absent "
            "from the pinocchio model (see omitted_joints), so they are not unassigned "
            "state columns."
        )

    return {
        "joint_names": joint_names,
        "state_dim": len(joint_names),
        "index_convention": (
            "groups.start is inclusive and groups.end is exclusive. "
            "names == joint_names[start:end] == observation.state[start:end]."
        ),
        "groups": groups,
        "body_joint_names": body_joint_names,
        "body_joint_names_source": body_source,
        "hand_actuator_joint_names": {
            "left": list(model.supplemental_info.left_hand_actuated_joints),
            "right": list(model.supplemental_info.right_hand_actuated_joints),
            "note": (
                "g1_debug left_hand_q / left_hand_dq / last_left_hand_action "
                "(and the right-hand keys) follow this supplemental actuated order, "
                "which is the order get_configuration_from_actuated_joints scatters "
                "into q. motor_0..6 on the hand velocity and hand action columns "
                "use this order. groups.left_hand.names / groups.right_hand.names "
                "are observation.state order and differ: pinocchio places index, "
                "then middle, then thumb."
            ),
        },
        "unassigned": unassigned,
        "unassigned_note": unassigned_note,
        "omitted_joints": omitted,
        "omitted_joints_note": (
            "Present in g1_29dof_with_hand.urdf but not in RobotModel.joint_names. "
            "Pinocchio's URDF loader drops fixed joints. instantiate_g1_robot_model "
            "does not add a floating base (set_floating_base stays false)."
        ),
        "model": {
            "factory": "gear_sonic.data.features_sonic_vla.get_g1_robot_model",
            "waist_location": "lower_and_upper_body",
            "high_elbow_pose": False,
            "nq": int(model.num_dofs),
        },
    }


def main() -> int:
    from gear_sonic.data.features_sonic_vla import get_g1_robot_model

    model = get_g1_robot_model(waist_location="lower_and_upper_body", high_elbow_pose=False)
    try:
        layout = build_layout(model)
    except ValueError as exc:
        return _fail(str(exc))
    _LAYOUT_PATH.write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    spans = " ".join(
        f"{name}[{layout['groups'][name]['start']}:{layout['groups'][name]['end']}]"
        for name in GROUP_ORDER
    )
    print(
        f"wrote {_LAYOUT_PATH} state_dim={layout['state_dim']} "
        f"unassigned={layout['unassigned'] or '[]'} {spans}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
