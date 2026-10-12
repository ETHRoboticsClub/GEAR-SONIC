# ETHRC bundles

For the current PC/G1 network setup, Pico launch commands, stop controls, and
troubleshooting, see [Pico teleoperation setup](pico_teleop_setup.md).

`sonic/v1` manifests that run NVIDIA's released SONIC networks on the ETHRC
controls stack (`docs/sonic.md` there). A manifest names the release's files by
their paths on Hugging Face and binds them by SHA-256; the networks are never
copied or modified. A robot installs a bundle by downloading the release files
at the pinned revision and placing the manifest beside them.

| Bundle | Robot | Release |
|---|---|---|
| `g1/sonic_v1_1/sonic.yaml` | Unitree G1 29-DoF | `nvidia/GEAR-SONIC` `sonic_v1_1`, planner V2 |
| `g1/low_latency/sonic.yaml` | Unitree G1 29-DoF | `nvidia/GEAR-SONIC` `low_latency` (4-frame SMPL lookahead), planner V2 |

The G1 manifests are generated from `policy_parameters.hpp`, which holds the joint
order, default pose, action scales and gains the release does not carry:

```sh
python ethrc/g1_sonic_bundle.py --urdf-sha256 <G1 model identity>
python ethrc/g1_sonic_bundle.py --urdf-sha256 <G1 model identity> --check
```

The model identity comes from the controls repo's G1 description; regenerate
when it changes.
