"""Isaac Teleop client wrapping CloudXR + DeviceIO + OpenXR for the host.

Owns a CloudXR runtime (``CloudXRLauncher(run_embedded=True)``), opens an
OpenXR session, and starts the DeviceIO trackers (head, hands, controllers,
full body). The full-body vendor is the headset's own body tracking
(``body.pico-xr``) or an Xsens MVN suit (``body.xsens``), whose UDP stream the
``xsens_full_body_plugin`` child process republishes into the runtime.
Provides synchronous getters that the gear_sonic teleop readers poll on a
background thread.

See ``docs/source/tutorials/isaac_teleop_publisher_setup.md`` for the setup.
Requires ``isaaccapture[cloudxr]`` from ``pypi.nvidia.com`` (installed by
``install_scripts/install_pico.sh``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

import isaaccapture.deviceio as deviceio
import isaaccapture.oxr as oxr
from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.cloudxr.env_config import read_exported_env

from gear_sonic.utils.teleop.input_readers import BODY_VENDORS, XSENS_DEFAULT_PORT

# The plugin and the body.xsens vendor rendezvous on these; both sides must agree.
XSENS_COLLECTION_ID = "xsens_full_body"
XSENS_MAX_FLATBUFFER_SIZE = 4096  # bytes; one MVN pose is 784 B
# The plugin answers SIGTERM promptly, even inside its retry backoffs.
XSENS_PLUGIN_STOP_TIMEOUT_S = 5.0
# Printed once the plugin's OpenXR session and collection exist.
_XSENS_PLUGIN_READY_PREFIX = "listening on "

# argv: parent pid, then the plugin command.
_PDEATHSIG_EXEC = (
    "import ctypes, os, signal, sys\n"
    "ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG\n"
    "if os.getppid() != int(sys.argv[1]):\n"
    "    sys.exit(1)\n"
    "os.execv(sys.argv[2], sys.argv[2:])\n"
)

CLOUDXR_EULA_ENV = "CLOUDXR_EULA_ACCEPTED"
CLOUDXR_EULA_REFUSED_MESSAGE = "CloudXR EULA was not accepted"


def _default_pose_vec() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _controller_pose_vec(controller_data: Any) -> np.ndarray:
    """``[x, y, z, qx, qy, qz, qw]`` from ``ControllerSnapshot``; aim preferred, then grip."""
    if controller_data is None:
        return _default_pose_vec()
    controller_pose = None
    aim_pose = controller_data.aim_pose
    if aim_pose is not None and aim_pose.is_valid:
        controller_pose = aim_pose
    else:
        grip_pose = controller_data.grip_pose
        if grip_pose is not None and grip_pose.is_valid:
            controller_pose = grip_pose
    if controller_pose is None:
        return _default_pose_vec()
    p = controller_pose.pose.position
    o = controller_pose.pose.orientation
    pos = np.array([p.x, p.y, p.z], dtype=np.float64)
    quat = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)
    return np.concatenate([pos, quat])


def _pose_vec_from_head_data(head_pose: Any) -> np.ndarray:
    if head_pose is None or not head_pose.is_tracked:
        return _default_pose_vec()
    p = head_pose.pose.position
    o = head_pose.pose.orientation
    return np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float64)


def _controller_inputs(snapshot: Any) -> Any | None:
    if snapshot is None:
        return None
    return snapshot.inputs


def require_cloudxr_eula() -> None:
    """Exit unless the operator accepted the CloudXR EULA through the environment."""
    if os.environ.get(CLOUDXR_EULA_ENV) != "yes":
        print(CLOUDXR_EULA_REFUSED_MESSAGE, file=sys.stderr, flush=True)
        sys.exit(1)


class IsaacTeleopClient:
    """Single-process CloudXR + DeviceIO + OpenXR session.

    Args:
        app_name: OpenXR application name (shows up in CloudXR runtime logs).
        use_adb: If True, route signalling/WebRTC over a USB ADB tunnel
            (``setup_oob`` + ``usb_local`` on the launcher) so the headset
            reaches the host on loopback without needing shared Wi-Fi.
            Requires ``adb`` and ``coturn`` on PATH.
        cloudxr_install_dir: Where to install/find the CloudXR runtime.
            Defaults to ``~/.cloudxr``.
        cloudxr_env_config: Path to ``cloudxr.env`` (selects ``NV_DEVICE_PROFILE``).
            Defaults to ``~/cloudxr.env``; created by ``install_pico.sh``.
        body_vendor: Full-body source, one of ``BODY_VENDORS``.
        xsens_plugin: Path to the ``xsens_full_body_plugin`` binary (xsens only).
        xsens_address: IPv4 address of the one interface the MVN PC reaches
            (xsens only). The UDP stream carries no authentication.
        xsens_port: UDP port MVN Studio streams to (xsens only).
    """

    def __init__(
        self,
        app_name: str = "GearSonicIsaacTeleopClient",
        use_adb: bool = False,
        cloudxr_install_dir: str | Path | None = None,
        cloudxr_env_config: str | Path | None = None,
        body_vendor: str = "pico-xr",
        xsens_plugin: str | Path | None = None,
        xsens_address: str | None = None,
        xsens_port: int = XSENS_DEFAULT_PORT,
    ) -> None:
        if body_vendor not in BODY_VENDORS:
            raise ValueError(f"body_vendor must be one of {BODY_VENDORS}, got {body_vendor!r}")
        if body_vendor == "xsens" and (not xsens_plugin or not xsens_address):
            raise ValueError("body_vendor 'xsens' requires xsens_plugin and xsens_address")
        self._app_name = app_name
        self._use_adb = bool(use_adb)
        self._cloudxr_install_dir = str(
            cloudxr_install_dir if cloudxr_install_dir is not None else Path.home() / ".cloudxr"
        )
        self._cloudxr_env_config = str(
            cloudxr_env_config if cloudxr_env_config is not None else Path.home() / "cloudxr.env"
        )
        self._body_vendor = body_vendor
        self._xsens_plugin = str(xsens_plugin) if xsens_plugin else None
        self._xsens_address = xsens_address
        self._xsens_port = int(xsens_port)

        self._cloudxr_launcher: CloudXRLauncher | None = None
        self._xsens_proc: subprocess.Popen | None = None
        self._xsens_ready = threading.Event()

        self._exit_stack: ExitStack | None = None
        self._deviceio_session: Any = None
        self._head_tracker: Any = None
        self._hand_tracker: Any = None
        self._controller_tracker: Any = None
        self._body_tracker: Any = None

    @property
    def has_session(self) -> bool:
        return self._deviceio_session is not None

    def start_runtime(self) -> None:
        """Start the CloudXR runtime this process owns, then the Xsens plugin if selected.

        Call from the main thread: the embedded service installs SIGTERM/SIGINT
        handlers there so a signal still unwinds through :meth:`close`.
        """
        require_cloudxr_eula()
        # run_embedded: this process owns the runtime and stops it on exit. The
        # default would attach to (or start) a detached service that outlives us.
        self._cloudxr_launcher = CloudXRLauncher(
            install_dir=self._cloudxr_install_dir,
            env_config=self._cloudxr_env_config,
            accept_eula=True,
            setup_oob=self._use_adb,
            usb_local=self._use_adb,
            host_client=False,
            run_embedded=True,
        )
        if self._body_vendor == "xsens":
            self._start_xsens_plugin()

    def _start_xsens_plugin(self) -> None:
        cmd = [
            self._xsens_plugin,
            f"--collection-id={XSENS_COLLECTION_ID}",
            f"--address={self._xsens_address}",
            f"--port={self._xsens_port}",
            f"--max-flatbuffer-size={XSENS_MAX_FLATBUFFER_SIZE}",
        ]
        if sys.platform.startswith("linux"):
            # The plugin dies with this process even when it is killed outright.
            # Set after fork in a wrapper that execs the plugin (PDEATHSIG survives
            # exec) rather than in preexec_fn, which is unsafe in a threaded process.
            cmd = [sys.executable, "-c", _PDEATHSIG_EXEC, str(os.getpid()), *cmd]
        # The runtime's env file is how native processes find the runtime.
        env = {**os.environ, **read_exported_env(self._cloudxr_launcher.env_file)}
        print(f"Starting Xsens plugin: {' '.join(cmd)}", flush=True)
        self._xsens_ready.clear()
        self._xsens_proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        threading.Thread(
            target=self._forward_xsens_output, args=(self._xsens_proc,), daemon=True
        ).start()

    def _forward_xsens_output(self, proc: subprocess.Popen) -> None:
        """Relay plugin stdout (a pipe nobody else drains) and note when it is ready."""
        for line in proc.stdout:
            if line.startswith(_XSENS_PLUGIN_READY_PREFIX):
                self._xsens_ready.set()
            print(f"[xsens_full_body_plugin] {line}", end="", flush=True)

    def _wait_for_xsens_plugin(self, stop_event: threading.Event) -> None:
        """Block until the plugin created its collection; a reader started first never sees it."""
        code = self._xsens_proc.poll()
        if code is not None and self._xsens_ready.is_set():
            # It ran before, so its collection went away with it: start it again.
            print(
                f"IsaacTeleopClient: xsens_full_body_plugin exited with code {code}; restarting it.",
                file=sys.stderr,
                flush=True,
            )
            self._start_xsens_plugin()
        while not self._xsens_ready.wait(0.1):
            if stop_event.is_set():
                return
            code = self._xsens_proc.poll()
            if code is not None:
                print(
                    f"IsaacTeleopClient: xsens_full_body_plugin exited with code {code}; "
                    "the session opens without Xsens body tracking.",
                    file=sys.stderr,
                    flush=True,
                )
                return

    def open_session(self, stop_event: threading.Event) -> None:
        """Open the OpenXR + DeviceIO session.

        Blocks until a headset has connected to the runtime once (OpenXR waits
        for the HMD system), so call it off the main thread.
        """
        if self._xsens_proc is not None:
            self._wait_for_xsens_plugin(stop_event)
        stack = ExitStack()
        try:
            self._head_tracker = deviceio.HeadTracker()
            self._hand_tracker = deviceio.HandTracker()
            self._controller_tracker = deviceio.ControllerTracker()
            self._body_tracker = deviceio.FullBodyTracker()
            trackers = [
                self._head_tracker,
                self._hand_tracker,
                self._controller_tracker,
                self._body_tracker,
            ]
            vendor_params = (
                {
                    "collection_id": XSENS_COLLECTION_ID,
                    "max_flatbuffer_size": str(XSENS_MAX_FLATBUFFER_SIZE),
                }
                if self._body_vendor == "xsens"
                else {}
            )
            vendor_config = deviceio.VendorConfig(
                [
                    (
                        self._body_tracker,
                        deviceio.TrackerVendor(f"body.{self._body_vendor}", vendor_params),
                    )
                ]
            )
            # The vendor config goes to both calls: without it here the vendor's
            # extensions are never requested and the body never arrives.
            required_extensions = deviceio.DeviceIOSession.get_required_extensions(
                trackers, vendor_config=vendor_config
            )
            oxr_session = stack.enter_context(
                oxr.OpenXRSession(self._app_name, required_extensions)
            )
            handles = oxr_session.get_handles()
            self._deviceio_session = stack.enter_context(
                deviceio.DeviceIOSession.run(trackers, handles, vendor_config=vendor_config)
            )
            self._exit_stack = stack
            print(f"Isaac Teleop session initialized (body vendor: {self._body_vendor}).")
        except BaseException:
            stack.close()
            self._clear_session_refs()
            raise

    def start_streaming(self) -> None:
        """Start the runtime and open the session (blocking); for the CLI below."""
        self.start_runtime()
        self.open_session(threading.Event())

    def _clear_session_refs(self) -> None:
        self._exit_stack = None
        self._deviceio_session = None
        self._head_tracker = None
        self._hand_tracker = None
        self._controller_tracker = None
        self._body_tracker = None

    def close_session(self) -> None:
        """Close OpenXR + DeviceIO; the runtime and plugin keep running for a reopen."""
        if self._exit_stack is not None:
            try:
                self._exit_stack.close()
            except Exception as e:
                print(f"IsaacTeleopClient: closing the session failed ({e}).")
        self._clear_session_refs()

    def _get_tracker_data(self) -> dict[str, Any] | None:
        """Poll current tracking data and return it as a dictionary.

        Returns:
            Dict with keys ``left_controller``, ``right_controller``, ``head``,
            ``left_hand``, ``right_hand``, ``full_body``. Each value is the
            tracker's schema view, or None while that device is inactive.
            None when there is no session.
        """
        if self._deviceio_session is None:
            return None
        try:
            self._deviceio_session.update()
        except RuntimeError as e:
            print(f"IsaacTeleopClient: DeviceIO update failed ({e}); closing the session.")
            self.close_session()
            return None

        session = self._deviceio_session
        return {
            "left_controller": self._controller_tracker.get_left_controller(session),
            "right_controller": self._controller_tracker.get_right_controller(session),
            "head": self._head_tracker.get_head(session),
            "left_hand": self._hand_tracker.get_left_hand(session),
            "right_hand": self._hand_tracker.get_right_hand(session),
            "full_body": self._body_tracker.get_body_pose(session),
        }

    def get_pose_by_name(self, name: str) -> np.ndarray:
        """Return ``[x, y, z, qx, qy, qz, qw]`` for ``name`` ∈ {left_controller, right_controller, headset}."""
        raw = self._get_tracker_data()
        if raw is None:
            return _default_pose_vec()

        if name == "left_controller":
            return _controller_pose_vec(raw.get("left_controller"))
        if name == "right_controller":
            return _controller_pose_vec(raw.get("right_controller"))
        if name == "headset":
            return _pose_vec_from_head_data(raw.get("head"))
        raise ValueError(
            f"Invalid name: {name}. Valid names: 'left_controller', 'right_controller', 'headset'."
        )

    def _snapshot_side(self, raw: dict[str, Any] | None, side: str) -> Any:
        if raw is None:
            return None
        key = "left_controller" if side == "left" else "right_controller"
        return raw.get(key)

    def get_key_value_by_name(self, name: str) -> float:
        """Return trigger/grip value for ``name`` ∈ {left,right}_{trigger,grip}."""
        raw = self._get_tracker_data()
        if name == "left_trigger":
            inp = _controller_inputs(self._snapshot_side(raw, "left"))
            return float(inp.trigger_value) if inp is not None else 0.0
        if name == "right_trigger":
            inp = _controller_inputs(self._snapshot_side(raw, "right"))
            return float(inp.trigger_value) if inp is not None else 0.0
        if name == "left_grip":
            inp = _controller_inputs(self._snapshot_side(raw, "left"))
            return float(inp.squeeze_value) if inp is not None else 0.0
        if name == "right_grip":
            inp = _controller_inputs(self._snapshot_side(raw, "right"))
            return float(inp.squeeze_value) if inp is not None else 0.0
        raise ValueError(
            f"Invalid name: {name}. Valid names: "
            "'left_trigger', 'right_trigger', 'left_grip', 'right_grip'."
        )

    def get_button_state_by_name(self, name: str) -> bool:
        """Return True/False for face buttons and stick clicks.

        Valid names: ``A``, ``B``, ``X``, ``Y``,
        ``left_menu_button``, ``right_menu_button``,
        ``left_axis_click``, ``right_axis_click``.
        """
        raw = self._get_tracker_data()
        left = _controller_inputs(self._snapshot_side(raw, "left"))
        right = _controller_inputs(self._snapshot_side(raw, "right"))

        if name == "A":
            return right is not None and float(right.primary_click) > 0.5
        if name == "B":
            return right is not None and float(right.secondary_click) > 0.5
        if name == "X":
            return left is not None and float(left.primary_click) > 0.5
        if name == "Y":
            return left is not None and float(left.secondary_click) > 0.5
        if name in ("left_menu_button", "right_menu_button"):
            # Pico-specific menu button is not exposed via standard OpenXR
            # input bindings — DeviceIO doesn't surface it for the supported
            # controllers, so report not-pressed.
            return False
        if name == "left_axis_click":
            return left is not None and float(left.thumbstick_click) > 0.5
        if name == "right_axis_click":
            return right is not None and float(right.thumbstick_click) > 0.5
        raise ValueError(
            f"Invalid name: {name}. Valid names: 'A', 'B', 'X', 'Y', "
            "'left_menu_button', 'right_menu_button', 'left_axis_click', 'right_axis_click'."
        )

    def get_joystick_state(self, controller: str) -> list[float]:
        """Return ``[x, y]`` joystick state for ``controller`` ∈ {left, right}."""
        side = controller.lower()
        if side not in ("left", "right"):
            raise ValueError(
                f"Invalid controller: {controller}. Valid controllers: 'left', 'right'."
            )

        raw = self._get_tracker_data()
        inp = _controller_inputs(self._snapshot_side(raw, side))
        if inp is None:
            return [0.0, 0.0]
        return [float(inp.thumbstick_x), float(inp.thumbstick_y)]

    def get_timestamp_ns(self) -> int:
        """Return the host monotonic timestamp in nanoseconds."""
        return int(time.monotonic_ns())

    def _stop_xsens_plugin(self) -> None:
        proc = self._xsens_proc
        self._xsens_proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=XSENS_PLUGIN_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print("IsaacTeleopClient: xsens_full_body_plugin ignored SIGTERM; killing it.")
            proc.kill()
            proc.wait()

    def close(self) -> None:
        """Close the session, stop the Xsens plugin, then stop the CloudXR runtime."""
        self.close_session()
        self._stop_xsens_plugin()
        if self._cloudxr_launcher is not None:
            launcher = self._cloudxr_launcher
            self._cloudxr_launcher = None
            launcher.stop()


def main() -> None:
    """Poll ``IsaacTeleopClient`` getters periodically and print (Ctrl+C to stop).

    Usage::

        source .venv_teleop/bin/activate
        export CLOUDXR_EULA_ACCEPTED=yes

        # Setup verification: bring CloudXR up, populate ~/.cloudxr/, exit clean
        python -m gear_sonic.utils.teleop.isaac_teleop_client --init-only

        # Live print (default): poll getters and print until Ctrl+C
        python -m gear_sonic.utils.teleop.isaac_teleop_client --hz 5
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Print IsaacTeleopClient getter outputs at a fixed rate."
    )
    parser.add_argument("--hz", type=float, default=5.0, help="Print rate in Hz (default: 5)")
    parser.add_argument(
        "--use-adb", action="store_true", help="Route CloudXR over USB ADB (OOB / usb-local)."
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help=(
            "Bring CloudXR up to populate ~/.cloudxr/ ownership + env file, "
            "then exit. Use as a setup verification step."
        ),
    )
    args = parser.parse_args()

    client = IsaacTeleopClient(use_adb=args.use_adb)

    if args.init_only:
        client.start_runtime()
        # Confirm the runtime actually populated the install dir before we tear down.
        run_env = Path.home() / ".cloudxr" / "run" / "cloudxr.env"
        if run_env.exists():
            print(f"[OK] CloudXR runtime initialized; {run_env} written.")
            client.close()
            sys.exit(0)
        print(
            f"[ERROR] CloudXR runtime did not populate {run_env} — check the "
            "earlier log for IsaacTeleopClient errors.",
            file=sys.stderr,
        )
        client.close()
        sys.exit(1)

    client.start_streaming()

    period = 1.0 / max(0.1, float(args.hz))
    pose_names = ("left_controller", "right_controller", "headset")
    key_names = ("left_trigger", "right_trigger", "left_grip", "right_grip")
    button_names = (
        "A",
        "B",
        "X",
        "Y",
        "left_menu_button",
        "right_menu_button",
        "left_axis_click",
        "right_axis_click",
    )
    joy_sides = ("left", "right")

    try:
        while True:
            t0 = time.time()
            print("=" * 72, flush=True)
            print(
                f"time={time.strftime('%H:%M:%S')}  get_timestamp_ns={client.get_timestamp_ns()}",
                flush=True,
            )

            for name in pose_names:
                v = client.get_pose_by_name(name)
                print(
                    f"  get_pose_by_name({name!r}): "
                    f"{np.array2string(v, precision=4, suppress_small=True)}",
                    flush=True,
                )

            for name in key_names:
                print(
                    f"  get_key_value_by_name({name!r}): {client.get_key_value_by_name(name):.4f}",
                    flush=True,
                )

            for name in button_names:
                print(
                    f"  get_button_state_by_name({name!r}): {client.get_button_state_by_name(name)}",
                    flush=True,
                )

            for side in joy_sides:
                j = client.get_joystick_state(side)
                print(f"  get_joystick_state({side!r}): {j}", flush=True)

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
