"""Local diagnostics and startup gating for the standalone PICO simulation."""

import argparse
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time

import msgpack
import zmq

ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / ".local/pico-sim/readiness.json"
PROFILE = "teleop-sim"


def command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def field(name):
    return command("nmcli", "-g", name, "connection", "show", PROFILE)


def network_status():
    if not shutil.which("nmcli") or not field("connection.id"):
        print("[NETWORK] Saved teleop-sim profile unavailable. Configure it in Network settings first.")
        return False
    interface = field("connection.interface-name")
    active = command("nmcli", "-g", "GENERAL.CONNECTION", "device", "show", interface) == PROFILE
    address = command("nmcli", "-g", "IP4.ADDRESS", "device", "show", interface)
    band = {"a": "5 GHz", "bg": "2.4 GHz"}.get(field("802-11-wireless.band"), "automatic")
    print(f"[NETWORK] {PROFILE}: {'ACTIVE' if active else 'OFF'}; {interface}; {band}; "
          f"channel {field('802-11-wireless.channel')}; PC {address or 'no IPv4'}")
    print("          PICO PC Service must use the hotspot IPv4 address (normally 10.42.0.1).")
    iw = shutil.which("iw")
    local_iw = Path.home() / ".local/share/gear-sonic/wifi-diagnostics/root/usr/sbin/iw"
    if not iw and local_iw.is_file():
        iw = str(local_iw)
    if active and iw:
        try:
            stations = subprocess.run([iw, "dev", interface, "station", "dump"],
                                      capture_output=True, text=True, timeout=5)
            if stations.returncode == 0:
                count = sum(line.startswith("Station ") for line in stations.stdout.splitlines())
                print(f"          Associated Wi-Fi clients: {count} (association alone does not prove tracking works).")
            else:
                print("          Wi-Fi client association details unavailable with current permissions.")
        except (OSError, subprocess.TimeoutExpired):
            print("          Wi-Fi client association details unavailable.")
    if band == "2.4 GHz":
        print("          Current profile differs from the measured 5 GHz setup; validate this band separately.")
    return active


def network_action(action):
    if action == "status":
        return 0 if network_status() else 1
    if not field("connection.id"):
        raise SystemExit("Saved teleop-sim profile is missing; no new profile was created.")
    if field("802-11-wireless.mode") != "ap" or field("ipv4.method") != "shared":
        raise SystemExit("teleop-sim must be an AP with shared IPv4. Check its saved profile.")
    if action == "restart":
        if subprocess.run(["tmux", "-L", "gear-sonic-pico", "has-session", "-t", "pico-sim"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise SystemExit("Stop the simulation first: bash install_scripts/run_pico_sim.sh stop")
        # Restart the existing profile, preserving SSID, password and headset credentials.
        active = PROFILE in command("nmcli", "-g", "NAME", "connection", "show", "--active").splitlines()
        if active:
            subprocess.run(["nmcli", "--wait", "15", "connection", "down", "id", PROFILE], check=True)
    interface = field("connection.interface-name")
    if command("nmcli", "-g", "GENERAL.CONNECTION", "device", "show", interface) != PROFILE:
        subprocess.run(["nmcli", "--wait", "20", "connection", "up", "id", PROFILE], check=True)
    return 0 if network_status() else 1


def live_status(seconds=1.0):
    context = zmq.Context()
    subscribers = []
    latest = {}
    for port, topic in ((5558, b"tracking"), (5557, b"g1_debug")):
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.SUBSCRIBE, topic)
        subscriber.setsockopt(zmq.CONFLATE, 1)
        subscriber.connect(f"tcp://127.0.0.1:{port}")
        subscribers.append((subscriber, topic))
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            for subscriber, topic in subscribers:
                try:
                    payload = subscriber.recv(zmq.NOBLOCK)
                    latest[topic] = (time.monotonic(), msgpack.unpackb(payload[len(topic):], raw=False))
                except (zmq.Again, ValueError, msgpack.UnpackException):
                    pass
            time.sleep(0.01)
    finally:
        for subscriber, _ in subscribers:
            subscriber.close(linger=0)
        context.term()
    now = time.monotonic()
    stamp, frame = latest.get(b"tracking", (0, {}))
    sample_age = now - frame.get("sample_time_ns", 0) / 1e9
    tracking_live = 0 <= now - stamp < 0.2 and 0 <= sample_age < 0.2
    present = [tracking_live and frame.get(key) is not None
               for key in ("head", "left_controller", "right_controller", "body")]
    print("[TRACKING] " + " ".join(f"{name}={'OK' if ok else 'MISSING'}" for name, ok in
                                  zip(("head", "left", "right", "body"), present)))
    stamp, _ = latest.get(b"g1_debug", (0, {}))
    policy_live = 0 <= now - stamp < 0.2
    print(f"[POLICY] {'ACTIVE: live action feedback' if policy_live else 'No live action feedback'}")
    try:
        status = json.loads(STATUS.read_text())
        if not 0 <= now - status["sample_time"] < 1.5:
            raise ValueError("stale simulator status")
    except (OSError, ValueError, KeyError):
        print("[SIMULATOR] No current readiness report. Start the updated launcher.")
        return False
    band = "ATTACHED" if status["band_enabled"] else "RELEASED"
    ready = status["ready"] and policy_live and all(present[:3])
    print(f"[SIMULATOR] Suspension {band}; pelvis height {status['height']:.2f} m")
    print(f"[RELEASE] {'READY' if ready else status['reason'] or 'Wait for fresh headset/controller tracking.'}")
    if not policy_live:
        print("          Init Done means loaded. Calibrate/start once after the manager is ready.")
    return ready


def wait_controller(log_path, timeout):
    print("Waiting for controller Init Done before accepting PICO calibration/start.", flush=True)
    deadline = time.monotonic() + timeout
    next_report = time.monotonic() + 15
    next_process_check = 0.0
    while time.monotonic() < deadline:
        try:
            ready = "Init Done" in log_path.read_text(errors="replace")
        except FileNotFoundError:
            ready = False
        if ready:
            # Let the already-created ZMQ subscriber finish its PUB/SUB handshake.
            time.sleep(0.5)
            print("Controller initialized. Starting manager; calibrate/start only after its controls appear.", flush=True)
            return
        if time.monotonic() >= next_process_check:
            for component in ("controller", "simulator"):
                dead = command("tmux", "-L", "gear-sonic-pico", "display-message", "-p", "-t",
                               f"pico-sim:{component}", "#{pane_dead}")
                if dead == "1":
                    raise SystemExit(f"{component} exited. Check .local/pico-sim/logs/{component}.log, then restart.")
            next_process_check = time.monotonic() + 1.0
        if time.monotonic() >= next_report:
            print("Still loading controller/TensorRT. Keep suspension attached; do not calibrate yet.", flush=True)
            next_report = time.monotonic() + 15
        time.sleep(0.25)
    raise SystemExit(f"Controller did not initialize within {timeout:g}s. See {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    network = sub.add_parser("network")
    network.add_argument("operation", choices=("status", "up", "restart"), default="status", nargs="?")
    sub.add_parser("status")
    wait = sub.add_parser("wait-controller")
    wait.add_argument("--log", type=Path, required=True)
    wait.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.action == "network":
        return network_action(args.operation)
    if args.action == "wait-controller":
        wait_controller(args.log, args.timeout)
        return 0
    network_status()
    try:
        with socket.create_connection(("127.0.0.1", 60061), timeout=0.2):
            print("[PC SERVICE] Local SDK port is listening (this does not prove headset connectivity).")
    except OSError:
        print("[PC SERVICE] Not listening. Start the launcher or check service.log.")
    return 0 if live_status() else 1


if __name__ == "__main__":
    raise SystemExit(main())
