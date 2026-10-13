"""IsaacTeleopClient lifecycle against stand-ins for the isaaccapture modules.

Checks what the real runtime cannot show without hardware: the EULA gate, that
the CloudXR runtime is owned by this process, and that the Xsens plugin starts
with the runtime's environment after the runtime and before the DeviceIO
session, and is stopped on close.
"""

import importlib
import stat
import sys
import threading
import types

import pytest

RUNTIME_ENV = {"XR_RUNTIME_JSON": "/tmp/fake-cloudxr/openxr_cloudxr.json"}


class _Context:
    def __init__(self, events, name, value=None):
        self.events, self.name, self.value = events, name, value

    def __enter__(self):
        self.events.append(f"enter {self.name}")
        return self.value if self.value is not None else self

    def __exit__(self, *exc):
        self.events.append(f"exit {self.name}")


def _fake_isaaccapture(events, env_file):
    deviceio = types.ModuleType("isaaccapture.deviceio")
    for name in ("HeadTracker", "HandTracker", "ControllerTracker", "FullBodyTracker"):
        setattr(deviceio, name, type(name, (), {}))

    class TrackerVendor:
        def __init__(self, id, params=None):
            self.id, self.params = id, dict(params or {})

    class VendorConfig:
        def __init__(self, tracker_vendors=()):
            self.tracker_vendors = list(tracker_vendors)

    class DeviceIOSession:
        @staticmethod
        def get_required_extensions(trackers, vendor_config=None):
            events.append(("get_required_extensions", vendor_config))
            return ["XR_fake"]

        @staticmethod
        def run(trackers, handles, vendor_config=None):
            events.append(("run", vendor_config))
            return _Context(events, "deviceio", types.SimpleNamespace(update=lambda: None))

    deviceio.TrackerVendor, deviceio.VendorConfig = TrackerVendor, VendorConfig
    deviceio.DeviceIOSession = DeviceIOSession

    oxr = types.ModuleType("isaaccapture.oxr")

    class OpenXRSession(_Context):
        def __init__(self, app_name, extensions):
            super().__init__(events, "openxr")
            events.append(("OpenXRSession", list(extensions)))

        def get_handles(self):
            return "handles"

    oxr.OpenXRSession = OpenXRSession

    cloudxr = types.ModuleType("isaaccapture.cloudxr")

    class CloudXRLauncher:
        def __init__(self, **kwargs):
            events.append(("CloudXRLauncher", kwargs))
            self.env_file = env_file

        def stop(self):
            events.append("launcher.stop")

    cloudxr.CloudXRLauncher = CloudXRLauncher
    env_config = types.ModuleType("isaaccapture.cloudxr.env_config")
    env_config.read_exported_env = lambda path: dict(RUNTIME_ENV) if path == env_file else {}

    package = types.ModuleType("isaaccapture")
    return {
        "isaaccapture": package,
        "isaaccapture.deviceio": deviceio,
        "isaaccapture.oxr": oxr,
        "isaaccapture.cloudxr": cloudxr,
        "isaaccapture.cloudxr.env_config": env_config,
    }


@pytest.fixture
def client_module(monkeypatch, tmp_path):
    events = []
    for name, module in _fake_isaaccapture(events, tmp_path / "cloudxr.env").items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "gear_sonic.utils.teleop.isaac_teleop_client", raising=False)
    module = importlib.import_module("gear_sonic.utils.teleop.isaac_teleop_client")
    yield module, events
    sys.modules.pop("gear_sonic.utils.teleop.isaac_teleop_client", None)


@pytest.fixture
def fake_plugin(tmp_path):
    """A plugin that records its arguments and environment, then waits for SIGTERM."""
    record = tmp_path / "plugin_record"
    script = tmp_path / "xsens_full_body_plugin"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" > {record}\n'
        f'echo "$XR_RUNTIME_JSON" >> {record}\n'
        'echo "listening on 192.168.1.5:9764 -> push_buffer"\n'
        "exec sleep 60\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script, record


def test_without_the_eula_the_client_exits(client_module, monkeypatch, capsys):
    module, events = client_module
    monkeypatch.delenv("CLOUDXR_EULA_ACCEPTED", raising=False)
    client = module.IsaacTeleopClient()

    with pytest.raises(SystemExit) as exit_info:
        client.start_runtime()

    assert exit_info.value.code != 0
    assert "CloudXR EULA was not accepted" in capsys.readouterr().err
    assert events == []


def test_runtime_is_owned_by_this_process(client_module, monkeypatch):
    module, events = client_module
    monkeypatch.setenv("CLOUDXR_EULA_ACCEPTED", "yes")
    client = module.IsaacTeleopClient()

    client.start_runtime()
    client.open_session(threading.Event())

    (_, kwargs), *_ = events
    assert kwargs["run_embedded"] is True
    assert kwargs["accept_eula"] is True
    run_vendor = next(e[1] for e in events if e[0] == "run")
    [(_, vendor)] = run_vendor.tracker_vendors
    assert vendor.id == "body.pico-xr"

    client.close()
    assert events[-3:] == ["exit deviceio", "exit openxr", "launcher.stop"]


def test_xsens_plugin_runs_between_runtime_and_session(client_module, monkeypatch, fake_plugin):
    module, events = client_module
    monkeypatch.setenv("CLOUDXR_EULA_ACCEPTED", "yes")
    monkeypatch.delenv("XR_RUNTIME_JSON", raising=False)
    script, record = fake_plugin
    client = module.IsaacTeleopClient(
        body_vendor="xsens", xsens_plugin=script, xsens_address="192.168.1.5", xsens_port=9765
    )

    client.start_runtime()
    plugin = client._xsens_proc
    assert events[0][0] == "CloudXRLauncher"
    assert not any(e[0] == "run" for e in events if isinstance(e, tuple))

    client.open_session(threading.Event())

    args, runtime_json = record.read_text().splitlines()
    assert args.split() == [
        "--collection-id=xsens_full_body",
        "--address=192.168.1.5",
        "--port=9765",
        "--max-flatbuffer-size=4096",
    ]
    assert runtime_json == RUNTIME_ENV["XR_RUNTIME_JSON"]
    vendor_configs = [e[1] for e in events if isinstance(e, tuple) and e[0] != "CloudXRLauncher"]
    vendor_configs = [v for v in vendor_configs if hasattr(v, "tracker_vendors")]
    assert len(vendor_configs) == 2  # extensions and session both see the vendor
    for config in vendor_configs:
        [(_, vendor)] = config.tracker_vendors
        assert vendor.id == "body.xsens"
        assert vendor.params == {"collection_id": "xsens_full_body", "max_flatbuffer_size": "4096"}
    assert plugin.poll() is None

    client.close()
    assert plugin.poll() is not None
    assert events[-1] == "launcher.stop"


def test_a_failed_plugin_does_not_keep_the_session_closed(
    client_module, monkeypatch, tmp_path, capsys
):
    module, events = client_module
    monkeypatch.setenv("CLOUDXR_EULA_ACCEPTED", "yes")
    script = tmp_path / "broken_plugin"
    script.write_text("#!/bin/sh\necho 'bind to UDP failed' >&2\nexit 3\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    client = module.IsaacTeleopClient(
        body_vendor="xsens", xsens_plugin=script, xsens_address="127.0.0.1"
    )
    client.start_runtime()

    client.open_session(threading.Event())

    assert "exited with code 3" in capsys.readouterr().err
    assert client.has_session  # head, controllers and hands still work
    client.close()


def test_xsens_requires_plugin_and_address(client_module):
    module, _ = client_module
    with pytest.raises(ValueError):
        module.IsaacTeleopClient(body_vendor="xsens", xsens_address="127.0.0.1")
    with pytest.raises(ValueError):
        module.IsaacTeleopClient(body_vendor="xsens", xsens_plugin="/bin/true")


def test_plugin_that_died_after_it_was_ready_is_restarted_before_a_reopen(
    client_module, monkeypatch, fake_plugin
):
    module, events = client_module
    monkeypatch.setenv("CLOUDXR_EULA_ACCEPTED", "yes")
    script, _ = fake_plugin
    client = module.IsaacTeleopClient(
        body_vendor="xsens", xsens_plugin=script, xsens_address="192.168.1.5"
    )
    client.start_runtime()
    client.open_session(threading.Event())
    first = client._xsens_proc

    first.kill()
    first.wait()
    client.close_session()
    client.open_session(threading.Event())

    assert client._xsens_proc is not first
    assert client._xsens_proc.poll() is None
    assert client.has_session
    client.close()
    assert client._xsens_proc is None
