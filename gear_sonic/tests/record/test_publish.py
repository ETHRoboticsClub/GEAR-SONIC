"""Publish CLI against a fake Hugging Face client. No network."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import pytest

from gear_sonic.data.record.publish import delete_raw_session, push_dataset
from gear_sonic.data.record.session_log import SessionWriter
from gear_sonic.scripts.publish_episodes import main, teleop_controller_running
from gear_sonic.scripts.run_pico_session_recorder import write_session_json
from gear_sonic.tests.record.test_end_to_end import _g1_payload, _jpeg, _tracking_payload

_TOKEN = "hf_dummy_DO_NOT_LEAK_9f3a"
_V3_REPO = "ETHRC/g1_pico_sonic"
_GROOT_REPO = "ETHRC/g1_pico_sonic_gr00t"


class FakeHub:
    def __init__(self, *, private_override: dict | None = None):
        self.calls = []
        self.private = {} if private_override is None else dict(private_override)
        self._commit = 0

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False, **kwargs):
        self.calls.append(
            (
                "create_repo",
                {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "private": private,
                    "exist_ok": exist_ok,
                },
            )
        )
        self.private.setdefault(repo_id, private)
        return f"https://huggingface.co/datasets/{repo_id}"

    def repo_info(self, repo_id, repo_type=None, **kwargs):
        self.calls.append(("repo_info", {"repo_id": repo_id, "repo_type": repo_type}))
        return SimpleNamespace(private=self.private[repo_id])

    def upload_folder(self, *, folder_path, repo_id, repo_type="dataset", commit_message=""):
        self.calls.append(
            (
                "upload_folder",
                {
                    "folder_path": str(folder_path),
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "commit_message": commit_message,
                },
            )
        )
        self._commit += 1
        return SimpleNamespace(oid=f"{self._commit:040x}")

    def delete_tag(self, repo_id, *, tag, repo_type=None):
        self.calls.append(("delete_tag", {"repo_id": repo_id, "tag": tag, "repo_type": repo_type}))

    def create_tag(self, repo_id, *, tag, revision=None, repo_type=None, **kwargs):
        self.calls.append(
            (
                "create_tag",
                {"repo_id": repo_id, "tag": tag, "revision": revision, "repo_type": repo_type},
            )
        )


def test_publish_then_second_run_is_idle(tmp_path, monkeypatch, capsys):
    session = tmp_path / "session_a"
    _write_session(session, success=True)
    token_file = tmp_path / "hf_token"
    _write_token(token_file)
    v3 = tmp_path / "v3"
    groot = tmp_path / "groot"
    raw = tmp_path / "raw"
    raw.mkdir()
    api = FakeHub()
    monkeypatch.setattr(
        "gear_sonic.scripts.publish_episodes.build_hub_api",
        lambda token: api,
    )
    argv = _argv(session, v3, groot, token_file) + ["--raw-root", str(raw)]
    assert main(argv, controller_running=lambda: False) == 0
    captured = capsys.readouterr()
    assert _TOKEN not in captured.out
    assert _TOKEN not in captured.err

    creates = [call[1] for call in api.calls if call[0] == "create_repo"]
    uploads = [call[1] for call in api.calls if call[0] == "upload_folder"]
    created_tags = [call[1] for call in api.calls if call[0] == "create_tag"]
    tags = [call["tag"] for call in created_tags]
    deleted = [call[1]["tag"] for call in api.calls if call[0] == "delete_tag"]
    assert [call["private"] for call in creates] == [False, False]
    assert {call["repo_id"] for call in creates} == {_V3_REPO, _GROOT_REPO}
    assert all(call["repo_type"] == "dataset" and call["exist_ok"] is True for call in creates)
    assert {Path(call["folder_path"]) for call in uploads} == {v3, groot}
    assert {call["repo_id"] for call in uploads} == {_V3_REPO, _GROOT_REPO}
    assert tags == ["v3.0", "v2.1"]
    assert deleted == ["v3.0", "v2.1"]
    assert [call["revision"] for call in created_tags] == [f"{index:040x}" for index in (1, 2)]
    assert "session_a" in captured.out
    assert "success=True" in captured.out
    assert (v3 / "README.md").is_file()
    assert (groot / "README.md").is_file()
    assert "quality.valid 0" in (v3 / "README.md").read_text(encoding="utf-8")
    sources = [
        json.loads(line)
        for line in (v3 / "meta" / "sources.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(sources) == 1
    assert sources[0]["session_id"] == "session_a"
    assert sources[0]["success"] is True
    assert not (groot / "meta" / "stats.json").exists()
    assert f"deleted raw session {session}" in captured.out
    assert not session.exists()
    assert raw.is_dir()
    assert v3.is_dir()
    assert groot.is_dir()

    calls_after_push = len(api.calls)
    _write_session(session, success=True)
    assert main(argv, controller_running=lambda: False) == 0
    again = capsys.readouterr()
    assert "nothing to publish" in again.out
    assert "deleted raw session" not in again.out
    assert session.is_dir()
    assert _TOKEN not in again.out
    assert _TOKEN not in again.err
    assert len(api.calls) == calls_after_push


def test_no_push_makes_no_api_calls(tmp_path, monkeypatch, capsys):
    session = tmp_path / "session_b"
    _write_session(session, success=False)
    token_file = tmp_path / "hf_token"
    _write_token(token_file)
    api = FakeHub()

    def factory(_token):
        api.calls.append(("factory",))
        return api

    monkeypatch.setattr("gear_sonic.scripts.publish_episodes.build_hub_api", factory)
    code = main(
        _argv(session, tmp_path / "v3", tmp_path / "groot", token_file) + ["--no-push"],
        controller_running=lambda: False,
    )
    captured = capsys.readouterr()
    assert code == 0
    assert api.calls == []
    assert _TOKEN not in captured.out
    assert _TOKEN not in captured.err
    assert "push: skipped" in captured.out
    assert "deleted raw session" not in captured.out
    assert session.is_dir()
    assert (tmp_path / "v3" / "meta" / "info.json").is_file()
    assert (tmp_path / "groot" / "meta" / "info.json").is_file()


def test_running_controller_refuses(tmp_path, monkeypatch, capsys):
    def factory(_token):
        raise AssertionError("API was constructed")

    monkeypatch.setattr("gear_sonic.scripts.publish_episodes.build_hub_api", factory)
    code = main([str(tmp_path / "unused")], controller_running=lambda: True)
    captured = capsys.readouterr()
    assert code == 1
    assert "g1_deploy_onnx" in captured.err
    assert _TOKEN not in captured.err


def test_exited_controller_does_not_block_publishing(tmp_path):
    def process(pid: int, comm: str, state: str) -> None:
        entry = tmp_path / str(pid)
        entry.mkdir()
        (entry / "comm").write_text(comm + "\n", encoding="utf-8")
        (entry / "stat").write_text(f"{pid} ({comm}) {state} 1 1 1\n", encoding="utf-8")

    process(10, "bash", "S")
    process(11, "g1_deploy_onnx_", "Z")
    assert teleop_controller_running(tmp_path) is False
    process(12, "g1_deploy_onnx_", "R")
    assert teleop_controller_running(tmp_path) is True


def test_other_visibility_raises_without_upload(tmp_path):
    api = FakeHub(private_override={_V3_REPO: True})
    with pytest.raises(RuntimeError, match="private"):
        push_dataset(
            api,
            tmp_path,
            _V3_REPO,
            private=False,
            tag="v3.0",
            commit_message="add episodes",
        )
    assert not any(call[0] == "upload_folder" for call in api.calls)
    assert any(call[0] == "create_repo" for call in api.calls)


def test_failed_push_is_retried_on_the_next_run(tmp_path, monkeypatch, capsys):
    session = tmp_path / "session_c"
    _write_session(session, success=True)
    token_file = tmp_path / "hf_token"
    _write_token(token_file)
    v3 = tmp_path / "v3"
    groot = tmp_path / "groot"

    class FailingGroot(FakeHub):
        def upload_folder(self, *, folder_path, repo_id, **kwargs):
            if repo_id == _GROOT_REPO:
                raise ConnectionError("network down")
            return super().upload_folder(folder_path=folder_path, repo_id=repo_id, **kwargs)

    failing = FailingGroot()
    monkeypatch.setattr("gear_sonic.scripts.publish_episodes.build_hub_api", lambda token: failing)
    assert main(_argv(session, v3, groot, token_file), controller_running=lambda: False) == 1
    failed = capsys.readouterr()
    assert session.is_dir()
    assert "deleted raw session" not in failed.out

    api = FakeHub()
    monkeypatch.setattr("gear_sonic.scripts.publish_episodes.build_hub_api", lambda token: api)
    assert main(_argv(session, v3, groot, token_file), controller_running=lambda: False) == 0
    out = capsys.readouterr().out
    uploads = [call[1]["repo_id"] for call in api.calls if call[0] == "upload_folder"]
    assert uploads == [_GROOT_REPO]
    assert "no new episodes" in out
    assert f"deleted raw session {session}" in out
    assert not session.exists()
    assert len((v3 / "meta" / "sources.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_failed_conversion_is_converted_on_the_next_run(tmp_path, monkeypatch, capsys):
    session = tmp_path / "session_d"
    _write_session(session, success=True)
    token_file = tmp_path / "hf_token"
    _write_token(token_file)
    v3 = tmp_path / "v3"
    groot = tmp_path / "groot"
    argv = _argv(session, v3, groot, token_file) + ["--no-push"]

    def fail(*args, **kwargs):
        raise RuntimeError("conversion interrupted")

    monkeypatch.setattr("gear_sonic.data.record.convert.segment.convert_dataset", fail)
    assert main(argv, controller_running=lambda: False) == 1
    assert (v3 / "meta" / "info.json").is_file()
    assert not (groot / "meta" / "info.json").exists()
    capsys.readouterr()

    monkeypatch.undo()
    assert main(argv, controller_running=lambda: False) == 0
    info = json.loads((groot / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 1
    assert session.is_dir()


def test_skipped_window_keeps_the_raw_session(tmp_path, monkeypatch, capsys):
    session = tmp_path / "session_skip"
    bounds = _write_session(session, success=True, skipped_window=True)
    token_file = tmp_path / "hf_token"
    _write_token(token_file)
    v3 = tmp_path / "v3"
    groot = tmp_path / "groot"
    api = FakeHub()
    monkeypatch.setattr("gear_sonic.scripts.publish_episodes.build_hub_api", lambda token: api)
    code = main(_argv(session, v3, groot, token_file), controller_running=lambda: False)
    captured = capsys.readouterr()
    assert code == 0
    assert _TOKEN not in captured.out
    assert _TOKEN not in captured.err
    assert session.is_dir()
    assert "deleted raw session" not in captured.out
    assert "no robot state in the window" in captured.out
    keep = (
        f"keep raw session {session}: "
        f"window {bounds['skipped_start_ns']}-{bounds['skipped_stop_ns']} "
        "was skipped (no robot state in the window)"
    )
    assert keep in captured.out
    sources = [
        json.loads(line)
        for line in (v3 / "meta" / "sources.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(sources) == 1
    assert sources[0]["start_ns"] == bounds["start_ns"]
    assert sources[0]["stop_ns"] == bounds["stop_ns"]
    assert {call[1]["repo_id"] for call in api.calls if call[0] == "upload_folder"} == {
        _V3_REPO,
        _GROOT_REPO,
    }


def test_delete_failure_after_push_exits_nonzero(tmp_path, monkeypatch, capsys):
    session = tmp_path / "session_busy"
    _write_session(session, success=True)
    token_file = tmp_path / "hf_token"
    _write_token(token_file)
    v3 = tmp_path / "v3"
    groot = tmp_path / "groot"

    def fail_delete(*_args, **_kwargs):
        raise OSError("device busy")

    monkeypatch.setattr("gear_sonic.data.record.publish.delete_raw_session", fail_delete)
    monkeypatch.setattr(
        "gear_sonic.scripts.publish_episodes.build_hub_api",
        lambda token: FakeHub(),
    )
    code = main(_argv(session, v3, groot, token_file), controller_running=lambda: False)
    captured = capsys.readouterr()
    assert code == 1
    assert session.is_dir()
    assert "v3 repo:" in captured.out
    assert "groot repo:" in captured.out
    assert "https://huggingface.co/datasets/" in captured.out
    assert "push succeeded, but failed to delete raw session" in captured.err
    assert "device busy" in captured.err
    assert _TOKEN not in captured.out
    assert _TOKEN not in captured.err


def test_delete_raw_session_keeps_outside_symlink_and_roots(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    v3 = tmp_path / "v3"
    groot = tmp_path / "groot"
    v3.mkdir()
    groot.mkdir()
    session = tmp_path / "passed" / "session_a"
    session.mkdir(parents=True)
    (session / "events.jsonl").write_text("{}\n", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    (external / "precious").write_text("keep", encoding="utf-8")
    (session / "linked_dir").symlink_to(external, target_is_directory=True)
    outside = tmp_path / "keep.txt"
    outside.write_text("safe", encoding="utf-8")
    (session / "linked_file").symlink_to(outside)

    delete_raw_session(session, raw, v3, groot)

    assert not session.exists()
    assert (external / "precious").read_text(encoding="utf-8") == "keep"
    assert outside.read_text(encoding="utf-8") == "safe"
    assert raw.is_dir()
    assert v3.is_dir()
    assert groot.is_dir()
    assert (tmp_path / "passed").is_dir()


def test_delete_raw_session_refuses_roots_and_symlinks(tmp_path):
    raw = tmp_path / "raw"
    session = raw / "session_a"
    session.mkdir(parents=True)
    v3 = tmp_path / "v3"
    groot = tmp_path / "groot"
    v3.mkdir()
    groot.mkdir()
    with pytest.raises(RuntimeError, match="refusing to delete"):
        delete_raw_session(raw, raw, v3, groot)
    with pytest.raises(RuntimeError, match="refusing to delete"):
        delete_raw_session(tmp_path, raw, v3, groot)
    link = tmp_path / "link"
    link.symlink_to(session, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        delete_raw_session(link, raw, v3, groot)
    assert session.is_dir()
    assert raw.is_dir()
    assert v3.is_dir()
    assert groot.is_dir()


def _argv(session: Path, v3: Path, groot: Path, token_file: Path) -> list[str]:
    return [
        str(session),
        "--v3-root",
        str(v3),
        "--groot-root",
        str(groot),
        "--repo-id",
        _V3_REPO,
        "--groot-repo-id",
        _GROOT_REPO,
        "--token-file",
        str(token_file),
        "--fps",
        "50",
        "--height",
        "16",
        "--width",
        "16",
        "--min-segment-s",
        "0.02",
    ]


def _write_token(path: Path) -> None:
    path.write_text(_TOKEN + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _write_session(root: Path, *, success: bool, skipped_window: bool = False) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    write_session_json(root, task="walk to the table", source_commit="test-commit", settings={"camera": ""})
    writer = SessionWriter(root, queue_size=256, disk_free_bytes=lambda: 10 * 1024**3)
    try:
        start = 10_000_000_000
        count = 8
        step = 20_000_000
        for index in range(count):
            stamp = start + index * step
            writer.append_state(stamp, stamp, _g1_payload(index))
            writer.append_tracking(stamp, stamp, _tracking_payload(index, stamp))
            writer.append_jpeg(stamp, index, _jpeg(40))
        stop = start + (count - 1) * step
        writer.mark_event("episode_start", start)
        writer.mark_event("episode_stop", stop, {"success": success})
        skipped_start = None
        skipped_stop = None
        if skipped_window:
            skipped_start = stop + 5_000_000_000
            skipped_stop = skipped_start + 1_000_000_000
            writer.mark_event("episode_start", skipped_start)
            writer.mark_event("episode_stop", skipped_stop, {"success": False})
        return {
            "start_ns": start,
            "stop_ns": stop,
            "skipped_start_ns": skipped_start,
            "skipped_stop_ns": skipped_stop,
        }
    finally:
        writer.close()
