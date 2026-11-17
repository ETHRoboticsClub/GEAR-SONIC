"""Publish CLI against a fake Hugging Face client. No network."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import pytest

from gear_sonic.data.record.publish import push_dataset
from gear_sonic.data.record.session_log import SessionWriter
from gear_sonic.scripts.publish_episodes import main
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
    api = FakeHub()
    monkeypatch.setattr(
        "gear_sonic.scripts.publish_episodes.build_hub_api",
        lambda token: api,
    )
    argv = _argv(session, v3, groot, token_file)
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

    calls_after_push = len(api.calls)
    assert main(argv, controller_running=lambda: False) == 0
    again = capsys.readouterr()
    assert "nothing to publish" in again.out
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


def _write_session(root: Path, *, success: bool) -> None:
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
    finally:
        writer.close()
