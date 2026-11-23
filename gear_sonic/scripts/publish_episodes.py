"""Append new teleop episodes and publish the LeRobot and GR00T datasets.

Usage::

    /home/unitree/.venvs/g1-record/bin/python gear_sonic/scripts/publish_episodes.py \\
        [session_dir ...] \\
        [--raw-root /home/unitree/datasets/raw] \\
        [--v3-root /home/unitree/datasets/g1_pico_sonic] \\
        [--groot-root /home/unitree/datasets/g1_pico_sonic_gr00t] \\
        [--repo-id ETHRC/g1_pico_sonic] \\
        [--groot-repo-id ETHRC/g1_pico_sonic_gr00t] \\
        [--private] \\
        [--token-file ~/.config/g1-record/hf_token] \\
        [--no-push] \\
        [--fps 50] [--height 360] [--width 640] \\
        [--min-segment-s 1.0] \\
        [--allow-during-teleop]

With no session directories, every subdirectory of ``--raw-root`` is scanned.
Each new Enter-marked window is appended to the LeRobot v3 dataset, then the
new episodes are segmented into the GR00T v2.1 dataset. Unless ``--no-push``,
both trees are uploaded. After both uploads are marked, each raw session whose
Enter-marked windows are all in the v3 ``meta/sources.jsonl`` is deleted. A
skipped window keeps its session. Encoding is refused while ``g1_deploy_onnx``
is running.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[2]
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def build_hub_api(token: str):
    """Return a Hub client. Tests replace this with a fake."""
    from huggingface_hub import HfApi

    return HfApi(token=token)


def teleop_controller_running(proc: Path = Path("/proc")) -> bool:
    """Whether a live ``g1_deploy_onnx`` process is in ``/proc``.

    An exited controller stays in ``/proc`` as a zombie until its tmux pane is
    closed; it no longer runs, so it does not count.
    """
    if not proc.is_dir():
        return False
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "comm").read_text(encoding="utf-8", errors="replace")
            stat_text = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        name = text.split("\n", 1)[0].strip("\x00 ")
        state = stat_text.rsplit(")", 1)[-1].split()[:1]
        if name.startswith("g1_deploy_onnx") and state != ["Z"]:
            return True
    return False


def main(argv: list[str] | None = None, *, controller_running=None) -> int:
    _bootstrap()
    try:
        return _publish(argv, controller_running)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _publish(argv, controller_running) -> int:
    from gear_sonic.data.record.build import build_rows
    from gear_sonic.data.record.convert.append import append_v21
    from gear_sonic.data.record.convert.segment import convert_dataset
    from gear_sonic.data.record.publish import (
        append_source,
        mark_pushed,
        needs_push,
        push_dataset,
        push_is_current,
        read_token,
        unconverted_episodes,
        window_already_appended,
        write_dataset_card,
    )
    from gear_sonic.data.record.schema import REPO_ID, load_layout
    from gear_sonic.data.record.session_log import SessionReader
    from gear_sonic.data.record.writer import append_episode
    from gear_sonic.scripts.build_episodes import assemble_episode, episode_windows

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "sessions",
        nargs="*",
        type=Path,
        help="Raw session directories. Default: every subdirectory of --raw-root.",
    )
    parser.add_argument("--raw-root", type=Path, default=Path("/home/unitree/datasets/raw"))
    parser.add_argument("--v3-root", type=Path, default=Path("/home/unitree/datasets/g1_pico_sonic"))
    parser.add_argument("--groot-root", type=Path, default=Path("/home/unitree/datasets/g1_pico_sonic_gr00t"))
    parser.add_argument("--repo-id", default="ETHRC/g1_pico_sonic")
    parser.add_argument("--groot-repo-id", default="ETHRC/g1_pico_sonic_gr00t")
    parser.add_argument("--private", action="store_true", help="Create private repos. Default is public.")
    parser.add_argument("--token-file", type=Path, default=Path("~/.config/g1-record/hf_token"))
    parser.add_argument("--no-push", action="store_true", help="Write the local datasets and skip the Hub.")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--min-segment-s", type=float, default=1.0)
    parser.add_argument(
        "--allow-during-teleop",
        action="store_true",
        help="Encode even if g1_deploy_onnx is running.",
    )
    args = parser.parse_args(argv)
    check = teleop_controller_running if controller_running is None else controller_running
    if not args.allow_during_teleop and check():
        print(
            "error: g1_deploy_onnx is running; video encoding must wait until teleop stops "
            "(pass --allow-during-teleop to override)",
            file=sys.stderr,
        )
        return 1
    if args.fps <= 0 or args.height <= 0 or args.width <= 0:
        raise ValueError(
            f"fps, height, and width must be positive, got {args.fps}, {args.height}, {args.width}"
        )
    if args.min_segment_s < 0:
        raise ValueError(f"--min-segment-s must be non-negative, got {args.min_segment_s}")

    sessions = _session_dirs(args.sessions, args.raw_root)
    layout = load_layout()
    min_frames = max(1, int(round(float(args.fps) * float(args.min_segment_s))))
    added: list[dict] = []
    skipped: dict[tuple[str, int, int], str] = {}
    for session in sessions:
        info = _session_info(session)
        reader = SessionReader(session)
        windows = episode_windows(list(reader.iter_events()), end_ns=_last_stamp(reader))
        for window in windows:
            if window_already_appended(
                args.v3_root,
                session.name,
                window["start_ns"],
                window["stop_ns"],
            ):
                continue
            record = _append_window(
                session,
                window,
                info=info,
                v3_root=args.v3_root,
                fps=args.fps,
                height=args.height,
                width=args.width,
                min_frames=min_frames,
                min_segment_s=args.min_segment_s,
                layout=layout,
                assemble_episode=assemble_episode,
                build_rows=build_rows,
                append_episode=append_episode,
                append_source=append_source,
                skipped=skipped,
            )
            if record is None:
                continue
            added.append(record)

    # Covers episodes a run appended to v3 but failed to convert, not only this run's.
    pending = unconverted_episodes(args.v3_root, args.groot_root)
    if pending:
        with tempfile.TemporaryDirectory(prefix="g1-publish-") as raw:
            converted = Path(raw) / "groot_v21"
            result = convert_dataset(
                args.v3_root,
                converted,
                min_segment_s=args.min_segment_s,
                episodes=pending,
                repo_id=REPO_ID,
            )
            if result.blocked:
                print(result.message, file=sys.stderr)
                return 1
            append_v21(args.groot_root, result.out_root)
    targets = [
        ("v3 repo", args.v3_root, args.repo_id, "v3.0"),
        ("groot repo", args.groot_root, args.groot_repo_id, "v2.1"),
    ]
    for _label, root, repo_id, _tag in targets:
        if (root / "meta" / "info.json").is_file():
            write_dataset_card(root, title=repo_id)

    urls: list[tuple[str, str]] = []
    pushed_this_run = False
    if not args.no_push:
        # A push that failed on an earlier run is retried here even when nothing new was added.
        due = [target for target in targets if needs_push(target[1], target[2])]
        if due:
            token = read_token(args.token_file)
            api = build_hub_api(token)
            message = (
                f"Add {len(added)} G1 pico teleop episode(s)"
                if added
                else "Upload G1 pico teleop episodes from an earlier run"
            )
            for label, root, repo_id, tag in due:
                url = push_dataset(
                    api,
                    root,
                    repo_id,
                    private=args.private,
                    tag=tag,
                    commit_message=message,
                )
                mark_pushed(root, repo_id, url)
                urls.append((label, url))
            pushed_this_run = True
    _print_summary(added, args.v3_root, args.groot_root, urls, pushed=not args.no_push)
    both_marked = push_is_current(args.v3_root, args.repo_id) and push_is_current(
        args.groot_root, args.groot_repo_id
    )
    if pushed_this_run and both_marked:
        errors = _delete_consumed_sessions(
            sessions,
            v3_root=args.v3_root,
            raw_root=args.raw_root,
            groot_root=args.groot_root,
            skipped=skipped,
        )
        if errors:
            for message in errors:
                print(f"error: {message}", file=sys.stderr)
            return 1
    return 0


def _append_window(
    session,
    window,
    *,
    info,
    v3_root,
    fps,
    height,
    width,
    min_frames,
    min_segment_s,
    layout,
    assemble_episode,
    build_rows,
    append_episode,
    append_source,
    skipped,
):
    assembled = assemble_episode(
        session,
        window["start_ns"],
        window["stop_ns"],
        height=height,
        width=width,
    )
    rows = build_rows(
        assembled["state"],
        assembled["tracking"],
        assembled["images"],
        fps=fps,
        t0_ns=assembled["t0_ns"],
        layout=layout,
    )
    frames = assembled["frames_rgb"]
    state_empty = not assembled["state"]
    t0_ns = assembled["t0_ns"]
    del assembled
    try:
        valid_rows = sum(row.valid != 0 for row in rows)
        reason = _skip_reason(
            rows,
            valid_rows=valid_rows,
            state_empty=state_empty,
            min_frames=min_frames,
            min_segment_s=min_segment_s,
            fps=fps,
        )
        if reason is not None:
            print(
                f"skip {session.name} {window['start_ns']}-{window['stop_ns']}: {reason}",
                flush=True,
            )
            skipped[(str(session), int(window["start_ns"]), int(window["stop_ns"]))] = reason
            return None
        metadata = {
            "source_commit": info["source_commit"],
            "success": window["success"],
        }
        episode_index = append_episode(
            rows,
            frames,
            v3_root,
            fps=fps,
            task=info["task"],
            t0_ns=t0_ns,
            metadata=metadata,
            image_hw=(height, width),
        )
        record = {
            "episode_index": int(episode_index),
            "session_id": session.name,
            "start_ns": int(window["start_ns"]),
            "stop_ns": int(window["stop_ns"]),
            "success": window["success"],
            "task": info["task"],
            "source_commit": info["source_commit"],
            "rows": len(rows),
            "valid_rows": int(valid_rows),
        }
        append_source(v3_root, record)
        return record
    finally:
        del frames
        del rows


def _skip_reason(rows, *, valid_rows, state_empty, min_frames, min_segment_s, fps) -> str | None:
    if state_empty or not rows:
        if state_empty:
            return "no robot state in the window"
        return "no frames on the grid"
    if valid_rows < min_frames:
        return (
            f"{valid_rows} valid rows is shorter than one {min_segment_s:g}s segment "
            f"({min_frames} frames at {fps} fps)"
        )
    from gear_sonic.data.record.convert.segment import segment_invalid

    if not segment_invalid([row.valid != 0 for row in rows], min_segment_frames=min_frames):
        return f"no run of {min_frames} consecutive valid rows ({min_segment_s:g}s at {fps} fps)"
    return None


def _session_dirs(positional: list[Path], raw_root: Path) -> list[Path]:
    if positional:
        missing = [str(path) for path in positional if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"session directory not found: {', '.join(missing)}")
        return list(positional)
    if not raw_root.is_dir():
        return []
    return sorted(path for path in raw_root.iterdir() if path.is_dir())


def _session_info(session_dir: Path) -> dict:
    path = session_dir / "session.json"
    if not path.is_file():
        raise ValueError(f"{path} is missing task")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    task = data.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError(f"{path} is missing task")
    source_commit = data.get("source_commit", "")
    if not isinstance(source_commit, str):
        source_commit = ""
    return {"task": task, "source_commit": source_commit}


def _last_stamp(reader) -> int | None:
    stamps = [stamp for stamp, _recv, _payload in reader.iter_state()]
    stamps.extend(stamp for stamp, _recv, _payload in reader.iter_tracking())
    stamps.extend(stamp for stamp, _sequence, _blob, _error in reader.iter_jpeg())
    if not stamps:
        return None
    return max(stamps)


def _print_summary(added, v3_root, groot_root, urls, *, pushed: bool) -> None:
    if not added and not urls:
        print("nothing to publish")
    elif not added:
        print("no new episodes; uploaded episodes from an earlier run")
    else:
        total_rows = sum(item["rows"] for item in added)
        total_valid = sum(item["valid_rows"] for item in added)
        print(f"episodes added: {len(added)}")
        for item in added:
            print(
                f"  {item['session_id']} episode {item['episode_index']} "
                f"success={item['success']} rows={item['rows']} valid_rows={item['valid_rows']}"
            )
        print(f"rows={total_rows} valid_rows={total_valid}")
    print(f"v3 root: {v3_root}")
    print(f"groot root: {groot_root}")
    if urls:
        for label, url in urls:
            print(f"{label}: {url}")
    elif added and not pushed:
        print("push: skipped")


def _delete_consumed_sessions(sessions, *, v3_root, raw_root, groot_root, skipped) -> list[str]:
    """Delete raw sessions whose windows are all in the v3 ledger.

    A skipped window, or any Enter-marked window missing from ``sources.jsonl``,
    keeps that session. The raw root and the dataset roots are never removed.
    """
    from gear_sonic.data.record.publish import (
        delete_raw_session,
        session_fully_consumed,
        window_already_appended,
    )
    from gear_sonic.data.record.session_log import SessionReader
    from gear_sonic.scripts.build_episodes import episode_windows

    errors = []
    for session in sessions:
        reader = SessionReader(session)
        windows = episode_windows(list(reader.iter_events()), end_ns=_last_stamp(reader))
        if session_fully_consumed(v3_root, session.name, windows):
            try:
                delete_raw_session(session, raw_root, v3_root, groot_root)
            except Exception as exc:
                errors.append(f"push succeeded, but failed to delete raw session {session}: {exc}")
            else:
                print(f"deleted raw session {session}")
            continue
        for window in windows:
            start_ns = int(window["start_ns"])
            stop_ns = int(window["stop_ns"])
            if window_already_appended(v3_root, session.name, start_ns, stop_ns):
                continue
            reason = skipped.get((str(session), start_ns, stop_ns))
            if reason:
                detail = f"was skipped ({reason})"
            else:
                detail = "is not in sources.jsonl"
            print(f"keep raw session {session}: window {start_ns}-{stop_ns} {detail}")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
