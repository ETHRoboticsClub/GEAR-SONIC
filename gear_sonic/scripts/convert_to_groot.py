"""Cut invalid G1 frames and attach GR00T v2.1 modality metadata.

Usage::

    /home/unitree/.venvs/g1-record/bin/python gear_sonic/scripts/convert_to_groot.py \\
        <v3_root> <out_root>

The input dataset is copied to a temporary directory first. The caller's v3
root is never modified and never moved. GR00T's ``convert_v3_to_v2.py`` moves
the dataset directory it is given. Its ``--root`` argument is the parent of
the dataset: the script joins ``root / repo_id`` (``unitree/g1_pico_sonic``).
This tool points that script at a second copy of the segmented v3 dataset.

Frames are re-encoded with ``RGBEncoderConfig(vcodec="h264", g=1)``. Read-back
opens ``LeRobotDataset`` with ``video_backend="pyav"`` because torchcodec does
not load on this board. If that decoder fails, frames are read with the ``av``
package straight from the mp4. Library callers can instead pass
``source_frames`` to ``write_segmented_dataset`` and skip decoding.

There is no ``ffmpeg`` binary on PATH. The GR00T script needs that binary to
split videos. When ``shutil.which("ffmpeg")`` is None this command still writes
the segmented v3 dataset, leaves ``CONVERT_BLOCKED.txt`` explaining that
ffmpeg is required, and exits with an error. It does not claim the v2.1
conversion ran. ``ffprobe`` is absent with ``ffmpeg``; the result records that
the codec probe was skipped and Python validation does not fail for that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[2]
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def main(argv: list[str] | None = None) -> int:
    _bootstrap()
    from gear_sonic.data.record.convert.segment import convert_dataset
    from gear_sonic.data.record.schema import REPO_ID

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("v3_root", type=Path, help="LeRobot v3 dataset directory to read.")
    parser.add_argument("out_root", type=Path, help="Directory to receive the segmented dataset.")
    parser.add_argument(
        "--min-segment-s",
        type=float,
        default=1.0,
        help="Drop contiguous valid runs shorter than fps times this many seconds (default: 1.0).",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help="Repo id joined by convert_v3_to_v2.py onto --root (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    try:
        result = convert_dataset(
            args.v3_root,
            args.out_root,
            min_segment_s=args.min_segment_s,
            repo_id=args.repo_id,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2))
    if result.blocked:
        print(result.message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
