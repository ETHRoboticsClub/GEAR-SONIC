"""One command from a G1 recording on the Hugging Face Hub to a fine-tuned GR00T N1.7.

    python ethrc/groot_finetune/train.py --dataset ETHRC/grasp_bottle_g1_gr00t \\
        --model-repo ETHRC/gr00t-g1-grasp-bottle

Run it with the Python of an Isaac-GR00T install at the pinned revision
(``setup_groot.sh``; on the cloud, ``cloud/train.sh`` does everything). Steps, each safe
to run again:

1. check: Hugging Face access (dataset, NVIDIA's gated Cosmos-Reason2-2B backbone, the base
   model, write access to the model repository), GPUs, disk, video decoding, W&B. Every
   problem is reported at once, before anything is downloaded.
2. prepare: download the ``_gr00t`` dataset the G1 recorder published, keep the episodes
   ``--episodes`` selects (success labels come from the matching v3 dataset, ``--labels``),
   compute GR00T's normalization statistics, and load a shard of training steps with
   GR00T's own sampler.
3. train: run GR00T's fine-tuning (``gr00t/experiment/launch_finetune.py``). If the run
   directory already holds checkpoints of this same run, training resumes from the latest.
4. upload: push the final model to the model repository (private), with a model card,
   the episode selection and the run settings. ``--upload-only`` retries just this step.

``--prepare-only`` stops after step 2 and needs no GPU: use it to check a dataset.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D  # noqa: E402

HERE = Path(__file__).resolve().parent
ISAAC_GR00T_REPO = "https://github.com/NVIDIA/Isaac-GR00T"
ISAAC_GR00T_REVISION = (HERE / "ISAAC_GR00T_REVISION").read_text(encoding="utf-8").strip()
BASE_MODEL = "nvidia/GR00T-N1.7-3B"
BACKBONE = "nvidia/Cosmos-Reason2-2B"  # gr00t/experiment/launch_finetune.py: model_name
EMBODIMENT = "UNITREE_G1_SONIC"
MIN_GPU_GB = 40  # Isaac-GR00T getting_started/hardware_recommendation.md: fine-tuning minimum
MIN_FREE_DISK_GB = 80
# Optimizer and RNG states only matter to resume training on the same disk.
UPLOAD_IGNORE = [
    "checkpoint-*",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state*.pth",
    "global_step*",
    "*/optimizer.pt",
    "*/scheduler.pt",
    "*/rng_state*.pth",
    "train.log",
]


class Failure(RuntimeError):
    pass


def say(message: str) -> None:
    print(message, flush=True)


def groot_dir(explicit: str | None) -> Path:
    """The Isaac-GR00T checkout whose ``gr00t`` package this Python imports."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        try:
            import gr00t
        except ImportError:
            raise Failure(
                "cannot import gr00t: run this with the Python of an Isaac-GR00T install "
                "(bash ethrc/groot_finetune/setup_groot.sh, then `uv run --project <that dir> "
                "python ethrc/groot_finetune/train.py ...`)"
            ) from None
        path = Path(gr00t.__file__).resolve().parents[1]
    if not (path / "gr00t" / "experiment" / "launch_finetune.py").is_file():
        raise Failure(f"{path} is not an Isaac-GR00T checkout (no gr00t/experiment/launch_finetune.py)")
    return path


def groot_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class RunSettings:
    """What makes two runs the same run (and a checkpoint resumable)."""

    dataset: str
    dataset_revision: str | None
    labels: str | None
    episodes: str
    base_model: str
    groot_revision: str | None
    max_steps: int
    batch_size: int
    learning_rate: float
    save_steps: int
    num_gpus: int


def run_dir(args: argparse.Namespace) -> Path:
    if args.workdir:
        return Path(args.workdir).expanduser().resolve()
    return (Path.cwd() / "runs" / Path(args.dataset.rstrip("/")).name).resolve()


def labels_dataset(args: argparse.Namespace) -> str | None:
    if args.labels == "none":
        return None
    return args.labels or D.default_labels_dataset(args.dataset.rstrip("/"))


# ---------------------------------------------------------------------------- checks


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True


def _hf_checks(args: argparse.Namespace, training: bool) -> list[Check]:
    from huggingface_hub import HfApi, get_token, hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    checks: list[Check] = []
    api = HfApi()
    remote = [name for name in (args.dataset, labels_dataset(args)) if name and not Path(name).expanduser().exists()]
    if get_token() is None and not remote and not training:
        return [Check("Hugging Face token", True, "not needed for local datasets")]
    if get_token() is None:
        return [
            Check(
                "Hugging Face token",
                False,
                "no token: export HF_TOKEN=hf_... or run `hf auth login` (needs read access to "
                "the dataset and Cosmos-Reason2-2B, and write access to --model-repo)",
            )
        ]
    try:
        user = api.whoami()["name"]
        checks.append(Check("Hugging Face token", True, f"logged in as {user}"))
    except HfHubHTTPError as error:
        return [Check("Hugging Face token", False, f"the token is refused: {error}")]

    for name in remote:
        what = "dataset" if name == args.dataset else "success labels"
        try:
            info = api.dataset_info(name, revision=args.revision if name == args.dataset else None)
            checks.append(Check(what, True, f"{name} @ {info.sha[:12]}"))
        except RepositoryNotFoundError:
            checks.append(
                Check(
                    what,
                    False,
                    f"{user} cannot read dataset {name}: check the name, or give the token read "
                    "access (a private dataset is invisible otherwise)"
                    + ("; --labels none trains on every episode" if what == "success labels" else ""),
                )
            )
        except HfHubHTTPError as error:
            checks.append(Check(what, False, f"{name}: {error}"))
    if not training:
        return checks

    try:
        hf_hub_download(BACKBONE, "config.json")
        checks.append(Check("Cosmos-Reason2-2B access", True, "license accepted"))
    except GatedRepoError:
        checks.append(
            Check(
                "Cosmos-Reason2-2B access",
                False,
                f"GR00T N1.7 builds on NVIDIA's gated {BACKBONE}. Sign in to huggingface.co "
                f"as {user}, open https://huggingface.co/{BACKBONE} and accept its terms "
                "(approval is usually quick), then run again.",
            )
        )
    except HfHubHTTPError as error:
        checks.append(Check("Cosmos-Reason2-2B access", False, str(error)))

    try:
        api.model_info(args.base_model)
        checks.append(Check("base model", True, args.base_model))
    except HfHubHTTPError as error:
        checks.append(Check("base model", False, f"{args.base_model}: {error}"))

    if args.model_repo and not args.no_upload:
        try:
            api.create_repo(args.model_repo, repo_type="model", private=True, exist_ok=True)
            checks.append(Check("model repository", True, f"{args.model_repo} (private) writable"))
        except HfHubHTTPError as error:
            checks.append(
                Check(
                    "model repository",
                    False,
                    f"{user} cannot create or write {args.model_repo}: {error}. A fine-grained "
                    "token needs write access to that namespace.",
                )
            )
    return checks


def _gpu_checks(args: argparse.Namespace) -> list[Check]:
    try:
        import torch
    except ImportError:
        return [Check("GPU", False, "PyTorch is not installed: run setup_groot.sh")]
    if not torch.cuda.is_available():
        return [Check("GPU", False, f"no CUDA GPU visible; training needs one ({MIN_GPU_GB} GB or more)")]
    count = torch.cuda.device_count()
    checks = []
    names = []
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1e9
        names.append(f"{props.name} {gb:.0f} GB")
        if gb < MIN_GPU_GB:
            checks.append(
                Check(
                    f"GPU {i} memory",
                    False,
                    f"{props.name} has {gb:.0f} GB; fine-tuning needs about {MIN_GPU_GB} GB. "
                    "Lower --batch-size if you try anyway.",
                    fatal=False,
                )
            )
    checks.insert(0, Check("GPU", True, f"{count}x " + ", ".join(sorted(set(names)))))
    if args.num_gpus and args.num_gpus > count:
        checks.append(Check("GPU count", False, f"--num-gpus {args.num_gpus} but {count} visible"))
    try:
        import flash_attn  # noqa: F401

        checks.append(Check("flash-attn", True, "importable"))
    except ImportError as error:
        checks.append(Check("flash-attn", False, f"cannot import flash_attn: {error}"))
    return checks


def video_decoding_error() -> str | None:
    """Why GR00T cannot decode videos here, or None when it can."""
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception as error:  # ImportError, OSError or RuntimeError from its native loader
        return (
            f"{str(error).splitlines()[0]}. torchcodec runs on Linux with FFmpeg 4-7 "
            "(Ubuntu: sudo apt-get install -y ffmpeg)"
        )
    return None


def _video_check(training: bool) -> Check:
    error = video_decoding_error()
    if error is None:
        return Check("video decoding (torchcodec)", True, "loads")
    if not training:
        error += "; the training-step check is skipped"
    return Check("video decoding (torchcodec)", False, error, fatal=training)


def run_checks(args: argparse.Namespace, workdir: Path, groot: Path, need_gpu: bool) -> None:
    checks: list[Check] = []
    revision = groot_revision(groot)
    checks.append(
        Check(
            "Isaac-GR00T",
            True,
            f"{groot} @ {revision[:12] if revision else 'unknown revision'}"
            + ("" if revision in (None, ISAAC_GR00T_REVISION) else f" (pinned: {ISAAC_GR00T_REVISION[:12]})"),
            fatal=False,
        )
    )
    if revision not in (None, ISAAC_GR00T_REVISION):
        checks[-1].ok = False
    checks += _hf_checks(args, training=need_gpu)
    checks.append(_video_check(training=need_gpu))
    workdir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(workdir).free / 1e9
    checks.append(
        Check(
            "disk",
            free_gb >= MIN_FREE_DISK_GB,
            f"{free_gb:.0f} GB free in {workdir}"
            + ("" if free_gb >= MIN_FREE_DISK_GB else f"; plan on {MIN_FREE_DISK_GB} GB or more"),
            fatal=need_gpu,
        )
    )
    if need_gpu:
        checks += _gpu_checks(args)
    if args.wandb_project:
        netrc = Path("~/.netrc").expanduser()
        has_key = bool(os.environ.get("WANDB_API_KEY")) or (
            netrc.exists() and "api.wandb.ai" in netrc.read_text()
        )
        checks.append(
            Check("W&B", has_key, "key present" if has_key else "set WANDB_API_KEY or `wandb login`")
        )

    say("\nChecks")
    width = max(len(c.name) for c in checks)
    for c in checks:
        mark = "ok  " if c.ok else ("FAIL" if c.fatal else "warn")
        say(f"  [{mark}] {c.name.ljust(width)}  {c.detail}")
    failed = [c for c in checks if not c.ok and c.fatal]
    if failed:
        raise Failure(f"{len(failed)} check(s) failed; fix them and run the same command again")


# ---------------------------------------------------------------------------- prepare


def prepare(args: argparse.Namespace, workdir: Path) -> tuple[Path, dict]:
    say(f"\nReading {args.dataset}...")
    source = D.resolve(args.dataset, args.revision)
    labels_name = labels_dataset(args)
    labels = None
    if labels_name:
        say(f"Reading success labels from {labels_name}...")
        labels = D.read_sources(D.fetch_sources(labels_name))
    elif args.episodes != "all":
        say("No success labels (--labels none, or the dataset name does not end in _gr00t): "
            "every episode counts as unlabeled.")  # fmt: skip
    data_dir = workdir / "data"
    report = D.select_episodes(
        source, data_dir, policy=args.episodes, labels=labels, labels_from=labels_name
    )
    minutes = report["total_frames"] / report["fps"] / 60
    say(f"Episodes: {len(report['kept'])} kept ({minutes:.1f} min), {len(report['dropped'])} dropped")
    for row in report["dropped"]:
        say(f"  dropped episode {row['episode']}: {row['reason']}")

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.stats import main as stats_main

    say("Computing normalization statistics...")
    stats_main(data_dir, EmbodimentTag[EMBODIMENT])
    if video_decoding_error() is None:
        check_training_data(data_dir)
    else:
        say("Training-step check skipped: GR00T cannot decode videos on this machine.")
    return data_dir, report


def check_training_data(data_dir: Path) -> None:
    """Sample training steps exactly as fine-tuning will (GR00T's sharded sampler and
    modality config), without the model's processor."""
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    import numpy as np

    tag = EmbodimentTag[EMBODIMENT]
    config = MODALITY_CONFIGS[tag.value]
    horizon = max(config["action"].delta_indices) + 1
    if horizon != D.ACTION_HORIZON:
        raise Failure(
            f"GR00T's {tag.value} config predicts {horizon} steps, dataset.py assumes "
            f"{D.ACTION_HORIZON}: update ethrc/groot_finetune/dataset.py"
        )
    dataset = ShardedSingleStepDataset(
        data_dir, tag, config, shard_size=256, episode_sampling_rate=1.0
    )
    steps = sum(dataset.shard_lengths)
    if steps == 0:
        raise Failure("the selected episodes give no training step")
    dataset.processor = lambda messages: messages[0]["content"]
    sample = dataset.get_shard(0)[0]
    token = sample.actions["motion_token"]
    if token.shape != (D.ACTION_HORIZON, D.TOKEN_DIM) or not np.isfinite(token).all():
        raise Failure(f"a sampled action chunk has shape {token.shape} or non-finite values")
    image = sample.images["ego_view"][0]
    say(
        f"Training data OK: {steps} training steps in {len(dataset)} shards; a sample holds "
        f"a {np.asarray(image).shape} image, {len(sample.states)} state groups and a "
        f"{token.shape} token chunk for {sample.text!r}"
    )


# ---------------------------------------------------------------------------- train


def settings(args: argparse.Namespace, report: dict, groot: Path, num_gpus: int) -> RunSettings:
    return RunSettings(
        dataset=args.dataset,
        dataset_revision=report["source"]["revision"],
        labels=report["labels_from"],
        episodes=args.episodes,
        base_model=args.base_model,
        groot_revision=groot_revision(groot),
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        num_gpus=num_gpus,
    )


def checkpoints(output: Path) -> list[Path]:
    found = [p for p in output.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()]
    return sorted(found, key=lambda p: int(p.name.split("-")[-1]))


def is_finished(output: Path, run: RunSettings) -> bool:
    state = output / "ethrc_run.json"
    if not state.is_file():
        return False
    saved = json.loads(state.read_text())
    has_model = (output / "model.safetensors").exists() or any(output.glob("model-*.safetensors"))
    return has_model and saved.get("settings") == asdict(run) and saved.get("finished") is True


def finetune_command(
    args: argparse.Namespace, data_dir: Path, output: Path, num_gpus: int, resume: bool
) -> list[str]:
    """``gr00t/experiment/launch_finetune.py`` and its flags, with examples/finetune.sh's
    defaults (learning-rate schedule, sampling, color jitter) and two checkpoints kept."""
    cmd = [
        "gr00t/experiment/launch_finetune.py",
        "--base_model_path", args.base_model,
        "--dataset_path", str(data_dir),
        "--embodiment_tag", EMBODIMENT,
        "--num_gpus", str(num_gpus),
        "--output_dir", str(output),
        "--save_steps", str(args.save_steps),
        "--save_total_limit", "2",
        "--max_steps", str(args.max_steps),
        "--warmup_ratio", "0.05",
        "--weight_decay", "1e-5",
        "--learning_rate", str(args.learning_rate),
        "--global_batch_size", str(args.batch_size),
        "--dataloader_num_workers", str(args.dataloader_workers),
        "--shard_size", "1024",
        "--num_shards_per_epoch", "100000",
        "--episode_sampling_rate", "0.1",
        "--color_jitter_params", "brightness", "0.3", "contrast", "0.4", "saturation", "0.5",
        "hue", "0.08",
    ]  # fmt: skip
    if args.wandb_project:
        cmd += ["--use_wandb", "--wandb_project", args.wandb_project]
    if resume:
        cmd += ["--resume_from_checkpoint"]
    return cmd


def train(
    args: argparse.Namespace, workdir: Path, data_dir: Path, run: RunSettings, groot: Path
) -> Path:
    output = workdir / "output"
    state_file = output / "ethrc_run.json"
    if is_finished(output, run):
        say(f"This run already finished in {output}; skipping training.")
        return output
    resume = False
    if output.exists() and (checkpoints(output) or any(output.iterdir())):
        previous = (
            json.loads(state_file.read_text()).get("settings") if state_file.is_file() else None
        )
        if previous == asdict(run) and checkpoints(output) and not args.fresh:
            resume = True
            say(f"Resuming from {checkpoints(output)[-1].name}.")
        elif args.fresh or previous != asdict(run):
            stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            moved = workdir / f"output.previous-{stamp}"
            say(f"{output} holds another run (different settings or --fresh); moving it to {moved.name}.")
            output.rename(moved)
    output.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"settings": asdict(run), "finished": False}, indent=2))

    cmd = finetune_command(args, data_dir, output, run.num_gpus, resume)
    env = dict(os.environ)
    if run.num_gpus == 1:
        # As examples/finetune.sh: one visible GPU, or HF Trainer wraps the model in
        # DataParallel and fails.
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        full = [sys.executable, *cmd]
    else:
        full = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={run.num_gpus}",
                f"--master_port={env.get('MASTER_PORT', '29500')}", *cmd]  # fmt: skip
    say(f"\nTraining: {' '.join(full)}\nLog: {output / 'train.log'}\n")
    with open(output / "train.log", "a", encoding="utf-8") as log:
        process = subprocess.Popen(
            full, cwd=groot, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )  # fmt: skip
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise Failure(
            f"training exited with code {code} (log: {output / 'train.log'}). Running the same "
            "command again resumes from the latest checkpoint."
        )
    state_file.write_text(json.dumps({"settings": asdict(run), "finished": True}, indent=2))
    return output


# ---------------------------------------------------------------------------- upload


def model_card(model_repo: str, run: RunSettings, report: dict) -> str:
    kept, dropped = report["kept"], report["dropped"]
    minutes = report["total_frames"] / report["fps"] / 60
    successes = sum(1 for row in kept if row["success"] is True)
    revision = run.groot_revision or ISAAC_GR00T_REVISION
    local = model_repo.split("/")[-1]
    tasks = ", ".join(f"`{task}`" for task in report["tasks"]) or "none"
    return f"""---
base_model: {run.base_model}
tags: [gr00t, unitree-g1, sonic, robotics, ethrc]
---

# GR00T N1.7 fine-tuned on `{run.dataset}`

GR00T N1.7 ({run.base_model}) fine-tuned for the Unitree G1 with the SONIC whole-body
controller (embodiment `{EMBODIMENT}`) by the ETH Robotics Club, from teleoperation
recorded on the robot (GEAR-SONIC `docs/g1_teleop_recording.md`).

| | |
|---|---|
| Dataset | `{run.dataset}` @ `{run.dataset_revision or "local"}` |
| Episodes | {len(kept)} used ({minutes:.1f} min, {successes} marked successful), {len(dropped)} dropped (`--episodes {run.episodes}`; see `ethrc_selection.json`) |
| Tasks | {tasks} |
| Steps | {run.max_steps}, global batch {run.batch_size}, learning rate {run.learning_rate}, {run.num_gpus} GPU(s) |
| Isaac-GR00T | `{revision}` |
| Trained | {dt.date.today().isoformat()} |

Inputs: head camera (`ego_view`), 43 joint positions in NVIDIA's G1 order, projected
gravity, task text. Outputs: 40 future SONIC motion tokens (64 values each) and 7
finger targets per Dex3 hand.

Serve it with NVIDIA's Isaac-GR00T at the same revision:

```bash
git clone {ISAAC_GR00T_REPO} && cd Isaac-GR00T && git checkout {revision}
hf download {model_repo} --local-dir checkpoints/{local}
python gr00t/eval/run_gr00t_server.py --model-path checkpoints/{local} \\
    --embodiment-tag {EMBODIMENT} --device cuda:0 --host 0.0.0.0 --port 5550
```
"""


def upload(args: argparse.Namespace, workdir: Path) -> None:
    from huggingface_hub import HfApi

    output = workdir / "output"
    state_file = output / "ethrc_run.json"
    if not state_file.is_file() or not json.loads(state_file.read_text()).get("finished"):
        raise Failure(f"no finished training in {output}: nothing to upload")
    run = RunSettings(**json.loads(state_file.read_text())["settings"])
    selection = workdir / "data" / "meta" / D.REPORT
    report = json.loads(selection.read_text())
    (output / "README.md").write_text(model_card(args.model_repo, run, report))
    shutil.copy2(selection, output / D.REPORT)
    api = HfApi()
    api.create_repo(args.model_repo, repo_type="model", private=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            say(f"Uploading {output} to {args.model_repo} (attempt {attempt})...")
            commit = api.upload_folder(
                repo_id=args.model_repo,
                folder_path=output,
                ignore_patterns=UPLOAD_IGNORE,
                commit_message=f"GR00T N1.7 fine-tuned on {run.dataset} ({run.max_steps} steps)",
            )
            say(f"Uploaded: {commit.commit_url}")
            return
        except Exception as error:  # network errors and Hub hiccups: retry, then explain
            if attempt == 3:
                raise Failure(
                    f"upload failed: {error}. The model is safe in {output}; retry with the same "
                    "command plus --upload-only."
                ) from error
            say(f"  upload failed ({error}); retrying in {30 * attempt} s")
            time.sleep(30 * attempt)


# ---------------------------------------------------------------------------- main


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True,
                   help="the recorder's GR00T dataset: Hub id (namespace/name_gr00t) or local path")  # fmt: skip
    p.add_argument("--model-repo", help="Hub model repository for the result (namespace/name)")
    p.add_argument("--revision", help="dataset branch, tag or commit (default: main)")
    p.add_argument("--workdir", help="run directory (default: ./runs/<dataset name>)")
    p.add_argument("--groot-dir", help="Isaac-GR00T checkout (default: where gr00t is imported from)")
    g = p.add_argument_group("data")
    g.add_argument(
        "--labels",
        help="v3 dataset holding meta/sources.jsonl with the success labels: Hub id or local "
        "path (default: --dataset without its _gr00t suffix); 'none' to use no labels",
    )
    g.add_argument("--episodes", choices=D.EPISODE_POLICIES, default="not-failed",
                   help="not-failed (default): drop episodes marked failed; successful: keep "
                   "only those marked successful; all: keep every episode")  # fmt: skip
    g = p.add_argument_group("training")
    g.add_argument("--base-model", default=BASE_MODEL)
    g.add_argument("--max-steps", type=int, default=10000)
    g.add_argument("--batch-size", type=int, default=32, help="global batch size")
    g.add_argument("--learning-rate", type=float, default=1e-4)
    g.add_argument("--save-steps", type=int, default=1000)
    g.add_argument("--num-gpus", type=int, help="default: every visible GPU")
    g.add_argument("--dataloader-workers", type=int, default=4)
    g.add_argument("--wandb-project", help="log to this W&B project (needs WANDB_API_KEY)")
    g.add_argument("--fresh", action="store_true", help="start over instead of resuming")
    g = p.add_argument_group("modes")
    g.add_argument("--prepare-only", action="store_true", help="checks and data only; no GPU")
    g.add_argument("--no-upload", action="store_true", help="keep the model on this machine")
    g.add_argument("--upload-only", action="store_true", help="only upload a finished run")
    args = p.parse_args(argv)
    if not args.prepare_only and not args.no_upload and not args.model_repo:
        p.error("--model-repo is required to upload the result (or pass --no-upload)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    workdir = run_dir(args)
    try:
        if args.upload_only:
            upload(args, workdir)
            return 0
        groot = groot_dir(args.groot_dir)
        run_checks(args, workdir, groot, need_gpu=not args.prepare_only)
        data_dir, report = prepare(args, workdir)
        if args.prepare_only:
            say(f"\nPrepared {data_dir}. Run without --prepare-only on a GPU machine to train.")
            return 0
        import torch

        run = settings(args, report, groot, args.num_gpus or torch.cuda.device_count())
        train(args, workdir, data_dir, run, groot)
        if args.no_upload:
            say(f"\nDone. The model is in {workdir / 'output'}.")
            return 0
        upload(args, workdir)
        say("\nDone.")
        return 0
    except (Failure, D.DatasetError) as error:
        say(f"\nerror: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
