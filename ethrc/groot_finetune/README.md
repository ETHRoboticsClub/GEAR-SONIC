# Fine-tune GR00T N1.7 on G1 recordings

From a task recorded on the G1 to a GR00T N1.7 model that drives it, in one command:

```bash
ethrc/groot_finetune/cloud/train.sh ETHRC/grasp_bottle_g1_gr00t ETHRC/gr00t-g1-grasp-bottle
```

The input is what the G1 recorder exports (`docs/g1_teleop_recording.md`, "Export"):
a pair of Hugging Face datasets per task, `<name>` (LeRobot v3.0, with each take's
success label) and `<name>_gr00t` (LeRobot v2.1 with `meta/modality.json` for GR00T's
`UNITREE_G1_SONIC` embodiment). Training reads the `_gr00t` one and takes the labels from
its twin. The output is a private model repository on the Hub.

## What it does

`train.py` runs four steps. Each one is safe to run again:

1. **check**: Hugging Face access (dataset, labels, NVIDIA's gated
   `nvidia/Cosmos-Reason2-2B`, the base model `nvidia/GR00T-N1.7-3B`, write access to
   the model repository), GPU memory, disk, and video decoding. It reports every problem
   at once, before downloading anything.
2. **prepare**: download the dataset and keep the episodes `--episodes` selects:
   - `not-failed` (default) drops the takes answered `n`;
   - `successful` keeps only the takes answered `y`;
   - `all` keeps every take.
   Episodes shorter than the 40-step action horizon are always dropped. It then computes
   GR00T's normalization statistics and samples training steps with GR00T's own loader.
   The selection is written to `ethrc_selection.json`.
3. **train**: GR00T's `gr00t/experiment/launch_finetune.py` with the defaults of
   Isaac-GR00T's `examples/finetune.sh`. It keeps the last two checkpoints. If the run
   directory holds checkpoints of the same run, it resumes from the latest.
4. **upload**: the final model, a model card and the selection go to `--model-repo`, as
   a private repository. To retry only the upload, add `--upload-only`.

GR00T itself is not in this repository. `setup_groot.sh` clones NVIDIA's
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) at the commit in
`ISAAC_GR00T_REVISION` (376ba89, 2026-08-10) and installs it with GR00T's own dGPU
script. The club's fork `ETHRC-Humanoid-Isaac-GR00T` has the same `gr00t` package at
that commit.

## Before the first run

You need three things, once:

1. **Hugging Face.** Your token must read the datasets (the `ETHRC` ones are private)
   and write the model repository. Sign in to huggingface.co and accept the terms of
   [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B): GR00T
   N1.7 is built on it and the check fails until you do. Then `hf auth login`, or
   `export HF_TOKEN=...`.
2. **SkyPilot**, for the cloud route: `uv tool install 'skypilot[aws]'`, then
   `sky api login -e https://sky.ethrobotics.ch`.
3. **A GPU budget.** It needs one GPU with at least 40 GB: an AWS L40S first, an H100 as
   the fallback (about three times the price). The machine stops itself 30 minutes after
   the job ends (`IDLE_MINUTES`); its disk is billed until `sky down`.

## Run it

Check the dataset first. This needs no GPU and runs anywhere Isaac-GR00T is installed:

```bash
cd ~/Isaac-GR00T && uv run python <GEAR-SONIC>/ethrc/groot_finetune/train.py \
    --dataset ETHRC/grasp_bottle_g1_gr00t --prepare-only
```

Then train:

- **On the club's cloud.** Only this folder is uploaded; the machine clones
  Isaac-GR00T itself.

  ```bash
  ethrc/groot_finetune/cloud/train.sh ETHRC/grasp_bottle_g1_gr00t ETHRC/gr00t-g1-grasp-bottle
  sky logs gr00t-grasp-bottle-g1-gr00t    # follow
  ```

  Running the same command again resumes the run. Extra `train.py` flags go after the two
  names, for example `--max-steps 20000 --episodes successful`.

- **On a GPU machine you have:**

  ```bash
  bash ethrc/groot_finetune/setup_groot.sh ~/Isaac-GR00T
  cd ~/Isaac-GR00T && uv run python <GEAR-SONIC>/ethrc/groot_finetune/train.py \
      --dataset ETHRC/grasp_bottle_g1_gr00t --model-repo ETHRC/gr00t-g1-grasp-bottle
  ```

Useful flags:

| Flag | Effect |
|---|---|
| `--labels <v3 dataset or path>` | Read the success labels elsewhere. `none` uses no labels. |
| `--max-steps` | Number of training steps. Default 10000. |
| `--batch-size` | Global batch size. Default 32. |
| `--learning-rate` | Default 1e-4. |
| `--wandb-project` | Log to W&B. Needs `WANDB_API_KEY`. |
| `--fresh` | Start over instead of resuming. |
| `--no-upload` | Keep the model on the machine. |

## Use the model

The model card on the Hub gives the commands. With Isaac-GR00T at the same revision:

```bash
hf download ETHRC/gr00t-g1-grasp-bottle --local-dir checkpoints/gr00t-g1-grasp-bottle
python gr00t/eval/run_gr00t_server.py --model-path checkpoints/gr00t-g1-grasp-bottle \
    --embodiment-tag UNITREE_G1_SONIC --device cuda:0 --host 0.0.0.0 --port 5550
```

Running it on the robot means sending its motion tokens to the SONIC decoder in place of
the PICO. That bridge is not part of this folder.

## Tests

```bash
python -m pytest ethrc/groot_finetune/tests -q    # CPU only; needs nothing but pytest
```

## What has been checked (2026-11-21)

- The CPU tests: 16 passed.
- `--prepare-only` with Isaac-GR00T at the pinned commit, in a CPU container. It ran on
  the real `ETHRC/grasp_bottle_g1_gr00t`, read from the Hub, with its labels from
  `ETHRC/grasp_bottle_g1`. It kept 4 episodes (1.7 min), dropped the failed take,
  computed the statistics, and sampled 5057 training steps. A sample held a 360x640
  image, 8 state groups and a 40x64 token chunk with the task text.
- **Not yet run:** training on a GPU, the cloud launch, and the upload.
- **Dataset size:** five takes test the pipeline. They are not enough data for a useful
  policy; plan on several dozen successful takes per task.
