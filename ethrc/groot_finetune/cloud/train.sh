#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on a G1 recording on a cloud GPU through the club's SkyPilot server.
#
#   ethrc/groot_finetune/cloud/train.sh <dataset_gr00t> <model-repo> [train.py flags...]
#   ethrc/groot_finetune/cloud/train.sh ETHRC/grasp_bottle_g1_gr00t ETHRC/gr00t-g1-grasp-bottle --max-steps 20000
#
# Needs: the `sky` client logged in to the club server, and a Hugging Face token in HF_TOKEN
# or from `hf auth login` (read: the dataset, its v3 twin with the success labels, and
# nvidia/Cosmos-Reason2-2B; write: the model repository). WANDB_API_KEY plus
# --wandb-project log to W&B.
#
# Environment: SKY_CLUSTER names the cluster (default gr00t-<dataset name>); IDLE_MINUTES
# is how long the machine waits after the job before stopping itself (default 30).
# Running this again with the same cluster resumes where the previous run stopped.
set -euo pipefail
self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$self")/.."

if [ "$#" -lt 2 ]; then
  sed -n '2,15p' "$self" | sed 's/^# \{0,1\}//'
  exit 2
fi
DATASET="$1"
MODEL_REPO="$2"
shift 2

SKY_SERVER="${SKY_SERVER:-https://sky.ethrobotics.ch}"
command -v sky > /dev/null || {
  echo "the SkyPilot client is missing: uv tool install 'skypilot[aws]', then sky api login -e $SKY_SERVER" >&2
  exit 1
}
# Without a login, `sky launch` quietly starts a local API server on this computer instead
# of using the club's. Refuse that.
api="$(sky api info 2>&1 || true)"
case "$api" in
  *"${SKY_SERVER#https://}"*) ;;
  *)
    echo "sky is not connected to the club server $SKY_SERVER: run sky api login -e $SKY_SERVER" >&2
    exit 1
    ;;
esac
if [ -z "${HF_TOKEN:-}" ]; then
  HF_TOKEN="$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2> /dev/null || true)"
fi
[ -n "$HF_TOKEN" ] || {
  echo "no Hugging Face token: export HF_TOKEN=hf_... or run hf auth login" >&2
  exit 1
}
export HF_TOKEN

name="$(basename "$DATASET" | tr '[:upper:]_.' '[:lower:]--' | tr -cd 'a-z0-9-' | cut -c1-40)"
CLUSTER="${SKY_CLUSTER:-gr00t-$name}"
IDLE="${IDLE_MINUTES:-30}"

# WANDB_PROJECT is a train.py flag; pull it out so the task can pass it on.
WANDB_PROJECT=""
EXTRA=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-project=*) WANDB_PROJECT="${1#*=}"; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
secrets=(--secret HF_TOKEN)
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY
  secrets+=(--secret WANDB_API_KEY)
elif [ -n "$WANDB_PROJECT" ]; then
  echo "--wandb-project needs WANDB_API_KEY in the environment" >&2
  exit 1
fi

echo "Cluster $CLUSTER: $DATASET -> $MODEL_REPO"
sky launch -y -c "$CLUSTER" -i "$IDLE" cloud/g1_groot_train.sky.yaml "${secrets[@]}" \
  --env DATASET="$DATASET" --env MODEL_REPO="$MODEL_REPO" \
  --env WANDB_PROJECT="$WANDB_PROJECT" --env EXTRA_ARGS="${EXTRA[*]:-}" -d
# The club server may not apply -i from a newer client, so set it explicitly: the machine
# stops (disk kept) IDLE minutes after the job ends.
sky autostop -y -i "$IDLE" "$CLUSTER"

cat << EOF

Submitted. The machine stops by itself $IDLE minutes after the job ends; its disk (and the
run's checkpoints) stay until you delete it.
  follow:          sky logs $CLUSTER
  status:          sky status $CLUSTER
  resume or retry: run this same command again
  stop now:        sky stop -y $CLUSTER
  delete:          sky down -y $CLUSTER    (only once the model is on the Hub)
EOF
