#!/usr/bin/env bash
# Serve a fine-tuned GR00T model from a club DGX Spark through the club's SkyPilot server.
#
#   ethrc/groot_finetune/cloud/serve.sh <model-repo>       e.g. antonin-leroy/gr00t-g1-grasp-bottle-real
#
# Needs the `sky` client logged in to the club server and a Hugging Face token (HF_TOKEN or
# `hf auth login`) that reads the model. The server listens inside the cluster only; reach
# it through SkyPilot's SSH, which forwards the port to this computer:
#
#   ssh -N -L 0.0.0.0:5550:localhost:5550 gr00t-serve
#
# and point the robot at this computer (GR00T_HOST=<its address> in run_vla_real.sh).
# Environment: SKY_CLUSTER (default gr00t-serve). Stop it when done: sky down -y gr00t-serve
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
[ "$#" -eq 1 ] || { sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
SKY_SERVER="${SKY_SERVER:-https://sky.ethrobotics.ch}"
case "$(sky api info 2>&1 || true)" in
  *"${SKY_SERVER#https://}"*) ;;
  *) echo "sky is not connected to the club server: sky api login -e $SKY_SERVER" >&2; exit 1 ;;
esac
if [ -z "${HF_TOKEN:-}" ]; then
  HF_TOKEN="$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2> /dev/null || true)"
fi
[ -n "$HF_TOKEN" ] || { echo "no Hugging Face token: export HF_TOKEN=hf_... or run hf auth login" >&2; exit 1; }
export HF_TOKEN
CLUSTER="${SKY_CLUSTER:-gr00t-serve}"
cd "$root"
sky launch -y -d -c "$CLUSTER" "$here/g1_groot_serve.sky.yaml" --secret HF_TOKEN \
  --env MODEL_REPO="$1" --env GROOT_REVISION="$(tr -d '[:space:]' < "$here/../ISAAC_GR00T_REVISION")"
cat << MSG

Submitted. The first start installs GR00T and downloads the model (about 20 min); the
server is up when \`sky logs $CLUSTER\` shows "Server is ready". Then, on this computer:
  ssh -N -L 0.0.0.0:5550:localhost:5550 $CLUSTER
MSG
