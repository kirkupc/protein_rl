#!/usr/bin/env bash
# setup.sh — Create a GCP deep-learning VM for protein binder experiments
#
# Machine : n1-standard-8  (8 vCPU, 30 GB RAM)
# GPU     : 1× NVIDIA T4   (16 GB VRAM, sm_75)
# Image   : pytorch-latest-gpu (Deep Learning VM — CUDA + conda pre-installed)
# Disk    : 300 GB SSD
# Cost    : ~$0.60/hr on-demand (stop VM when not in use)
#
# Usage:
#   bash setup.sh
#   GCP_PROJECT=my-project ZONE=us-west1-b bash setup.sh
#
# After setup:
#   ssh binder-vm              — connect
#   ssh binder-vm nvidia-smi  — verify GPU

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-us-central1-a}"
REGION="${ZONE%-*}"            # strip AZ suffix: us-central1-a → us-central1

INSTANCE="binder-vm"
MACHINE_TYPE="n1-standard-8"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="300GB"
DISK_TYPE="pd-ssd"

STATIC_IP_NAME="${INSTANCE}-ip"
SSH_CONFIG="${HOME}/.ssh/config"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ──────────────────────────────────────────────────────────────────────────────

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33mWARN: $*\033[0m"; }
die()  { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

# ── Preflight ─────────────────────────────────────────────────────────────────
command -v gcloud >/dev/null 2>&1 || die "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
[[ -z "$PROJECT" ]] && die "GCP project not set. Export GCP_PROJECT or: gcloud config set project <PROJECT>"

log "Configuration"
echo "  Project  : $PROJECT"
echo "  Zone     : $ZONE"
echo "  Instance : $INSTANCE  ($MACHINE_TYPE + ${GPU_COUNT}×T4)"
echo "  Image    : $IMAGE_FAMILY ($IMAGE_PROJECT)"

# ── Static IP (idempotent) ────────────────────────────────────────────────────
log "Reserving static IP ($STATIC_IP_NAME in $REGION)"
if gcloud compute addresses describe "$STATIC_IP_NAME" \
     --region="$REGION" --project="$PROJECT" &>/dev/null; then
  warn "Address $STATIC_IP_NAME already exists — reusing."
else
  gcloud compute addresses create "$STATIC_IP_NAME" \
    --region="$REGION" \
    --project="$PROJECT"
fi

EXTERNAL_IP=$(gcloud compute addresses describe "$STATIC_IP_NAME" \
  --region="$REGION" --project="$PROJECT" --format='value(address)')
echo "  IP: $EXTERNAL_IP"

# ── Create instance (idempotent) ──────────────────────────────────────────────
if gcloud compute instances describe "$INSTANCE" \
     --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
  warn "Instance $INSTANCE already exists. Skipping creation."
  # Ensure it's running
  STATUS=$(gcloud compute instances describe "$INSTANCE" \
    --zone="$ZONE" --project="$PROJECT" --format='value(status)')
  if [[ "$STATUS" != "RUNNING" ]]; then
    log "Starting stopped instance…"
    gcloud compute instances start "$INSTANCE" --zone="$ZONE" --project="$PROJECT"
  fi
else
  log "Creating VM…"
  gcloud compute instances create "$INSTANCE" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$DISK_SIZE" \
    --boot-disk-type="$DISK_TYPE" \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --address="$EXTERNAL_IP" \
    --scopes=default,storage-rw
  echo "  VM created."
fi

# ── Wait for SSH ──────────────────────────────────────────────────────────────
log "Waiting for SSH (up to 3 min)…"
for i in $(seq 1 36); do
  if gcloud compute ssh "$INSTANCE" \
       --zone="$ZONE" --project="$PROJECT" \
       --command="echo ready" \
       --ssh-flag="-o ConnectTimeout=5 -o BatchMode=yes" \
       --quiet 2>/dev/null; then
    echo "  SSH ready (attempt $i)."
    break
  fi
  [[ $i -eq 36 ]] && die "SSH did not become available after 3 min."
  printf "  [%02d/36] waiting…\r" "$i"
  sleep 5
done

# ── Get remote username ───────────────────────────────────────────────────────
REMOTE_USER=$(gcloud compute ssh "$INSTANCE" \
  --zone="$ZONE" --project="$PROJECT" \
  --command="whoami" --quiet 2>/dev/null | tr -d '[:space:]')
echo "  Remote user: $REMOTE_USER"

# ── Update ~/.ssh/config ──────────────────────────────────────────────────────
log "Updating $SSH_CONFIG"
touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"

# Remove existing binder-vm block
python3 - "$SSH_CONFIG" <<'PYEOF'
import sys, re
path = sys.argv[1]
with open(path) as f:
    text = f.read()
text = re.sub(r'\n# >>> binder-vm >>>.*?# <<< binder-vm <<<\n', '\n',
              text, flags=re.DOTALL)
with open(path, 'w') as f:
    f.write(text)
PYEOF

# Append new block
cat >> "$SSH_CONFIG" <<SSHEOF

# >>> binder-vm >>>
Host binder-vm
    HostName ${EXTERNAL_IP}
    User ${REMOTE_USER}
    IdentityFile ~/.ssh/google_compute_engine
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 10
# <<< binder-vm <<<
SSHEOF
echo "  Added: ssh binder-vm"

# ── Upload and run bootstrap script ──────────────────────────────────────────
log "Uploading bootstrap script…"
gcloud compute scp \
  "${SCRIPT_DIR}/infra/vm_bootstrap.sh" \
  "${INSTANCE}:~/vm_bootstrap.sh" \
  --zone="$ZONE" --project="$PROJECT" --quiet

log "Running bootstrap (≈60–90 min — grab a coffee)…"
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" -- \
  bash -l ~/vm_bootstrap.sh 2>&1 | tee "${SCRIPT_DIR}/infra/bootstrap.log"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo " Setup complete!"
echo ""
echo "  ssh binder-vm                    — connect"
echo "  ssh binder-vm nvidia-smi         — verify GPU"
echo "  ssh binder-vm 'conda env list'   — list envs"
echo ""
echo "  Conda environments:"
echo "    rfdiffusion  — RFdiffusion backbone generation"
echo "    proteinmpnn  — ProteinMPNN sequence design"
echo "    chai1        — Chai-1 in silico evaluation"
echo "    analysis     — BioPython, biotite, transformers + this project"
echo ""
echo "  Cost: ~\$0.60/hr while running"
echo "  Stop: gcloud compute instances stop $INSTANCE --zone=$ZONE --project=$PROJECT"
echo "  Delete: bash teardown.sh"
echo "══════════════════════════════════════════════════════════════════════"
