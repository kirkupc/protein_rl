#!/usr/bin/env bash
# teardown.sh — Delete the binder-vm GCP instance and release its static IP.
#
# Usage:
#   bash teardown.sh
#   GCP_PROJECT=my-project ZONE=us-central1-a bash teardown.sh

set -euo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-us-central1-a}"
REGION="${ZONE%-*}"
INSTANCE="binder-vm"
STATIC_IP_NAME="${INSTANCE}-ip"
SSH_CONFIG="${HOME}/.ssh/config"

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
die()  { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

[[ -z "$PROJECT" ]] && die "GCP project not set."
command -v gcloud >/dev/null 2>&1 || die "gcloud CLI not found."

echo "══════════════════════════════════════════════════════════════"
echo " This will PERMANENTLY delete:"
echo "   • VM instance : $INSTANCE (zone: $ZONE)"
echo "   • Static IP   : $STATIC_IP_NAME (region: $REGION)"
echo "   • SSH config  : binder-vm entry in $SSH_CONFIG"
echo ""
echo " Any data on the boot disk will be lost."
echo "══════════════════════════════════════════════════════════════"
read -rp "Type 'yes' to confirm: " CONFIRM
[[ "$CONFIRM" == "yes" ]] || { echo "Aborted."; exit 0; }

# ── Delete instance ───────────────────────────────────────────────────────────
log "Deleting VM instance: $INSTANCE"
if gcloud compute instances describe "$INSTANCE" \
     --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
  gcloud compute instances delete "$INSTANCE" \
    --zone="$ZONE" \
    --project="$PROJECT" \
    --quiet
  echo "  Deleted."
else
  echo "  Instance not found — skipping."
fi

# ── Release static IP ─────────────────────────────────────────────────────────
log "Releasing static IP: $STATIC_IP_NAME"
if gcloud compute addresses describe "$STATIC_IP_NAME" \
     --region="$REGION" --project="$PROJECT" &>/dev/null; then
  gcloud compute addresses delete "$STATIC_IP_NAME" \
    --region="$REGION" \
    --project="$PROJECT" \
    --quiet
  echo "  Released."
else
  echo "  Address not found — skipping."
fi

# ── Remove from SSH config ────────────────────────────────────────────────────
log "Removing binder-vm from $SSH_CONFIG"
if [[ -f "$SSH_CONFIG" ]] && grep -q "# >>> binder-vm >>>" "$SSH_CONFIG"; then
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
  echo "  Removed."
else
  echo "  No binder-vm entry found — skipping."
fi

echo ""
echo "Teardown complete."
