#!/usr/bin/env bash
# infra/vm_bootstrap.sh — Run on the GCP VM after creation.
# Sets up conda environments for RFdiffusion, ProteinMPNN, Chai-1, and analysis.
#
# Do NOT run locally — this is uploaded and executed by setup.sh.
# Safe to re-run: each step is idempotent (checks before creating).

set -euo pipefail

log()  { echo -e "\n\033[1;32m[bootstrap] $*\033[0m"; }
step() { echo -e "\033[0;36m  → $*\033[0m"; }

TOOLS="$HOME/tools"
mkdir -p "$TOOLS"

# ── Conda ─────────────────────────────────────────────────────────────────────
# DL VM image ships conda at /opt/conda; initialise for this script
CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
[[ -f "$CONDA_SH" ]] || { echo "ERROR: conda not found at $CONDA_SH"; exit 1; }
# shellcheck source=/dev/null
source "$CONDA_SH"

log "Conda version: $(conda --version)"
log "GPU check:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── System packages ───────────────────────────────────────────────────────────
log "System packages"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  git wget curl build-essential tmux htop nvtop tree \
  libgl1-mesa-glx libglib2.0-0 2>/dev/null

# ══════════════════════════════════════════════════════════════════════════════
# ENV 1 — rfdiffusion
# Official path: conda yaml from the repo (pins PyTorch 1.12+cu116, safe on T4)
# Ref: https://github.com/RosettaCommons/RFdiffusion
# ══════════════════════════════════════════════════════════════════════════════
log "ENV: rfdiffusion"

step "Cloning RFdiffusion"
if [[ ! -d "$TOOLS/RFdiffusion/.git" ]]; then
  git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion.git \
    "$TOOLS/RFdiffusion"
else
  echo "  Already cloned — skipping."
fi

step "Creating conda env from SE3nv.yml"
if conda env list | grep -q "^rfdiffusion "; then
  echo "  Env 'rfdiffusion' already exists — skipping creation."
else
  # Use their official yaml; it pins torch 1.12+cu116 (T4 = sm_75, CUDA 11.6 ✓)
  conda env create -n rfdiffusion \
    -f "$TOOLS/RFdiffusion/env/SE3nv.yml"
fi

step "Installing SE3Transformer (custom RFdiffusion fork)"
conda run -n rfdiffusion bash -c "
  cd '$TOOLS/RFdiffusion/env/SE3Transformer'
  pip install --quiet --no-cache-dir -r requirements.txt
  python setup.py install --quiet 2>/dev/null
"

step "Installing RFdiffusion package"
conda run -n rfdiffusion pip install --quiet --no-cache-dir \
  -e "$TOOLS/RFdiffusion"

step "Downloading RFdiffusion model weights"
mkdir -p "$TOOLS/RFdiffusion/models"
BASE_URL="http://files.ipd.uw.edu/pub/RFdiffusion"

declare -A WEIGHTS=(
  ["Base_ckpt.pt"]="6f5902ac237024bdd0c176cb93063dc6"
  ["Complex_base_ckpt.pt"]="e29311f6f1bf1af907f9ef9f44b8328b"
  ["Complex_Fold_base_ckpt.pt"]="60f09a193fb5e5ccdc4980417708dbab"
  ["InpaintSeq_ckpt.pt"]="74f51cfb8b440f50d70878e05361d8f0"
  ["InpaintSeq_Fold_ckpt.pt"]="76d00716416567174cdb7ca96e208296"
  ["ActiveSite_ckpt.pt"]="5532d2e1f3a4738decd58b19d633b3c3"
  ["Base_epoch8_ckpt.pt"]="12fc204edeae5b57713c5ad7dcb97d39"
  ["Complex_beta_ckpt.pt"]="f572d396fae9206628714fb2ce00f72e"
)

for fname in "${!WEIGHTS[@]}"; do
  dest="$TOOLS/RFdiffusion/models/$fname"
  if [[ -f "$dest" ]]; then
    echo "  $fname — already downloaded."
  else
    echo "  Downloading $fname…"
    wget -q --show-progress \
      -O "$dest" \
      "${BASE_URL}/${WEIGHTS[$fname]}/${fname}"
  fi
done

step "rfdiffusion smoke test"
conda run -n rfdiffusion python -c "
import torch
print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
import rfdiffusion
print('  rfdiffusion import OK')
"

# ══════════════════════════════════════════════════════════════════════════════
# ENV 2 — proteinmpnn
# Minimal env: PyTorch 2 + numpy + biopython.
# ProteinMPNN is script-based (no installable package); clone is sufficient.
# Ref: https://github.com/dauparas/ProteinMPNN
# ══════════════════════════════════════════════════════════════════════════════
log "ENV: proteinmpnn"

step "Cloning ProteinMPNN"
if [[ ! -d "$TOOLS/ProteinMPNN/.git" ]]; then
  git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git \
    "$TOOLS/ProteinMPNN"
else
  echo "  Already cloned — skipping."
fi

step "Creating conda env"
if conda env list | grep -q "^proteinmpnn "; then
  echo "  Env 'proteinmpnn' already exists — skipping creation."
else
  conda create -y -n proteinmpnn python=3.9
fi

step "Installing dependencies"
conda run -n proteinmpnn pip install --quiet --no-cache-dir \
  torch==2.0.1+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

conda run -n proteinmpnn pip install --quiet --no-cache-dir \
  numpy biopython

step "proteinmpnn smoke test"
conda run -n proteinmpnn python -c "
import torch
print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
import numpy, Bio
print('  numpy + biopython import OK')
"

# Write a convenience wrapper so other scripts can call ProteinMPNN without
# activating the env manually
cat > "$TOOLS/run_proteinmpnn.sh" <<'EOF'
#!/usr/bin/env bash
# run_proteinmpnn.sh — thin wrapper around protein_mpnn_run.py
# Usage: bash ~/tools/run_proteinmpnn.sh [args...]
source /opt/conda/etc/profile.d/conda.sh
conda activate proteinmpnn
python ~/tools/ProteinMPNN/protein_mpnn_run.py "$@"
EOF
chmod +x "$TOOLS/run_proteinmpnn.sh"

# ══════════════════════════════════════════════════════════════════════════════
# ENV 3 — chai1
# Ref: https://github.com/chaidiscovery/chai-lab
# Weights are downloaded automatically on first inference call.
# ══════════════════════════════════════════════════════════════════════════════
log "ENV: chai1"

step "Creating conda env"
if conda env list | grep -q "^chai1 "; then
  echo "  Env 'chai1' already exists — skipping creation."
else
  conda create -y -n chai1 python=3.10
fi

step "Installing chai_lab"
# chai_lab[extras] includes all optional but recommended dependencies
conda run -n chai1 pip install --quiet --no-cache-dir "chai_lab[extras]"

step "chai1 smoke test"
conda run -n chai1 python -c "
import chai_lab
print(f'  chai_lab {chai_lab.__version__} import OK')
import torch
print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
"

# ══════════════════════════════════════════════════════════════════════════════
# ENV 4 — analysis
# Full scientific stack matching requirements.txt + the protein_rl project.
# ══════════════════════════════════════════════════════════════════════════════
log "ENV: analysis"

step "Creating conda env"
if conda env list | grep -q "^analysis "; then
  echo "  Env 'analysis' already exists — skipping creation."
else
  conda create -y -n analysis python=3.10
fi

step "Installing PyTorch (CUDA 11.8)"
conda run -n analysis pip install --quiet --no-cache-dir \
  torch==2.0.1+cu118 torchvision \
  --extra-index-url https://download.pytorch.org/whl/cu118

step "Installing requirements.txt packages"
conda run -n analysis pip install --quiet --no-cache-dir \
  numpy>=1.24 \
  pandas>=2.0 \
  matplotlib>=3.7 \
  seaborn>=0.12 \
  scipy>=1.11 \
  biopython==1.84 \
  py3Dmol>=2.0 \
  torchvision \
  "transformers>=4.35" \
  biotite>=0.38 \
  mdanalysis>=2.6 \
  "scikit-learn>=1.3" \
  "gpytorch>=1.11" \
  "jupyterlab>=4.0" \
  "ipywidgets>=8.0" \
  "plotly>=5.17" \
  "kaleido>=0.2" \
  "tqdm>=4.65" \
  "pyyaml>=6.0" \
  "requests>=2.31"

step "Installing chai_lab into analysis env (for evaluate.py)"
conda run -n analysis pip install --quiet --no-cache-dir "chai_lab[extras]"

step "Installing the protein_rl project itself (editable)"
if [[ -d "$HOME/protein_rl" ]]; then
  # Repo already on VM (e.g. from git clone)
  conda run -n analysis pip install --quiet --no-cache-dir -e "$HOME/protein_rl"
fi
# Note: user can 'git clone' the repo to ~/protein_rl and re-run this step

step "analysis smoke test"
conda run -n analysis python -c "
import numpy, pandas, matplotlib, scipy, Bio, biotite, transformers, gpytorch
import torch
print(f'  numpy {numpy.__version__}, pandas {pandas.__version__}')
print(f'  biopython {Bio.__version__}, biotite {biotite.__version__}')
print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
print('  All analysis imports OK')
"

# ══════════════════════════════════════════════════════════════════════════════
# Shell configuration
# ══════════════════════════════════════════════════════════════════════════════
log "Configuring shell"

# Bashrc additions (idempotent guard)
if ! grep -q "# >>> protein-binder-aliases >>>" "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<'BASHRC'

# >>> protein-binder-aliases >>>
alias ca-rfd='conda activate rfdiffusion'
alias ca-mpnn='conda activate proteinmpnn'
alias ca-chai='conda activate chai1'
alias ca-ana='conda activate analysis'

# Quick GPU status
alias gpus='nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader'

# Project shortcuts
export TOOLS="$HOME/tools"
export RFDIFFUSION_DIR="$TOOLS/RFdiffusion"
export PROTEINMPNN_DIR="$TOOLS/ProteinMPNN"
alias rfdiff='conda run -n rfdiffusion python $RFDIFFUSION_DIR/scripts/run_inference.py'
alias mpnn='bash $TOOLS/run_proteinmpnn.sh'
# <<< protein-binder-aliases <<<
BASHRC
  echo "  .bashrc updated."
fi

# ── Final summary ─────────────────────────────────────────────────────────────
log "Bootstrap complete"
echo ""
echo "  Conda environments:"
conda env list | grep -E "rfdiffusion|proteinmpnn|chai1|analysis" | sed 's/^/    /'
echo ""
echo "  Tools:"
ls -1 "$TOOLS/"
echo ""
echo "  GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'
echo ""
echo "  Disk usage:"
df -h / | awk 'NR>1 {print "    " $0}'
echo ""
echo "  Quick start:"
echo "    ca-rfd    → conda activate rfdiffusion"
echo "    ca-mpnn   → conda activate proteinmpnn"
echo "    ca-chai   → conda activate chai1"
echo "    ca-ana    → conda activate analysis"
