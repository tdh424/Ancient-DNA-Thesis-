#!/bin/bash
# 0_setup.sh — Build Gargammel and verify dependencies. Run once before anything else.

set -euo pipefail

echo "========================================"
echo "  Ancient DNA Pipeline — Setup"
echo "========================================"

# ── Load cluster modules ──────────────────────────────────────────────────────
module load miniconda/24.5.0 2>/dev/null || true
module load gcc/11.2.0        2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate ancient-dna 2>/dev/null || true

# ── Create data directory structure ──────────────────────────────────────────
mkdir -p data/{raw/{train,val,test},genomes}
mkdir -p Code/outputs/{models,results,figures,logs}
mkdir -p logs
echo "Directory structure created."

# ── Build Gargammel ───────────────────────────────────────────────────────────
echo ""
echo "Building Gargammel..."
cd Code/Gargammel
make setup-package
cd ../..
echo "  [OK] Gargammel built."

# ── Set GARGAMMEL_BIN for use by 3_simulate.sh ───────────────────────────────
GARGAMMEL_BIN="$(pwd)/Code/Gargammel/Package"
echo ""
echo "Gargammel binaries are in: $GARGAMMEL_BIN/src/"
echo ""
echo "Add this line to your ~/.bashrc or pass it to 3_simulate.sh:"
echo "  export GARGAMMEL_BIN=$GARGAMMEL_BIN/src"
echo ""
echo "========================================"
echo "  Setup complete. Run next:"
echo "  bash Code/1_download_genomes.sh 50"
echo "========================================"
