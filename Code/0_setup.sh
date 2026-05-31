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

# ── Check NCBI datasets CLI ───────────────────────────────────────────────────
echo ""
echo "Checking dependencies..."
if command -v datasets &>/dev/null; then
    echo "  [OK] NCBI datasets CLI: $(datasets --version 2>&1 | head -1)"
else
    echo "  [MISSING] NCBI datasets CLI — installing..."
    mkdir -p ~/bin
    DATASETS_URL="https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64/datasets"
    if command -v curl &>/dev/null; then
        curl -fsSL "$DATASETS_URL" -o ~/bin/datasets
    elif command -v wget &>/dev/null; then
        wget -q "$DATASETS_URL" -O ~/bin/datasets
    else
        echo "  [ERROR] Neither curl nor wget found. Install manually:"
        echo "    mkdir -p ~/bin"
        echo "    curl -fsSL $DATASETS_URL -o ~/bin/datasets"
        echo "    chmod +x ~/bin/datasets"
        exit 1
    fi
    chmod +x ~/bin/datasets
    export PATH="$HOME/bin:$PATH"
    echo "  [OK] NCBI datasets CLI installed to ~/bin/datasets"
    echo ""
    echo "  Add this to your ~/.bashrc so it persists:"
    echo "    export PATH=\"\$HOME/bin:\$PATH\""
fi

# ── Build Gargammel ───────────────────────────────────────────────────────────
echo ""
echo "Building Gargammel..."
cd Code/Gargammel
make setup-package
cd ../..
echo "  [OK] Gargammel built."

# ── Set GARGAMMEL_BIN for use by 1_simulate.sh ───────────────────────────────
GARGAMMEL_BIN="$(pwd)/Code/Gargammel/Package"
echo ""
echo "Gargammel binaries are in: $GARGAMMEL_BIN/src/"
echo ""
echo "Add this line to your ~/.bashrc or pass it to 1_simulate.sh:"
echo "  export GARGAMMEL_BIN=$GARGAMMEL_BIN/src"
echo ""
echo "========================================"
echo "  Setup complete. Run next:"
echo "  bash Code/1_download_genomes.sh 50"
echo "========================================"
