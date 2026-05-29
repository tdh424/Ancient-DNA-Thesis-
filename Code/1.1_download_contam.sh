#!/bin/bash
# 1b_download_contam.sh — Download contamination references for realistic
# ancient-DNA library simulation (Setup B).
#
# Empirical aDNA libraries from bone/dental calculus typically contain:
#   - 5–30%   endogenous bacterial         (damaged — already downloaded by 1_download.sh)
#   - 30–70%  human contamination          (handlers, lab) — UNDAMAGED
#   - 10–40%  environmental bacteria       (soil, fungi)   — UNDAMAGED
#
# This script downloads:
#   1. Human chromosome 21 (~48 Mbp) — representative of human contamination
#   2. 10 common soil / environmental bacterial genomes
#
# Usage:
#   bash Code/1.1_download_contam.sh
#
# Outputs:
#   data/contam/human/chr21.fna
#   data/contam/env/<accession>/<accession>.fna  (×10)

set -euo pipefail

HUMAN_DIR="data/contam/human"
ENV_DIR="data/contam/env"

mkdir -p "$HUMAN_DIR" "$ENV_DIR"

# ── Step 1: human chr21 ──────────────────────────────────────────────────────
# Use NCBI Entrez REST to fetch chr21 of GRCh38 directly as FASTA.
# NC_000021.9 is the RefSeq accession for chromosome 21.
HUMAN_FA="$HUMAN_DIR/chr21.fna"
if [ ! -f "$HUMAN_FA" ] || [ ! -s "$HUMAN_FA" ]; then
    echo "Downloading human chr21 (GRCh38, NC_000021.9)..."
    curl -fsSL \
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_000021.9&rettype=fasta&retmode=text" \
        -o "$HUMAN_FA"
    # samtools (loaded as module) needed for indexing
    if command -v samtools &>/dev/null; then
        samtools faidx "$HUMAN_FA"
    fi
    echo "  $(grep -c '^>' "$HUMAN_FA") sequences, $(wc -c < "$HUMAN_FA") bytes"
fi

# ── Step 2: environmental bacteria ───────────────────────────────────────────
# A diverse panel of common soil / commensal / lab strains representative of
# typical environmental contamination in aDNA samples.
ENV_TAXA=(
    "Bacillus subtilis"
    "Pseudomonas putida"
    "Escherichia coli K-12"
    "Streptomyces coelicolor"
    "Mycobacterium smegmatis"
    "Staphylococcus epidermidis"
    "Lactobacillus plantarum"
    "Bifidobacterium longum"
    "Clostridioides difficile"
    "Acinetobacter baumannii"
)

if ! command -v datasets &>/dev/null; then
    echo "ERROR: NCBI 'datasets' CLI not found. Install via:"
    echo "  conda install -n ancient-dna -c conda-forge ncbi-datasets-cli"
    exit 1
fi

download_env_genome() {
    taxon="$1"
    safe=$(echo "$taxon" | tr ' /' '__')
    if compgen -G "$ENV_DIR/*/_*$safe*.fna" > /dev/null; then
        echo "  [skip] $taxon (already downloaded)"
        return
    fi

    tmp=$(mktemp -d)
    if datasets download genome taxon "$taxon" \
        --assembly-source RefSeq --assembly-level complete \
        --reference --include genome \
        --filename "$tmp/genome.zip" --no-progressbar 2>/dev/null; then
        unzip -q -o "$tmp/genome.zip" -d "$tmp/extracted" 2>/dev/null || true
        fna=$(find "$tmp/extracted" -name "*.fna" | head -1)
        if [ -n "$fna" ]; then
            acc=$(basename "$(dirname "$fna")")
            mkdir -p "$ENV_DIR/$acc"
            mv "$fna" "$ENV_DIR/$acc/${acc}.fna"
            command -v samtools &>/dev/null && samtools faidx "$ENV_DIR/$acc/${acc}.fna"
            echo "  [OK]   $taxon  →  $acc"
        else
            echo "  [FAIL] $taxon — no .fna in archive"
        fi
    else
        echo "  [FAIL] $taxon — download error"
    fi
    rm -rf "$tmp"
}

echo ""
echo "Downloading environmental bacteria..."
for taxon in "${ENV_TAXA[@]}"; do
    download_env_genome "$taxon"
done

n_env=$(find "$ENV_DIR" -name "*.fna" | wc -l)
echo ""
echo "========================================================"
echo "  Human chr21        : $([ -f "$HUMAN_FA" ] && echo OK || echo MISSING)"
echo "  Environmental bact : $n_env genomes"
echo "  Run next: bash Code/3_simulate.sh && bash Code/3.1_simulate_contam.sh"
echo "========================================================"
