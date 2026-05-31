#!/bin/bash
# 1.1_download_contam.sh — Download human chr21 and environmental bacteria as modern contamination references.

set -euo pipefail

HUMAN_DIR="data/contam/human"
ENV_DIR="data/contam/env"

mkdir -p "$HUMAN_DIR" "$ENV_DIR"

# ── Step 1: human chr21 ──────────────────────────────────────────────────────
# Fetch chr21 of GRCh38 (RefSeq NC_000021.9) as FASTA via NCBI Entrez.
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
# Common soil / commensal / lab strains representing environmental contamination.
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
