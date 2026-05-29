#!/bin/bash
# 16_run_udg_test.sh — Generate a UDG-treated test dataset for cross-protocol
# evaluation, then evaluate trained models on it.
#
# Idea: re-run the simulation pipeline with `UDG=true` (interior C→T removed,
# only terminal damage retained), but only for the test split, writing outputs
# into separate `data/raw_udg/` and `data/test_udg.npz` so the original
# training data is untouched.
#
# Pre-requisites:
#   - Trained models exist (denoiser_*.pt, classifier_probs*.npz)
#   - Conda env `ancient-dna` activated
#   - samtools loaded
#
# Usage:
#   bash Code/13_run_udg_test.sh
#
# Outputs:
#   data/raw_udg/test/{clean,damaged}.fasta
#   data/test_udg.npz
#   outputs/results/udg_evaluation.txt
#   outputs/figures/udg_evaluation.png

set -euo pipefail

mkdir -p data/raw_udg/{train,val,test}

# ── Step 1: bacterial endogenous reads with UDG damage (test split only) ────
# We still need to walk all genomes to honour the homology-aware split, but
# we'll only write out the 'test' split fragments. The others get tossed.
echo "── Step 1: simulating UDG bacterial reads (test only)..."
UDG=true \
OUT_DIR=data/raw_udg \
SPLITS=test \
bash Code/3_simulate.sh

# Empty out train/val so 2b_simulate_contam.sh and 3_dataset.py don't see them
> data/raw_udg/train/clean.fasta || true
> data/raw_udg/train/damaged.fasta || true
> data/raw_udg/val/clean.fasta || true
> data/raw_udg/val/damaged.fasta || true

# ── Step 2: contamination (no UDG dependency, same as main) ─────────────────
echo "── Step 2: adding contamination reads (test only)..."
OUT_DIR=data/raw_udg \
bash Code/3.1_simulate_contam.sh

# ── Step 3: encode → data/test_udg.npz ──────────────────────────────────────
echo "── Step 3: building data/test_udg.npz..."
RAW_DIR=data/raw_udg \
OUT_DIR=data \
OUT_SUFFIX=_udg \
SPLITS=test \
python Code/4_build_dataset.py

# ── Step 4: evaluate trained models on the UDG test set ─────────────────────
echo "── Step 4: evaluating trained models on UDG test set..."
python Code/13.1_evaluate_udg.py

echo ""
echo "========================================================"
echo "  UDG cross-protocol evaluation complete."
echo "  Results: outputs/results/udg_evaluation.txt"
echo "  Figure : outputs/figures/udg_evaluation.png"
echo "========================================================"
