#!/bin/bash
# 1_download_genomes.sh — Download bacterial reference genomes from NCBI in parallel.

set -euo pipefail

N=${1:-1000}
PARALLEL=${2:-8}
FNA_DIR="data/genomes/ncbi_dataset/data"

mkdir -p "$FNA_DIR"
mkdir -p data/raw/{train,val,test}

echo "========================================================"
echo "  Downloading $N bacterial genomes ($PARALLEL parallel)"
echo "========================================================"

# ── Step 1: Fetch all accession IDs (one fast API call) ───────────────────────
echo "Fetching accession list..."
ACCESSIONS=$(datasets summary genome taxon "bacteria" \
    --assembly-source RefSeq \
    --assembly-level complete \
    --exclude-atypical \
    --exclude-multi-isolate \
    --report ids_only \
    --as-json-lines \
    --limit "$N" 2>/dev/null \
    | python3 -c "
import sys, json
seen = []
for line in sys.stdin:
    try:
        obj = json.loads(line)
        for k,v in obj.items():
            if k.lower() in ('accession','currentaccession') and (v.startswith('GCF_') or v.startswith('GCA_')):
                if v not in seen: seen.append(v)
    except: pass
print('\n'.join(seen))
" | head -n "$N")

if [ -z "$ACCESSIONS" ]; then
    echo "ERROR: no bacterial accessions found — check NCBI datasets CLI."
    exit 1
fi

N_FOUND=$(echo "$ACCESSIONS" | wc -l)
echo "  Found $N_FOUND accessions — starting parallel download..."
echo ""

# ── Step 2: Download function (called in parallel via xargs) ──────────────────
export FNA_DIR

download_one() {
    acc="$1"
    dest="$FNA_DIR/$acc"

    # Skip if already downloaded
    if [ -f "$dest/${acc}.fna" ]; then
        echo "  [skip] $acc"
        return
    fi

    tmp=$(mktemp -d)
    ok=false
    for attempt in 1 2 3; do
        if datasets download genome accession "$acc" \
            --include genome \
            --filename "$tmp/genome.zip" \
            --no-progressbar 2>/dev/null; then
            ok=true
            break
        fi
        sleep $(( attempt * 5 ))
    done

    if $ok && [ -f "$tmp/genome.zip" ]; then
        found=$(unzip -l "$tmp/genome.zip" 2>/dev/null | grep '\.fna$' | awk '{print $NF}' | head -1)
        if [ -n "$found" ]; then
            unzip -q -j "$tmp/genome.zip" "$found" -d "$tmp/extracted" 2>/dev/null || true
            fna_file=$(find "$tmp/extracted" -name "*.fna" | head -1)
            if [ -n "$fna_file" ]; then
                mkdir -p "$dest"
                mv "$fna_file" "$dest/${acc}.fna"
                echo "  [OK]   $acc"
            else
                echo "  [FAIL] $acc — no .fna after unzip"
            fi
        else
            echo "  [FAIL] $acc — no .fna in archive"
        fi
    else
        echo "  [FAIL] $acc — download error after 3 attempts"
    fi
    rm -rf "$tmp"
}

export -f download_one

# ── Step 3: Run downloads in parallel ────────────────────────────────────────
echo "$ACCESSIONS" | xargs -P "$PARALLEL" -I{} bash -c 'download_one "$@"' _ {}

# ── Summary ───────────────────────────────────────────────────────────────────
N_OK=$(find "$FNA_DIR" -name "*.fna" | wc -l)
echo ""
echo "========================================================"
echo "  Total .fna files: $N_OK"
echo "  Genome files in : $FNA_DIR/"
echo ""
echo "  Done. Run next:"
echo "    bash run.sh"
echo "========================================================"
