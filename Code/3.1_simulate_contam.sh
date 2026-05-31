#!/bin/bash
# 3.1_simulate_contam.sh — Append modern human and environmental contamination to the simulated library.

set -euo pipefail

GARGAMMEL_BIN="${GARGAMMEL_BIN:-Code/Gargammel/Package/src}"
HUMAN_DIR="data/contam/human"
ENV_DIR="data/contam/env"
OUT_DIR="${OUT_DIR:-data/raw}"

# All three sources (bacterial, human, environmental) use the SAME
# log-normal(3.9, 0.4) length distribution, filtered to 30–100 bp.
HUMAN_LOC=3.9
HUMAN_SCALE=0.4
ENV_LOC=3.9
ENV_SCALE=0.4
MIN_LEN=30
MAX_LEN=100

# Sequencing error rate — must match 3_simulate.sh
SEQ_ERR_RATE="${SEQ_ERR_RATE:-0.005}"

# Per-split fragment counts (roughly 50% / 20% of the bacterial total)
HUMAN_FRAGS_TRAIN=650000
HUMAN_FRAGS_VAL=80000
HUMAN_FRAGS_TEST=80000
ENV_FRAGS_TRAIN=260000
ENV_FRAGS_VAL=32000
ENV_FRAGS_TEST=32000

# ── Sanity ──────────────────────────────────────────────────────────────────
if [ ! -f "$GARGAMMEL_BIN/fragSim" ]; then
    echo "ERROR: fragSim not found at $GARGAMMEL_BIN/fragSim"; exit 1
fi
HUMAN_FA="$HUMAN_DIR/chr21.fna"
if [ ! -s "$HUMAN_FA" ]; then
    echo "ERROR: $HUMAN_FA missing — run Code/1.1_download_contam.sh first."; exit 1
fi

module load samtools 2>/dev/null || true
[ -f "${HUMAN_FA}.fai" ] || samtools faidx "$HUMAN_FA"

# ── Helper: fragment + add prefix to headers + append to clean/damaged ──────
# Args: genome n_frags prefix split loc scale seed
simulate_clean_source() {
    local genome="$1"; local n_frags="$2"; local prefix="$3"
    local split="$4";  local loc="$5";     local scale="$6"
    local seed="${7:-0}"

    local tmp; tmp=$(mktemp -d)
    "$GARGAMMEL_BIN/fragSim" \
        -n "$n_frags" --loc "$loc" --scale "$scale" \
        -m "$MIN_LEN" -M "$MAX_LEN" \
        "$genome" > "$tmp/raw.fasta" 2>/dev/null || true

    local n; n=$(grep -c "^>" "$tmp/raw.fasta" 2>/dev/null || echo 0)
    if [ "${n:-0}" -eq 0 ]; then
        rm -rf "$tmp"; return 1
    fi

    # Prefix headers so 4_build_dataset.py can track source
    awk -v p="$prefix" '/^>/ {sub(/^>/, ">" p); print; next} {print}' \
        "$tmp/raw.fasta" > "$tmp/clean.fasta"

    # Apply sequencing errors to the 'damaged' copy only (clean stays as
    # ground truth — it represents the un-sequenced reference base).
    if [ "$(awk -v r=$SEQ_ERR_RATE 'BEGIN{print (r>0)?1:0}')" = "1" ]; then
        python3 - "$tmp/clean.fasta" "$tmp/damaged.fasta" "$SEQ_ERR_RATE" "$seed" <<'PY'
import sys, random
inp, out, rate, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
rng = random.Random(seed * 7919 + 13)
ALPH = 'ACGT'
with open(inp) as fi, open(out, 'w') as fo:
    for line in fi:
        if line.startswith('>'): fo.write(line); continue
        s = list(line.rstrip())
        for j, b in enumerate(s):
            if b in ALPH and rng.random() < rate:
                s[j] = ALPH[(ALPH.index(b) + rng.randint(1, 3)) % 4]
        fo.write(''.join(s) + '\n')
PY
    else
        cp "$tmp/clean.fasta" "$tmp/damaged.fasta"
    fi

    cat "$tmp/clean.fasta"   >> "$OUT_DIR/$split/clean.fasta"
    cat "$tmp/damaged.fasta" >> "$OUT_DIR/$split/damaged.fasta"
    rm -rf "$tmp"
    echo "  $n"
}

# ── Index env bacteria ──────────────────────────────────────────────────────
ENV_GENOMES=()
while IFS= read -r f; do ENV_GENOMES+=("$f"); done \
    < <(find "$ENV_DIR" -name "*.fna" 2>/dev/null | sort)
N_ENV=${#ENV_GENOMES[@]}

if [ "$N_ENV" -eq 0 ]; then
    echo "WARNING: no env bacteria in $ENV_DIR. Run 1.1_download_contam.sh."
fi
for g in "${ENV_GENOMES[@]}"; do [ -f "${g}.fai" ] || samtools faidx "$g"; done

echo "========================================================"
echo "  Simulating modern contamination (Setup B)"
echo "  Human  chr21   : $HUMAN_FA"
echo "  Env bacteria   : $N_ENV genomes"
echo "========================================================"

# ── Per-split simulation ────────────────────────────────────────────────────
for split in train val test; do
    case $split in
        train) HF=$HUMAN_FRAGS_TRAIN; EF=$ENV_FRAGS_TRAIN ;;
        val)   HF=$HUMAN_FRAGS_VAL;   EF=$ENV_FRAGS_VAL   ;;
        test)  HF=$HUMAN_FRAGS_TEST;  EF=$ENV_FRAGS_TEST  ;;
    esac

    echo ""
    echo "── $split ──────────────────────────────"
    if [ ! -f "$OUT_DIR/$split/clean.fasta" ]; then
        echo "  WARNING: $OUT_DIR/$split/clean.fasta missing — run 3_simulate.sh first."
        continue
    fi

    n_before_c=$(grep -c "^>" "$OUT_DIR/$split/clean.fasta" || true)

    # Human chr21 → HF fragments (longer log-normal: modern fresh DNA)
    printf "  human (%d frags, loc=%s)..." "$HF" "$HUMAN_LOC"
    simulate_clean_source "$HUMAN_FA" "$HF" "HUMAN_" "$split" \
        "$HUMAN_LOC" "$HUMAN_SCALE" 1001 || echo " FAIL"

    # Environmental bact → distribute EF over N_ENV genomes (intermediate length)
    if [ "$N_ENV" -gt 0 ]; then
        per=$(( EF / N_ENV ))
        printf "  env (%d frags / %d genomes, loc=%s)..." "$EF" "$N_ENV" "$ENV_LOC"
        total=0
        idx=0
        for g in "${ENV_GENOMES[@]}"; do
            n=$(simulate_clean_source "$g" "$per" "ENV_" "$split" \
                "$ENV_LOC" "$ENV_SCALE" $((2000 + idx)) 2>/dev/null || echo 0)
            total=$((total + ${n:-0}))
            idx=$((idx + 1))
        done
        echo "  $total"
    fi

    n_after_c=$(grep -c "^>" "$OUT_DIR/$split/clean.fasta" || true)
    echo "  Reads in $split/clean.fasta: $n_before_c → $n_after_c"
done

echo ""
echo "========================================================"
echo "  STATUS : Contamination added"
echo "  Run next: python Code/4_build_dataset.py"
echo "========================================================"
