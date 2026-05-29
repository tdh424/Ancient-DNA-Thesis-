#!/bin/bash
# 2_simulate.sh — Simulate realistic ancient DNA damage with Gargammel.
#
# Produces (clean, damaged) FASTA pairs using the Briggs et al. (2007) model.
# Includes both C→T (5' end) and G→A (3' end) damage symmetrically.
#
# fragSim parameters:
#   --loc 3.9  --scale 0.4   Log-normal fragment length (mean ~53 bp,
#                            typical aDNA size distribution)
#   -m 30  -M 150            Keep only reads 30–150 bp (discard extremes)
#
# deamSim -damage v,l,d,s — empirical Briggs (2007) fit to the Vi-33.16
# Neanderthal sample, Table 1 of Briggs et al., PNAS 104(37):14616–14621:
#   v = 0.0241    nick frequency (ν)
#   l = 0.3590    geometric decay constant from read ends (λ)
#   d = 0.00937   double-stranded interior deamination rate (δ_d)
#   s = 0.6815    single-stranded overhang deamination rate (δ_s)
#
# Resulting per-position C→T probability  p(k) = s·λ^k + d :
#   k=0  →  0.691      k=1  →  0.254     k=2  →  0.097
#   k=3  →  0.041      k=7  →  0.012     (G→A symmetric at the 3' end)
#
# Note: this script runs only fragSim + deamSim. We do NOT add Illumina
# sequencing errors (gargammel's art_illumina step) — every per-position
# difference between clean and damaged comes from the Briggs model only.
#
# Usage:
#   bash Code/3_simulate.sh
#
# Output → data/raw/{train,val,test}/clean.fasta + damaged.fasta

set -euo pipefail

GARGAMMEL_BIN="${GARGAMMEL_BIN:-Code/Gargammel/Package/src}"
GENOME_DIR="data/genomes/ncbi_dataset/data"
# OUT_DIR can be overridden so UDG variant lands in data/raw_udg/ etc.
OUT_DIR="${OUT_DIR:-data/raw}"
FRAGS_PER_GENOME=5000

# UDG mode. Set to "true" to suppress interior C→T (d=0),
# emulating UDG/USER enzymatic treatment used by many modern aDNA labs.
# Override at the command line: UDG=true bash Code/3_simulate.sh
UDG="${UDG:-false}"

# Fragment length: log-normal — short, typical aDNA endogenous distribution.
# loc=3.9, scale=0.4 → mean ~53bp. All sources
# use the SAME length distribution and the upper filter is dropped from
# 150 bp to 100 bp so that no read needs to be truncated at the encoding
# step (Code/4_build_dataset.py). This removes the length-as-shortcut
# loophole the classifier was using to discriminate bact vs. human.
LOC=3.9
SCALE=0.4
MIN_LEN=30
MAX_LEN=100

# Briggs (2007) damage parameters — fitted empirical values for the
# Vi-33.16 Neanderthal sample (Table 1, Briggs et al. PNAS 2007).
#       v        l        d         s
# v = nick frequency      l = geometric decay
# d = ds interior rate    s = ss overhang rate
# Per-genome `s` is sampled from Beta(α=4, β=2) so the library
# captures realistic preservation heterogeneity (s ranges ~0.2–0.95).  v, l, d
# are kept fixed (well-determined population parameters in Briggs 2007).
BRIGGS_V=0.0241
BRIGGS_L=0.3590
BRIGGS_D=0.00937
BRIGGS_S_MEAN=0.6815   # Vi-33.16 reference; per-genome s drawn around this

# Sequencing error rate applied AFTER deamSim.  Real Illumina
# reads carry ~0.1–1% error per base; we use 0.5% as a typical rate.
SEQ_ERR_RATE=0.005

# Train/val/test split fractions
TRAIN_FRAC=0.80
VAL_END_FRAC=0.90   # cumulative (80–90% = val)

if [ ! -f "$GARGAMMEL_BIN/fragSim" ]; then
    echo "ERROR: fragSim not found at $GARGAMMEL_BIN/fragSim"
    echo "Run bash Code/0_setup.sh first to build Gargammel."
    exit 1
fi

if [ ! -d "$GENOME_DIR" ]; then
    echo "ERROR: $GENOME_DIR not found. Run bash Code/1_download_genomes.sh first."
    exit 1
fi

mkdir -p "$OUT_DIR"/{train,val,test}

# Clear old FASTA files so each run starts fresh
for split in train val test; do
    rm -f "$OUT_DIR/$split/clean.fasta" "$OUT_DIR/$split/damaged.fasta"
done

# Index genomes with samtools faidx (fragSim requires .fai)
module load samtools 2>/dev/null || true
if ! command -v samtools &>/dev/null; then
    echo "ERROR: samtools not found. Load it with 'module load samtools'."
    exit 1
fi
echo "Indexing genomes..."
find "$GENOME_DIR" -name "*.fna" | while read -r f; do
    [ -f "${f}.fai" ] || samtools faidx "$f" 2>/dev/null
done
echo "  Done indexing."

# Collect genome files (sorted for deterministic train/val/test split —
# the same ordering is used by 4b_bwa.py and run_pydamage.sh so reads are
# always aligned against the genomes they were simulated from)
GENOMES=()
while IFS= read -r f; do
    GENOMES+=("$f")
done < <(find "$GENOME_DIR" -name "*.fna" 2>/dev/null | sort)

N_GENOMES=${#GENOMES[@]}
if [ "$N_GENOMES" -eq 0 ]; then
    echo "ERROR: no .fna files found in $GENOME_DIR"
    exit 1
fi

TRAIN_END=$(awk "BEGIN{printf \"%d\", $N_GENOMES * $TRAIN_FRAC}")
VAL_END=$(awk "BEGIN{printf \"%d\", $N_GENOMES * $VAL_END_FRAC}")

# Optional homology-aware split: if cluster_genomes.py has produced
# split_assignment.tsv, use it instead of the index-based split.
SPLIT_TSV="data/genomes/split_assignment.tsv"
USE_TSV=false
declare -A ACC_SPLIT
if [ -f "$SPLIT_TSV" ]; then
    USE_TSV=true
    while IFS=$'\t' read -r acc sp _rest; do
        [ "$acc" = "accession" ] && continue
        ACC_SPLIT["$acc"]="$sp"
    done < "$SPLIT_TSV"
    echo "Using homology-aware split from $SPLIT_TSV "
    echo "  (${#ACC_SPLIT[@]} accessions)"
fi

# Helper: sample s ~ Beta(α=4, β=2) (mean ≈ 0.667, var ≈ 0.032).  Implemented
# in Python because awk lacks Beta sampling.  Seeded by genome index for
# reproducibility.
sample_per_genome_s() {
    local seed="$1"; local mean="$2"
    python3 -c "
import numpy as np, sys
rng = np.random.default_rng($seed)
# Beta(4, 2) re-scaled so its mean matches BRIGGS_S_MEAN (=$mean).
raw  = rng.beta(4, 2)            # mean 0.6667
s    = raw * ($mean / 0.6667)
print(f'{min(s, 0.95):.4f}')
"
}

echo "========================================================"
echo "  Simulating aDNA damage for $N_GENOMES genomes"
echo "  Fragments per genome : $FRAGS_PER_GENOME"
echo "  Fragment length      : log-normal (loc=$LOC, scale=$SCALE) — endogenous"
echo "  Briggs base s,l,d,v  : variable s, l=$BRIGGS_L, d=$BRIGGS_D, v=$BRIGGS_V"
echo "  Per-genome s         : Beta(4,2) re-scaled around $BRIGGS_S_MEAN"
echo "  UDG mode             : $UDG  (true ⇒ d=0, only terminal C→T)"
echo "  Sequencing error     : $SEQ_ERR_RATE"
if $USE_TSV; then
    echo "  Split strategy       : homology-aware (Mash clusters)"
else
    echo "  Split strategy       : index-based (no clustering)"
fi
echo "========================================================"

OK=0
SKIP=0

# Effective `d` parameter — zero in UDG mode (no interior deamination)
EFF_D="$BRIGGS_D"
if [ "$UDG" = "true" ]; then
    EFF_D="0.0"
fi

for i in "${!GENOMES[@]}"; do
    genome="${GENOMES[$i]}"
    acc=$(basename "$(dirname "$genome")")

    if $USE_TSV && [ -n "${ACC_SPLIT[$acc]:-}" ]; then
        split="${ACC_SPLIT[$acc]}"
    elif [ "$i" -lt "$TRAIN_END" ]; then split="train"
    elif [ "$i" -lt "$VAL_END"   ]; then split="val"
    else                                 split="test"
    fi

    # Per-genome s — preservation heterogeneity
    g_s=$(sample_per_genome_s "$i" "$BRIGGS_S_MEAN")
    DAMAGE_G="$BRIGGS_V,$BRIGGS_L,$EFF_D,$g_s"

    tmp=$(mktemp -d)

    # Step 1: Fragment the genome into clean reads
    "$GARGAMMEL_BIN/fragSim" \
        -n "$FRAGS_PER_GENOME" \
        --loc "$LOC" \
        --scale "$SCALE" \
        -m "$MIN_LEN" \
        -M "$MAX_LEN" \
        "$genome" > "$tmp/clean.fasta" 2>"$tmp/fragsim.err" || true

    n_frags=$(grep -c "^>" "$tmp/clean.fasta" 2>/dev/null; true)
    n_frags=${n_frags:-0}
    if [ "$n_frags" -eq 0 ]; then
        err=$(cat "$tmp/fragsim.err" 2>/dev/null | head -1)
        echo "  [SKIP] $(basename $genome) — fragSim produced 0 fragments. Error: $err"
        rm -rf "$tmp"
        SKIP=$(( SKIP + 1 ))
        continue
    fi

    # Step 2: Apply deamination damage (per-genome s, optional UDG)
    "$GARGAMMEL_BIN/deamSim" \
        -damage "$DAMAGE_G" \
        "$tmp/clean.fasta" > "$tmp/damaged.fasta" 2>/dev/null || true

    n_dam=$(grep -c "^>" "$tmp/damaged.fasta" 2>/dev/null; true)
    n_dam=${n_dam:-0}
    if [ "$n_dam" -eq 0 ]; then
        echo "  [SKIP] $(basename $genome) — deamSim produced 0 fragments."
        rm -rf "$tmp"
        SKIP=$(( SKIP + 1 ))
        continue
    fi

    # Step 3: Apply Illumina sequencing errors to the damaged reads only.
    # Rate ≈ 0.5%/base.  Done in Python because Gargammel
    # does not provide a sequencer-error step.  Errors are uniform random
    # substitutions among A,C,G,T (deliberately including A↔A no-ops which
    # are filtered).
    if [ "$(awk -v r=$SEQ_ERR_RATE 'BEGIN{print (r>0)?1:0}')" = "1" ]; then
        python3 - "$tmp/damaged.fasta" "$tmp/damaged_err.fasta" "$SEQ_ERR_RATE" "$i" <<'PY'
import sys, random
inp, out, rate, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
rng   = random.Random(seed * 7919 + 13)
ALPH  = 'ACGT'
with open(inp) as fi, open(out, 'w') as fo:
    for line in fi:
        if line.startswith('>'):
            fo.write(line); continue
        seq = list(line.rstrip())
        for j, b in enumerate(seq):
            if b in ALPH and rng.random() < rate:
                # Pick a different base uniformly
                alt = ALPH[(ALPH.index(b) + rng.randint(1, 3)) % 4]
                seq[j] = alt
        fo.write(''.join(seq) + '\n')
PY
        mv "$tmp/damaged_err.fasta" "$tmp/damaged.fasta"
    fi

    cat "$tmp/clean.fasta"   >> "$OUT_DIR/$split/clean.fasta"
    cat "$tmp/damaged.fasta" >> "$OUT_DIR/$split/damaged.fasta"
    rm -rf "$tmp"
    OK=$(( OK + 1 ))

    if (( (i + 1) % 10 == 0 )); then
        echo "  Processed $((i+1))/$N_GENOMES genomes (train/val/test split)..."
    fi
done

echo ""
echo "========================================================"
echo "  Done: $OK genomes simulated, $SKIP skipped"

TOTAL_READS=0
ALL_OK=true
for split in train val test; do
    fasta="$OUT_DIR/$split/clean.fasta"
    if [ -f "$fasta" ]; then
        n=$(grep -c "^>" "$fasta"; true)
        n=${n:-0}
        TOTAL_READS=$(( TOTAL_READS + n ))
        echo "  $split : $n read pairs"
        [ "$n" -eq 0 ] && ALL_OK=false
    else
        echo "  $split : 0 read pairs (no data)"
        ALL_OK=false
    fi
done

echo ""
if [ "$ALL_OK" = true ] && [ "$OK" -gt 0 ] && [ "$TOTAL_READS" -gt 0 ]; then
    echo "  STATUS : SUCCESS — $TOTAL_READS reads across train/val/test"
    echo "  Run next:  python Code/4_build_dataset.py"
else
    echo "  STATUS : FAILED — $OK genomes OK, $SKIP skipped, $TOTAL_READS total reads"
    echo "           Check errors above and re-run."
    exit 1
fi
echo "========================================================"
