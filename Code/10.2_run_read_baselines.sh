#!/bin/bash
# 10.2_run_read_baselines.sh — Run the PMDtools read-level benchmark on the test split.

set -euo pipefail

GENOME_DIR="data/genomes/ncbi_dataset/data"
RAW_DIR="data/raw/test"
WORK_DIR="data/read_baselines"
SHARED_BAM="data/pydamage/oracle/test_aligned.bam"
THREADS="${THREADS:-4}"
SPLIT_TSV="data/genomes/split_assignment.tsv"

mkdir -p "$WORK_DIR"

# ── Resolve tools ────────────────────────────────────────────────────────────
# Add user-installed tool dirs explicitly (SLURM jobs don't source ~/.bashrc)
export PATH="$HOME/tools/PMDtools:$PATH"

need() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: $1 not on PATH."
        exit 1
    fi
}
need bwa; need samtools; need pmdtools

# ── Get / build BAM ──────────────────────────────────────────────────────────
BAM="$WORK_DIR/test_aligned.bam"
if [ -f "$SHARED_BAM" ]; then
    echo "Reusing alignment from $SHARED_BAM"
    ln -sf "$(realpath "$SHARED_BAM")"     "$BAM"
    ln -sf "$(realpath "$SHARED_BAM").bai" "$BAM.bai"
    REF=$(dirname "$SHARED_BAM")/test_reference.fa
elif [ ! -f "$BAM" ]; then
    REF="$WORK_DIR/test_reference.fa"
    echo "Building concatenated test reference..."
    : > "$REF"
    if [ -f "$SPLIT_TSV" ]; then
        while IFS=$'\t' read -r acc sp _rest; do
            [ "$acc" = "accession" ] && continue
            [ "$sp" != "test" ] && continue
            fna="$GENOME_DIR/$acc/$acc.fna"
            [ -f "$fna" ] && cat "$fna" >> "$REF"
        done < "$SPLIT_TSV"
    else
        N=$(find "$GENOME_DIR" -name "*.fna" | wc -l)
        TEST_START=$(awk "BEGIN{printf \"%d\", $N * 0.90}")
        i=0
        while IFS= read -r fna; do
            if [ "$i" -ge "$TEST_START" ]; then cat "$fna" >> "$REF"; fi
            i=$((i + 1))
        done < <(find "$GENOME_DIR" -name "*.fna" | sort)
    fi
    bwa index "$REF"
    samtools faidx "$REF"
    echo "Aligning damaged.fasta..."
    bwa aln  -l 1024 -n 0.01 -t "$THREADS" "$REF" "$RAW_DIR/damaged.fasta" > "$WORK_DIR/aln.sai"
    bwa samse "$REF" "$WORK_DIR/aln.sai" "$RAW_DIR/damaged.fasta" \
        | samtools sort -@ "$THREADS" -o "$BAM" -
    samtools index "$BAM"
    rm -f "$WORK_DIR/aln.sai"
fi

n_mapped=$(samtools view -c -F 4 "$BAM")
n_total=$(samtools view -c "$BAM")
echo "  BAM contains $n_mapped / $n_total mapped reads."

# ── PMDtools (Skoglund 2014) ─────────────────────────────────────────────────
PMD_OUT="$WORK_DIR/pmdtools_scores.csv"
if [ ! -f "$PMD_OUT" ]; then
    echo "Running PMDtools..."
    # --printDS prints PMD score per read; we capture stdout
    samtools view -h "$BAM" \
      | pmdtools --printDS --requirebaseq 0 2>/dev/null \
      | awk 'BEGIN{print "read_id,pmds"} NF>=4 && $1 !~ /^@/ {print $1","$4}' \
      > "$PMD_OUT"
    echo "  Wrote $PMD_OUT ($(($(wc -l < "$PMD_OUT") - 1)) reads)"
fi

echo ""
echo "Done. Now run:"
echo "  python Code/11_baselines_compare.py \\"
echo "      --pmdtools $PMD_OUT \\"
echo "      --pydamage data/pydamage/oracle/results/pydamage_results.csv"
