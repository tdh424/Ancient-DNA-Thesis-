#!/bin/bash
# 10.1_run_pydamage.sh — Run the pyDamage benchmark on the test split (MODE=oracle or denovo).

set -euo pipefail

MODE="${MODE:-oracle}"
GENOME_DIR="data/genomes/ncbi_dataset/data"
RAW_DIR="data/raw/test"
WORK_DIR="data/pydamage/${MODE}"
THREADS="${THREADS:-4}"
SPLIT_TSV="data/genomes/split_assignment.tsv"

mkdir -p "$WORK_DIR/results"

for tool in bwa samtools pydamage; do
    if ! command -v "$tool" &>/dev/null; then
        echo "ERROR: $tool not on PATH."
        exit 1
    fi
done

# ── Choose reference: oracle (source genomes) or denovo (MEGAHIT) ────────────
REF="$WORK_DIR/reference.fa"

if [ "$MODE" = "oracle" ]; then
    if [ ! -f "${REF}.bwt" ]; then
        echo "Building oracle reference from source genomes (test split)..."
        : > "$REF"
        if [ -f "$SPLIT_TSV" ]; then
            while IFS=$'\t' read -r acc sp _rest; do
                [ "$acc" = "accession" ] && continue
                [ "$sp" != "test" ] && continue
                fna="$GENOME_DIR/$acc/$acc.fna"
                [ -f "$fna" ] && cat "$fna" >> "$REF"
            done < "$SPLIT_TSV"
        else
            echo "ERROR: $SPLIT_TSV missing — required for the oracle reference."
            exit 1
        fi
        n_seq=$(grep -c "^>" "$REF" || true)
        echo "  Oracle reference contigs: $n_seq"
        bwa index "$REF"
        samtools faidx "$REF"
    fi

elif [ "$MODE" = "denovo" ]; then
    if ! command -v metaspades.py &>/dev/null && ! command -v spades.py &>/dev/null; then
        echo "ERROR: metaspades.py / spades.py not on PATH. Install via:"
        echo "    conda install -n ancient-dna -c bioconda spades"
        exit 1
    fi
    SPADES_BIN=$(command -v metaspades.py || command -v spades.py)
    if [ ! -f "${REF}.bwt" ]; then
        # Match Borry 2021: MetaSPAdes with k = 21, 33, 45 (k-mers tuned for
        # short ancient DNA molecules), then keep contigs ≥ 1000 bp.
        echo "Running de novo assembly with MetaSPAdes (k=21,33,45)..."
        ASSEMBLY_DIR="$WORK_DIR/assembly"
        rm -rf "$ASSEMBLY_DIR"
        "$SPADES_BIN" --meta -s "$RAW_DIR/damaged.fasta" \
                -o "$ASSEMBLY_DIR" -t "$THREADS" \
                -k 21,33,45 --only-assembler 2>&1 | tail -30
        # MetaSPAdes writes to contigs.fasta or scaffolds.fasta; pick contigs.
        RAW_CONTIGS="$ASSEMBLY_DIR/contigs.fasta"
        if [ ! -f "$RAW_CONTIGS" ]; then
            echo "ERROR: MetaSPAdes did not produce contigs.fasta."
            exit 1
        fi
        # Filter to contigs ≥ 1000 bp (Borry 2021 threshold for downstream
        # damage analysis).
        python3 -c "
from Bio import SeqIO
kept = [r for r in SeqIO.parse('$RAW_CONTIGS', 'fasta') if len(r.seq) >= 1000]
SeqIO.write(kept, '$REF', 'fasta')
print(f'  Kept {len(kept)} contigs ≥ 1000 bp')
"
        n_seq=$(grep -c "^>" "$REF" || true)
        if [ "$n_seq" -eq 0 ]; then
            echo "ERROR: 0 contigs ≥ 1000 bp after filtering."
            exit 1
        fi
        bwa index "$REF"
        samtools faidx "$REF"
    fi

else
    echo "ERROR: unknown MODE=$MODE. Use 'oracle' or 'denovo'."
    exit 1
fi

# ── Align test reads to the chosen reference ─────────────────────────────────
# BWA aln parameters match Borry 2021 (and the Skoglund 2014 setup for
# PMDtools): -n 0.01 (max edit-distance fraction), -o 2 (max gap opens),
# -l 16500 (effectively disables seeding, since aDNA reads are too short
# and damage-prone for BWA's default seed).
BAM="$WORK_DIR/test_aligned.bam"
if [ ! -f "$BAM" ]; then
    echo "Aligning $RAW_DIR/damaged.fasta..."
    bwa aln  -n 0.01 -o 2 -l 16500 -t "$THREADS" \
        "$REF" "$RAW_DIR/damaged.fasta" > "$WORK_DIR/aln.sai"
    bwa samse "$REF" "$WORK_DIR/aln.sai" "$RAW_DIR/damaged.fasta" \
        | samtools sort -@ "$THREADS" -o "$BAM" -
    samtools index "$BAM"
    rm -f "$WORK_DIR/aln.sai"
    n_mapped=$(samtools view -c -F 4 "$BAM")
    n_total=$(samtools view -c "$BAM")
    echo "  Mapped: $n_mapped / $n_total"
fi

# ── Run pyDamage ─────────────────────────────────────────────────────────────
RESULTS="$WORK_DIR/results/pydamage_results.csv"
if [ ! -f "$RESULTS" ]; then
    echo "Running pyDamage (mode=$MODE)..."
    rm -rf "$WORK_DIR/results"
    pushd "$WORK_DIR" >/dev/null
    yes y | pydamage --outdir results analyze test_aligned.bam || true
    popd >/dev/null
    [ -f "$RESULTS" ] || { echo "ERROR: pyDamage did not produce $RESULTS"; exit 1; }
fi

echo ""
echo "Done ($MODE). pyDamage results: $RESULTS"
echo "To compare against ML models:"
echo "  python Code/11_baselines_compare.py --pydamage $RESULTS"
