"""5.2_bwa_refs.py — Add BWA per-split reference bases to the NPZ datasets."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import pysam
except ImportError:
    sys.exit('pysam not found. Run: pip install pysam (inside conda env)')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path('data')
GENOME_DIR   = DATA_DIR / 'genomes' / 'ncbi_dataset' / 'data'
SPLITS       = ['train', 'val', 'test']
MAX_LEN      = 100

# Same fractions as 2_simulate.sh
TRAIN_FRAC   = 0.80
VAL_END_FRAC = 0.90

# BWA aln tuned for ~30-100 bp aDNA reads with Briggs damage.
#   - n=0.04: max edit distance per read. Empirically n=0.10 gave a LOWER
#     map rate than n=0.04 on this dataset (5.3% vs 6.6%), likely because
#     BWA's internal candidate-pruning becomes more aggressive when more
#     mismatches are allowed, so we keep the tighter setting.
#   - l=1024: disable seed (required for short reads with terminal damage).
#   - o=2:    max gap opens (unchanged).
#   - MAPQ_MIN=20: post-alignment filter; drops multi-mappers with ambiguous
#     placement. MAPQ=0 in bwa aln means "equally good alignment elsewhere",
#     which would give the model an arbitrary reference base. Empirically
#     this drops ~13% of aligned reads on the test set, leaving high-MAPQ
#     calls only.
BWA_N        = 0.04
BWA_O        = 2
BWA_L        = 1024
MAPQ_MIN     = 20
N_THREADS    = int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))

# Project-local tmp dir — cluster /tmp is often too small for 5+ GB refs
TMP_ROOT     = Path('outputs/tmp/bwa')

# Vocab: PAD=0 A=1 C=2 G=3 T=4 N=5
IDX_TO_BASE  = {0: 'N', 1: 'A', 2: 'C', 3: 'G', 4: 'T', 5: 'N'}
BASE_TO_IDX  = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
COMPLEMENT   = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
# ─────────────────────────────────────────────────────────────────────────────


def log(msg=''):
    print(msg, flush=True)


def check_dep(cmd):
    try:
        subprocess.run([cmd, '--version'], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def run(cmd, desc=''):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f'  ERROR ({desc}):\n{r.stderr[:600]}')
        sys.exit(1)
    return r


def decode(arr, length):
    return ''.join(IDX_TO_BASE.get(int(b), 'N') for b in arr[:length])


# ── Genome → split assignment (must mirror 2_simulate.sh exactly) ────────────

def split_genomes():
    all_g = sorted(GENOME_DIR.rglob('*.fna'))
    N     = len(all_g)
    if N == 0:
        sys.exit(f'No .fna files found under {GENOME_DIR}')
    train_end = int(N * TRAIN_FRAC)
    val_end   = int(N * VAL_END_FRAC)
    return {
        'train': all_g[:train_end],
        'val':   all_g[train_end:val_end],
        'test':  all_g[val_end:],
    }


# ── Per-split alignment ───────────────────────────────────────────────────────

def build_split_reference(genomes, ref_path):
    log(f'    Concatenating {len(genomes)} genomes → {ref_path.name}...')
    with open(ref_path, 'wb') as out:
        for g in genomes:
            with open(g, 'rb') as f:
                shutil.copyfileobj(f, out, length=4 * 1024 * 1024)
    sz = ref_path.stat().st_size / 1e9
    log(f'    Reference size: {sz:.2f} GB')
    log('    Building BWA index (this can take ~15-30 min for multi-GB refs)...')
    run(['bwa', 'index', str(ref_path)], 'bwa index')


def write_fastq(damaged_arr, lengths, fastq_path):
    log(f'    Writing {len(damaged_arr):,} reads as FASTQ...')
    with open(fastq_path, 'w') as f:
        for i in range(len(damaged_arr)):
            L   = int(lengths[i])
            seq = decode(damaged_arr[i], L)
            f.write(f'@r{i:08d}\n{seq}\n+\n{"I"*L}\n')


def extract_ref_bases(bam_path, N, lengths):
    """Parse BAM → ref_bwa (N, MAX_LEN). Reverse-strand alignments are flipped
    and complemented so positions are always in original read order.

    Reads with MAPQ < MAPQ_MIN are treated as unmapped (their ref_bwa stays
    all-zero), because bwa aln reports MAPQ=0 for reads that map equally well
    to multiple positions in the reference, and an arbitrarily-chosen reference
    base is worse than no reference at all."""
    ref_bwa     = np.zeros((N, MAX_LEN), dtype=np.uint8)
    n_mapped    = 0
    n_unmapped  = 0
    n_lowmapq   = 0
    mapq_hist   = {}
    bam = pysam.AlignmentFile(str(bam_path), 'rb')
    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            n_unmapped += 1
            continue
        mapq = int(read.mapping_quality)
        mapq_hist[mapq] = mapq_hist.get(mapq, 0) + 1
        if mapq < MAPQ_MIN:
            n_lowmapq += 1
            continue
        idx = int(read.query_name[1:])
        L = int(lengths[idx])
        bases = np.zeros(MAX_LEN, dtype=np.uint8)
        for qpos, _, ref_base in read.get_aligned_pairs(with_seq=True):
            if qpos is None or ref_base is None:
                continue
            base_char = ref_base.upper()
            if read.is_reverse:
                orig_pos  = L - 1 - qpos
                base_char = COMPLEMENT.get(base_char, 'N')
            else:
                orig_pos = qpos
            if 0 <= orig_pos < MAX_LEN:
                bases[orig_pos] = BASE_TO_IDX.get(base_char, 0)
        ref_bwa[idx] = bases
        n_mapped += 1
    bam.close()
    return ref_bwa, n_mapped, n_unmapped, n_lowmapq, mapq_hist


def align_split(name, genomes_for_split):
    path = DATA_DIR / f'{name}.npz'
    if not path.exists():
        log(f'  {name}: {path} not found — skipping.')
        return

    d = np.load(path)
    if 'ref_bwa' in d and os.environ.get('FORCE_BWA', '').lower() not in ('1', 'true', 'yes'):
        log(f'  {name}: ref_bwa already present — skipping (set FORCE_BWA=1 to re-run).')
        return

    damaged = d['damaged']
    lengths = d['lengths'].astype(np.int32)
    N       = len(damaged)
    log(f'  {name}: {N:,} reads,  {len(genomes_for_split)} reference genomes')

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmp        = Path(tempfile.mkdtemp(prefix=f'{name}_', dir=str(TMP_ROOT)))
    ref_path   = tmp / 'ref.fa'
    fastq_path = tmp / 'reads.fastq'

    try:
        build_split_reference(genomes_for_split, ref_path)
        write_fastq(damaged, lengths, fastq_path)

        log(f'    Running bwa aln (-l {BWA_L} -n {BWA_N} -o {BWA_O}, {N_THREADS} threads)...')
        sai = tmp / 'aln.sai'
        run(['bwa', 'aln',
             '-n', str(BWA_N), '-o', str(BWA_O), '-l', str(BWA_L),
             '-t', str(N_THREADS),
             str(ref_path), str(fastq_path),
             '-f', str(sai)], 'bwa aln')

        log('    Running bwa samse...')
        sam = tmp / 'aln.sam'
        run(['bwa', 'samse',
             str(ref_path), str(sai), str(fastq_path),
             '-f', str(sam)], 'bwa samse')

        log('    Sorting + indexing BAM...')
        bam = tmp / 'aln.bam'
        srt = tmp / 'aln_sorted.bam'
        run(['samtools', 'view', '-bS', '-@', str(N_THREADS), str(sam), '-o', str(bam)],
            'sam→bam')
        run(['samtools', 'sort', '-@', str(N_THREADS), str(bam), '-o', str(srt)],
            'sort')
        run(['samtools', 'index', str(srt)], 'index')

        log('    Extracting reference bases at aligned positions...')
        ref_bwa, n_mapped, n_unmapped, n_lowmapq, mapq_hist = extract_ref_bases(srt, N, lengths)
        n_aligned_any = n_mapped + n_lowmapq
        log(f'    Aligned (any MAPQ):  {n_aligned_any:,} / {N:,}  ({100*n_aligned_any/N:.1f}%)')
        log(f'    Kept (MAPQ >= {MAPQ_MIN}): {n_mapped:,} / {N:,}  ({100*n_mapped/N:.1f}%)')
        log(f'    Dropped (low MAPQ):  {n_lowmapq:,}  ({100*n_lowmapq/max(n_aligned_any,1):.1f}% of aligned)')
        log(f'    Unmapped:            {n_unmapped:,}')
        # Compact MAPQ histogram
        if mapq_hist:
            buckets = [(0,0), (1,9), (10,19), (20,29), (30,59), (60,60)]
            log('    MAPQ distribution (aligned reads):')
            for lo, hi in buckets:
                cnt = sum(v for k, v in mapq_hist.items() if lo <= k <= hi)
                if cnt:
                    log(f'      MAPQ {lo:>2}-{hi:<2}: {cnt:,}  ({100*cnt/n_aligned_any:.1f}%)')

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    bak = path.with_suffix('.npz.bak_bwa')
    if not bak.exists():
        shutil.copy(path, bak)
    arrays = dict(d)
    arrays['ref_bwa'] = ref_bwa
    np.savez_compressed(path, **arrays)
    log(f'    Saved {path.name}  (added ref_bwa, shape={ref_bwa.shape})')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='BWA reference precomputation')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'],
                        default='all',
                        help='Process only one split (for parallel SLURM jobs).')
    args = parser.parse_args()

    for dep in ['bwa', 'samtools']:
        if not check_dep(dep):
            sys.exit(f'Missing: {dep}. Load the module or add to PATH.')

    log('BWA reference precomputation — realistic per-split alignment')
    log('=' * 60)
    log(f'Params : -n {BWA_N}, -o {BWA_O}, -l {BWA_L}, MAPQ_MIN={MAPQ_MIN}')
    log(f'Threads: {N_THREADS}')

    splits = split_genomes()
    for name, genomes in splits.items():
        log(f'  {name:<6}: {len(genomes)} genomes')
    log('')

    todo = SPLITS if args.split == 'all' else [args.split]
    for name in todo:
        log(f'── Processing {name} split ──')
        align_split(name, splits[name])

    log('\nDone. Run: python Code/6_train_denoiser.py --variant bwa')


if __name__ == '__main__':
    main()
