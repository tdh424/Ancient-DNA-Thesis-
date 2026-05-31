"""4_build_dataset.py — Encode simulated FASTA pairs into NPZ datasets (train/val/test)."""

from pathlib import Path
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
# RAW_DIR / OUT_DIR / SPLITS_ENV can be overridden via environment variables so
# the same script can build the main dataset and an alternative one (UDG).
import os as _os
RAW_DIR           = Path(_os.environ.get('RAW_DIR', 'data/raw'))
OUT_DIR           = Path(_os.environ.get('OUT_DIR', 'data'))
OUT_SUFFIX        = _os.environ.get('OUT_SUFFIX', '')   # e.g. '_udg'
MIN_LEN           = 30     # discard reads shorter than this
MAX_LEN           = 100    # pad/truncate reads to this length
SPLITS            = _os.environ.get('SPLITS', 'train,val,test').split(',')
DAMAGE_PREVALENCE = 0.25   # fraction of output reads that carry any damage
                           # set to None to keep all reads as simulated

BASE_TO_IDX = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}
# ─────────────────────────────────────────────────────────────────────────────


SOURCE_TO_IDX = {'bacterial': 0, 'human': 1, 'env': 2}
IDX_TO_SOURCE = {v: k for k, v in SOURCE_TO_IDX.items()}


def header_to_source(header):
    """Map FASTA header → source code. HUMAN_/ENV_ prefixes set by 2b_simulate_contam.sh."""
    h = header.lstrip('>')
    if h.startswith('HUMAN_'): return SOURCE_TO_IDX['human']
    if h.startswith('ENV_'):   return SOURCE_TO_IDX['env']
    return SOURCE_TO_IDX['bacterial']


def parse_fasta_with_source(path):
    """Yield (sequence, source_code) pairs from a FASTA file."""
    seq = []; src = SOURCE_TO_IDX['bacterial']
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq:
                    yield ''.join(seq), src
                seq = []
                src = header_to_source(line)
            else:
                seq.append(line.upper())
    if seq:
        yield ''.join(seq), src


def parse_fasta(path):
    for s, _ in parse_fasta_with_source(path):
        yield s


def encode(seq, length):
    """Encode a DNA string to a fixed-length uint8 array (PAD=0)."""
    arr = np.zeros(length, dtype=np.uint8)
    for i, base in enumerate(seq[:length]):
        arr[i] = BASE_TO_IDX.get(base, 5)
    return arr


def process_split(name):
    clean_path   = RAW_DIR / name / 'clean.fasta'
    damaged_path = RAW_DIR / name / 'damaged.fasta'

    if not clean_path.exists() or not damaged_path.exists():
        print(f'  {name}: missing FASTA files in {RAW_DIR}/{name}/ — skipping.')
        print(f'         Run bash Code/3_simulate.sh first.')
        return

    clean_records   = list(parse_fasta_with_source(clean_path))
    damaged_records = list(parse_fasta_with_source(damaged_path))

    if len(clean_records) != len(damaged_records):
        print(f'  {name}: mismatch — {len(clean_records)} clean vs '
              f'{len(damaged_records)} damaged. Truncating.')
        n = min(len(clean_records), len(damaged_records))
        clean_records, damaged_records = clean_records[:n], damaged_records[:n]

    # Encode (track per-read source). Reads shorter than MIN_LEN are discarded;
    # reads longer than MAX_LEN are truncated to the first MAX_LEN bases. Truncation
    # preserves the 5' end (where C->T damage is concentrated) and keeps modern
    # contamination reads in the dataset at their realised length distribution.
    clean_out, damaged_out, lengths_out, source_out = [], [], [], []
    n_truncated = 0
    for (c, src_c), (d, src_d) in zip(clean_records, damaged_records):
        L = len(c)
        if L < MIN_LEN:
            continue
        if L > MAX_LEN:
            c = c[:MAX_LEN]
            d = d[:MAX_LEN]
            L = MAX_LEN
            n_truncated += 1
        clean_out.append(encode(c, MAX_LEN))
        damaged_out.append(encode(d, MAX_LEN))
        lengths_out.append(L)
        source_out.append(src_c)

    if n_truncated > 0:
        print(f'  {name}: truncated {n_truncated:,} reads >{MAX_LEN} bp to '
              f'{MAX_LEN} bp (5\' end retained)')

    if not clean_out:
        print(f'  {name}: no reads passed length filter ({MIN_LEN}–{MAX_LEN} bp).')
        return

    clean_arr   = np.stack(clean_out).astype(np.uint8)
    damaged_arr = np.stack(damaged_out).astype(np.uint8)
    lengths_arr = np.array(lengths_out, dtype=np.int32)
    sources_arr = np.array(source_out, dtype=np.uint8)

    # Source breakdown
    print(f'  {name}: source mix — '
          f'bact {(sources_arr==0).sum():,}  '
          f'human {(sources_arr==1).sum():,}  '
          f'env {(sources_arr==2).sum():,}  '
          f'(total {len(sources_arr):,})')

    # ── Prevalence control (Setup B: realistic mixed library) ────────────────
    # Ancient pool = bacterial reads with damage events (only bacterial reads
    # were exposed to deamSim). Modern pool = everything else: human + env
    # contamination + bacterial reads where Briggs produced no damage.
    #
    # We sub-sample WITHOUT discarding: bacterial reads dropped from the
    # ancient pool are reverted to clean (damaged ← clean) so they remain in
    # the dataset as 'modern bacterial commensal' reads — biologically the
    # right behaviour and keeps total dataset size constant.
    if DAMAGE_PREVALENCE is not None:
        rng = np.random.default_rng(42)
        N   = len(clean_arr)

        is_bact      = (sources_arr == 0)
        has_damage   = (damaged_arr != clean_arr).any(axis=1)
        # Only bacterial reads can be 'ancient'
        ancient_cand = np.where(is_bact & has_damage)[0]
        natural_prev = len(ancient_cand) / max(N, 1)

        n_target_ancient = int(round(N * DAMAGE_PREVALENCE))

        if len(ancient_cand) >= n_target_ancient:
            keep_anc = rng.choice(ancient_cand, size=n_target_ancient, replace=False)
            revert   = np.ones(N, dtype=bool)
            revert[keep_anc] = False
            damaged_arr = damaged_arr.copy()
            damaged_arr[revert] = clean_arr[revert]
        else:
            print(f'  {name}: WARNING — only {len(ancient_cand):,} damaged bacterial '
                  f'reads available, target was {n_target_ancient:,}. '
                  f'Final prevalence will be lower.')

        # Actual realised composition
        is_anc      = (damaged_arr != clean_arr).any(axis=1)
        actual_prev = is_anc.mean()
        modern_idx  = np.where(~is_anc)[0]
        modern_src  = sources_arr[modern_idx]
        print(f'  {name}: natural ancient-pool {natural_prev*100:.1f}%  →  '
              f'{is_anc.sum():,} ancient + {(~is_anc).sum():,} modern = {N:,} '
              f'({actual_prev*100:.1f}% ancient)')
        print(f'           Modern-pool composition: '
              f'human {(modern_src==1).sum():,} ({(modern_src==1).mean()*100:.1f}%)  '
              f'env {(modern_src==2).sum():,} ({(modern_src==2).mean()*100:.1f}%)  '
              f'bact-clean {(modern_src==0).sum():,} ({(modern_src==0).mean()*100:.1f}%)')

    # Damage statistics
    ct_mask = (damaged_arr == 4) & (clean_arr == 2)   # C→T deamination
    ga_mask = (damaged_arr == 1) & (clean_arr == 3)   # G→A deamination
    n_ct    = ct_mask.sum()
    n_ga    = ga_mask.sum()
    n_c     = (clean_arr == 2).sum()
    n_g     = (clean_arr == 3).sum()

    out_path = OUT_DIR / f'{name}{OUT_SUFFIX}.npz'
    np.savez(out_path, clean=clean_arr, damaged=damaged_arr,
             lengths=lengths_arr, sources=sources_arr)

    print(f'  {name}: {len(clean_arr):>8,} reads saved → {out_path}')
    print(f'           C→T damage: {n_ct:,}/{n_c:,} ({100*n_ct/max(n_c,1):.1f}% of C positions)')
    print(f'           G→A damage: {n_ga:,}/{n_g:,} ({100*n_ga/max(n_g,1):.1f}% of G positions)')
    print(f'           Length: {lengths_arr.min()}–{lengths_arr.max()} bp, '
          f'mean {lengths_arr.mean():.0f} bp')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 55)
    print('  Converting FASTA pairs → NPZ datasets')
    print('=' * 55)
    for split in SPLITS:
        process_split(split)
    print()
    print('Done. Run next:')
    print('  sbatch (or python) Code/5.1_evo2_refs.py')


if __name__ == '__main__':
    main()
