"""
8.4_cds_per_threshold.py — CDS proxy metrics across operating thresholds.

For each denoiser variant (seq_only, evo2, bwa), compute:
  - mean stop codons per read
  - C-recovery accuracy at true-C positions
at multiple thresholds (default 0.5 plus F1- and MCC-optimal points).

Reads outputs/results/test_denoised_<variant>.npz files (they contain
prob_c, prob_g) and re-applies the threshold locally instead of using
the stored denoised array (which was thresholded at 0.5 by 7_denoise.py).

Output: outputs/results/cds_per_threshold.txt
"""
from pathlib import Path
import numpy as np

RES = Path('outputs/results')
A, C, G, T = 1, 2, 3, 4
STOP_CODONS = {(T, A, A), (T, A, G), (T, G, A)}

# Thresholds to evaluate, per variant (variant -> list of (label, threshold))
THRESHOLDS = {
    'seq_only': [('default', 0.50), ('F1-opt', 0.32), ('MCC-opt', 0.26)],
    'evo2':     [('default', 0.50), ('F1-opt', 0.34), ('MCC-opt', 0.30)],
    'bwa':      [('default', 0.50), ('F1-opt', 0.33), ('MCC-opt', 0.27)],
}


def count_stop_codons(reads):
    N, L = reads.shape
    total = 0
    for frame in range(3):
        for k in range(frame, L - 2, 3):
            c0, c1, c2 = reads[:, k], reads[:, k+1], reads[:, k+2]
            for s in STOP_CODONS:
                total += int(((c0 == s[0]) & (c1 == s[1]) & (c2 == s[2])).sum())
    return total / N


def c_recovery(reads, clean):
    mask = (clean == C)
    return float((reads[mask] == C).mean()) if mask.sum() > 0 else float('nan')


def apply_threshold(damaged, prob_c, prob_g, tau):
    den = damaged.copy()
    den[(damaged == T) & (prob_c > tau)] = C
    if prob_g is not None:
        den[(damaged == A) & (prob_g > tau)] = G
    return den


def main():
    rows = []
    # ── Baselines: damaged and clean ─────────────────────────────────────────
    # Use whichever variant npz we have access to for the baselines
    first_npz = next(p for p in [RES / f'test_denoised_{v}.npz'
                                  for v in THRESHOLDS] if p.exists())
    d = np.load(first_npz)
    damaged = d['damaged'].astype(np.int32)
    clean   = d['clean'].astype(np.int32)
    lengths = d['lengths'].astype(np.int32)
    L = damaged.shape[1]
    pos = np.arange(L)[None, :]
    valid = pos < lengths[:, None]
    damaged = np.where(valid, damaged, 0)
    clean   = np.where(valid, clean,   0)

    stops_dmg = count_stop_codons(damaged)
    stops_cln = count_stop_codons(clean)
    acc_dmg = c_recovery(damaged, clean)
    acc_cln = c_recovery(clean, clean)

    rows.append(('damaged (no correction)', '—', '—', stops_dmg, 0.0, acc_dmg * 100))
    print(f'Damaged baseline: stops/read = {stops_dmg:.4f}  '
          f'C-recovery = {acc_dmg*100:.2f}%')
    print(f'Clean baseline:   stops/read = {stops_cln:.4f}  '
          f'C-recovery = {acc_cln*100:.2f}%')

    gap = stops_dmg - stops_cln  # damaged→clean gap in stop codons

    # ── Per-variant per-threshold sweep ──────────────────────────────────────
    for variant, thresholds in THRESHOLDS.items():
        npz_path = RES / f'test_denoised_{variant}.npz'
        if not npz_path.exists():
            print(f'  [skip] {variant}: {npz_path} not found')
            continue
        d = np.load(npz_path)
        damaged = np.where(valid, d['damaged'].astype(np.int32), 0)
        clean_v = np.where(valid, d['clean'].astype(np.int32), 0)
        prob_c = np.where(valid, d['prob_c'].astype(np.float32), 0.0)
        prob_g = (np.where(valid, d['prob_g'].astype(np.float32), 0.0)
                  if 'prob_g' in d else None)

        for label, tau in thresholds:
            den = apply_threshold(damaged, prob_c, prob_g, tau)
            stops = count_stop_codons(den)
            acc = c_recovery(den, clean_v)
            closed = (stops_dmg - stops) / gap * 100 if gap > 0 else float('nan')
            rows.append((variant, label, f'{tau:.2f}', stops, closed, acc * 100))
            print(f'  {variant:<10}  {label:<8}  tau={tau:.2f}  '
                  f'stops={stops:.4f}  gap_closed={closed:>5.1f}%  '
                  f'C-acc={acc*100:.2f}%')

    rows.append(('clean (upper bound)', '—', '—', stops_cln, 100.0, acc_cln * 100))

    # ── Write summary ────────────────────────────────────────────────────────
    out = ['CDS proxy metrics across denoiser operating thresholds',
           '=' * 78,
           f'  {"Variant":<24}  {"Pt":<8}  {"tau":>5}  {"Stops/read":>10}  '
           f'{"GapClosed":>10}  {"C-recovery":>11}',
           '  ' + '-' * 76]
    for variant, label, tau_s, stops, closed, acc in rows:
        cl = f'{closed:>9.1f}%' if isinstance(closed, float) else f'{closed:>10}'
        out.append(f'  {variant:<24}  {label:<8}  {tau_s:>5}  '
                   f'{stops:>10.4f}  {cl}  {acc:>10.2f}%')
    text = '\n'.join(out)
    (RES / 'cds_per_threshold.txt').write_text(text)
    print('\n' + text)
    print(f'\nSummary → {RES / "cds_per_threshold.txt"}')


if __name__ == '__main__':
    main()
