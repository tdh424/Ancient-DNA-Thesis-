"""8.2_bayesian_ceiling.py — Bayesian positional ceiling for damage-correction PR-AUC."""

from pathlib import Path
import numpy as np

from sklearn.metrics import average_precision_score

DATA_DIR = Path('outputs/results')
OUT_DIR  = Path('outputs/results')
MAX_LEN  = 100


def log(msg=''):
    print(msg, flush=True)


def main():
    log('Loading test_denoised.npz...')
    d       = np.load(DATA_DIR / 'test_denoised.npz')
    damaged = d['damaged'].astype(np.int32)
    clean   = d['clean'].astype(np.int32)
    lengths = d['lengths'].astype(np.int32)
    N, L    = damaged.shape
    log(f'  {N:,} reads, max length {L}')

    # Mask padded positions
    pos_idx = np.arange(L)[None, :]
    valid   = pos_idx < lengths[:, None]
    damaged = np.where(valid, damaged, 0)
    clean   = np.where(valid, clean,   0)

    # ── Empirical base frequencies from clean reads ───────────────────────────
    clean_valid = clean[valid]
    total = len(clean_valid)
    f = {b: (clean_valid == b).sum() / total for b in range(6)}
    f_C, f_T = f[2], f[4]
    f_G, f_A = f[3], f[1]
    log(f'\nEmpirical base frequencies (clean):')
    log(f'  A={f_A:.4f}  C={f_C:.4f}  G={f_G:.4f}  T={f_T:.4f}')

    # Shared position matrices used throughout
    pos_matrix    = np.tile(np.arange(L), (N, 1))           # (N, L) 5' positions
    dist3p_matrix = (lengths[:, None] - 1 - pos_matrix).clip(0, L - 1)  # 3' distances

    # ── Empirical C→T damage rate per 5' position ─────────────────────────────
    is_clean_C  = (clean == 2) & valid
    is_dmg_T    = (damaged == 4) & (clean == 2) & valid

    ct_total   = np.bincount(pos_matrix[is_clean_C], minlength=L).astype(np.float64)
    ct_damaged = np.bincount(pos_matrix[is_dmg_T],   minlength=L).astype(np.float64)

    damage_rate_ct = np.where(ct_total > 0, ct_damaged / ct_total, 0.0)

    # ── Empirical G→A damage rate per 3' position (vectorised) ───────────────

    is_clean_G  = (clean == 3) & valid
    is_dmg_A    = (damaged == 1) & (clean == 3) & valid

    ga_total   = np.bincount(dist3p_matrix[is_clean_G],  minlength=L).astype(np.float64)
    ga_damaged = np.bincount(dist3p_matrix[is_dmg_A],    minlength=L).astype(np.float64)

    damage_rate_ga = np.where(ga_total > 0, ga_damaged / ga_total, 0.0)

    log('\nEmpirical damage rates (5\' end C→T, first 10 positions):')
    for k in range(10):
        log(f'  pos {k:2d}: rate={damage_rate_ct[k]:.4f}  (n={int(ct_total[k]):,})')

    log('\nEmpirical damage rates (3\' end G→A, first 10 positions from 3\'):')
    for k in range(10):
        log(f'  dist {k:2d}: rate={damage_rate_ga[k]:.4f}  (n={int(ga_total[k]):,})')

    # ── Bayesian scores ───────────────────────────────────────────────────────
    # For each T at position k: P(was C | T, pos=k) = dr[k]*f_C / (f_T + dr[k]*f_C)
    log('\nComputing Bayesian scores...')

    t_mask   = (damaged == 4) & valid
    dmg_mask = t_mask & (clean == 2)

    # Build score array for C→T
    dr_ct_matrix  = damage_rate_ct[pos_matrix]              # (N, L)
    score_ct      = (dr_ct_matrix * f_C) / (f_T + dr_ct_matrix * f_C + 1e-12)

    pc_flat  = score_ct[t_mask]
    lbl_flat = dmg_mask[t_mask].astype(np.int32)
    prauc_ct = average_precision_score(lbl_flat, pc_flat)
    random_ct = lbl_flat.mean()
    log(f'\nC→T Bayesian PR-AUC : {prauc_ct:.4f}  (random baseline: {random_ct:.4f})')
    log(f'C→T NN PR-AUC       : 0.1485')

    # G→A: for each A at distance d3p from 3' end:
    # P(was G | A, dist=d3p) = dr_ga[d3p]*f_G / (f_A + dr_ga[d3p]*f_G)
    a_mask    = (damaged == 1) & valid
    ga_dmg_mask = a_mask & (clean == 3)

    dr_ga_matrix  = damage_rate_ga[dist3p_matrix]
    score_ga      = (dr_ga_matrix * f_G) / (f_A + dr_ga_matrix * f_G + 1e-12)

    pg_flat   = score_ga[a_mask]
    ga_lbl    = ga_dmg_mask[a_mask].astype(np.int32)
    prauc_ga  = average_precision_score(ga_lbl, pg_flat)
    random_ga = ga_lbl.mean()
    log(f'\nG→A Bayesian PR-AUC : {prauc_ga:.4f}  (random baseline: {random_ga:.4f})')
    log(f'G→A NN PR-AUC       : 0.0196')

    combined = 0.5 * (prauc_ct + prauc_ga)
    log(f'\nCombined Bayesian PR-AUC : {combined:.4f}')
    log(f'Combined NN PR-AUC       : 0.0840')

    # ── Save summary ──────────────────────────────────────────────────────────
    lines = [
        'Bayesian Population Ceiling',
        '=' * 50,
        f'Empirical base frequencies (clean):',
        f'  A={f_A:.4f}  C={f_C:.4f}  G={f_G:.4f}  T={f_T:.4f}',
        '',
        f'C→T Bayesian PR-AUC : {prauc_ct:.4f}  (random: {random_ct:.4f})',
        f'C→T NN PR-AUC       : 0.1485',
        f'C→T ratio NN/Bayes  : {0.1485/prauc_ct:.2f}x',
        '',
        f'G→A Bayesian PR-AUC : {prauc_ga:.4f}  (random: {random_ga:.4f})',
        f'G→A NN PR-AUC       : 0.0196',
        '',
        f'Combined Bayesian   : {combined:.4f}',
        f'Combined NN         : 0.0840',
        '',
        'Interpretation:',
        '  Bayesian uses only position-in-read (Briggs decay curve).',
        '  NN additionally uses Evo2 sequence context.',
        '  Gap NN > Bayesian shows value of sequence context.',
    ]
    out_path = OUT_DIR / 'bayesian_ceiling.txt'
    out_path.write_text('\n'.join(lines))
    log(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
