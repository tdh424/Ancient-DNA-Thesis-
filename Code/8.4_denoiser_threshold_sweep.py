"""8.4_denoiser_threshold_sweep.py — Threshold sweep across all denoiser variants."""

from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR  = Path('outputs')
RESULTS  = OUT_DIR / 'results'
FIGURES  = OUT_DIR / 'figures'

VARIANTS = ['seq_only', 'evo2', 'bwa']
DISPLAY  = {'seq_only': 'Seq-only', 'evo2': 'Evo2 (soft ref)', 'bwa': 'BWA (hard ref)'}
COLORS   = {'seq_only': '#3498db', 'evo2': '#2ecc71', 'bwa': '#e74c3c'}

# Encoding: PAD=0 A=1 C=2 G=3 T=4 N=5
A, C, G, T = 1, 2, 3, 4

# Threshold grid — fine for plot, coarse for table
GRID_FINE  = np.linspace(0.01, 0.99, 99)
GRID_TABLE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
              0.50, 0.60, 0.70, 0.80, 0.90]


def metrics_at_threshold(prob_c, prob_g, damaged, clean, lengths, t):
    """
    Compute precision/recall/F1/MCC at threshold t for both C→T and G→A,
    plus combined.
    Returns: dict with sens_ct, prec_ct, sens_ga, prec_ga, sens, prec, f1, mcc, n_corr
    """
    L      = damaged.shape[1]
    pos    = np.arange(L)[None, :]
    valid  = pos < lengths[:, None]
    is_T   = (damaged == T) & valid
    is_A   = (damaged == A) & valid
    true_ct = is_T & (clean == C)
    true_ga = is_A & (clean == G)

    flag_ct = is_T & (prob_c > t)
    flag_ga = is_A & (prob_g > t)

    tp_ct = (flag_ct & true_ct).sum()
    fp_ct = flag_ct.sum() - tp_ct
    fn_ct = true_ct.sum() - tp_ct
    tp_ga = (flag_ga & true_ga).sum()
    fp_ga = flag_ga.sum() - tp_ga
    fn_ga = true_ga.sum() - tp_ga

    # Combined: any damage event flagged correctly
    tp = tp_ct + tp_ga
    fp = fp_ct + fp_ga
    fn = fn_ct + fn_ga
    # TN: positions that are T or A AND original was T or A AND not flagged
    candidates = is_T.sum() + is_A.sum()
    tn = candidates - tp - fp - fn

    sens   = tp / max(tp + fn, 1)
    prec   = tp / max(tp + fp, 1)
    f1     = 2 * prec * sens / max(prec + sens, 1e-9)
    denom  = np.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
    mcc    = (float(tp)*float(tn) - float(fp)*float(fn)) / denom if denom > 0 else 0.0

    sens_ct = tp_ct / max(tp_ct + fn_ct, 1)
    prec_ct = tp_ct / max(tp_ct + fp_ct, 1)
    sens_ga = tp_ga / max(tp_ga + fn_ga, 1)
    prec_ga = tp_ga / max(tp_ga + fp_ga, 1)

    return dict(
        sens_ct=sens_ct, prec_ct=prec_ct,
        sens_ga=sens_ga, prec_ga=prec_ga,
        sens=sens, prec=prec, f1=f1, mcc=mcc,
        n_corr=int(flag_ct.sum() + flag_ga.sum()),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


def sweep(variant, grid):
    path = RESULTS / f'test_denoised_{variant}.npz'
    if not path.exists():
        print(f'  [SKIP] {path} not found')
        return None
    d = np.load(path)
    prob_c, prob_g = d['prob_c'], d['prob_g']
    damaged, clean, lengths = d['damaged'], d['clean'], d['lengths']

    rows = []
    for t in grid:
        m = metrics_at_threshold(prob_c, prob_g, damaged, clean, lengths, t)
        m['threshold'] = float(t)
        rows.append(m)
    return rows


def find_best(rows, key):
    best = max(rows, key=lambda r: r[key])
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print('Computing threshold sweeps...')
    fine = {}
    for v in VARIANTS:
        rows = sweep(v, GRID_FINE)
        if rows is not None:
            fine[v] = rows
            best_f1  = find_best(rows, 'f1')
            best_mcc = find_best(rows, 'mcc')
            print(f'  {v:10s}  best F1 = {best_f1["f1"]:.3f} @ thr={best_f1["threshold"]:.2f}  |  '
                  f'best MCC = {best_mcc["mcc"]:.3f} @ thr={best_mcc["threshold"]:.2f}')

    if not fine:
        print('No denoised NPZ files found. Run 7_denoise.py first.')
        return

    # ── Figure: 4-panel sweep ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Denoiser Performance vs Threshold  (per-model sweep)',
                 fontsize=14, fontweight='bold')

    # Helper to extract a metric series
    def series(rows, key):
        return [r[key] for r in rows]

    # Panel A — Precision-Recall curves (parametric in threshold)
    ax = axes[0, 0]
    for v in VARIANTS:
        if v not in fine: continue
        rows = fine[v]
        ax.plot(series(rows, 'sens'), series(rows, 'prec'),
                color=COLORS[v], lw=2, label=DISPLAY[v])
        # Mark optimal F1 point
        best = find_best(rows, 'f1')
        ax.scatter([best['sens']], [best['prec']],
                   color=COLORS[v], s=80, zorder=5, edgecolors='black')
        ax.annotate(f'F1={best["f1"]:.2f}\n@t={best["threshold"]:.2f}',
                    xy=(best['sens'], best['prec']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color=COLORS[v])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('(A) Precision-Recall (combined C→T + G→A)', fontsize=11)
    ax.legend(); ax.grid(alpha=0.25); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel B — F1 vs threshold
    ax = axes[0, 1]
    for v in VARIANTS:
        if v not in fine: continue
        rows = fine[v]
        ax.plot(series(rows, 'threshold'), series(rows, 'f1'),
                color=COLORS[v], lw=2, label=DISPLAY[v])
        best = find_best(rows, 'f1')
        ax.axvline(best['threshold'], color=COLORS[v], ls=':', alpha=0.6)
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('F1 score', fontsize=11)
    ax.set_title('(B) F1 vs threshold', fontsize=11)
    ax.legend(); ax.grid(alpha=0.25); ax.set_xlim(0, 1)

    # Panel C — Recall vs threshold
    ax = axes[1, 0]
    for v in VARIANTS:
        if v not in fine: continue
        rows = fine[v]
        ax.plot(series(rows, 'threshold'), series(rows, 'sens'),
                color=COLORS[v], lw=2, label=f'{DISPLAY[v]} (combined)')
        ax.plot(series(rows, 'threshold'), series(rows, 'sens_ct'),
                color=COLORS[v], lw=1, ls='--', alpha=0.6, label=f'{DISPLAY[v]} C→T')
        ax.plot(series(rows, 'threshold'), series(rows, 'sens_ga'),
                color=COLORS[v], lw=1, ls=':', alpha=0.6, label=f'{DISPLAY[v]} G→A')
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Recall', fontsize=11)
    ax.set_title('(C) Recall vs threshold (split by damage type)', fontsize=11)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel D — Precision vs threshold
    ax = axes[1, 1]
    for v in VARIANTS:
        if v not in fine: continue
        rows = fine[v]
        ax.plot(series(rows, 'threshold'), series(rows, 'prec'),
                color=COLORS[v], lw=2, label=DISPLAY[v])
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('(D) Precision vs threshold', fontsize=11)
    ax.legend(); ax.grid(alpha=0.25); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    out_png = FIGURES / 'denoiser_threshold_sweep.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure → {out_png}')

    # ── Text summary ─────────────────────────────────────────────────────────
    lines = [
        'Denoiser threshold sweep — performance at multiple operating points',
        '=' * 78,
        '',
    ]
    for v in VARIANTS:
        if v not in fine: continue
        lines.append(f'── {DISPLAY[v]} ───────────────────────────────────────────────')

        rows_table = sweep(v, GRID_TABLE)
        lines.append(f'  {"Thr":>5}  {"Corr":>10}  {"Recall":>7}  {"Prec":>7}  '
                     f'{"F1":>6}  {"MCC":>6}  {"R(C→T)":>7}  {"R(G→A)":>7}')
        lines.append('  ' + '-' * 78)
        for r in rows_table:
            lines.append(
                f'  {r["threshold"]:.2f}  {r["n_corr"]:>10,}  '
                f'{r["sens"]*100:>6.1f}%  {r["prec"]*100:>6.1f}%  '
                f'{r["f1"]:>6.3f}  {r["mcc"]:>6.3f}  '
                f'{r["sens_ct"]*100:>6.1f}%  {r["sens_ga"]*100:>6.1f}%'
            )

        rows_fine = fine[v]
        best_f1  = find_best(rows_fine, 'f1')
        best_mcc = find_best(rows_fine, 'mcc')
        lines += [
            '',
            f'  Best F1    = {best_f1["f1"]:.4f} at threshold {best_f1["threshold"]:.3f}  '
            f'(recall {best_f1["sens"]*100:.1f}%, precision {best_f1["prec"]*100:.1f}%)',
            f'  Best MCC   = {best_mcc["mcc"]:.4f} at threshold {best_mcc["threshold"]:.3f}',
            '',
        ]

    # Cross-model summary
    lines += ['═' * 78,
              'Cross-model summary at each model\'s F1-optimal threshold:',
              f'  {"Model":<22}  {"Thr":>5}  {"Recall":>7}  {"Prec":>7}  {"F1":>6}',
              '  ' + '-' * 50]
    for v in VARIANTS:
        if v not in fine: continue
        b = find_best(fine[v], 'f1')
        lines.append(f'  {DISPLAY[v]:<20}  {b["threshold"]:>5.2f}  '
                     f'{b["sens"]*100:>6.1f}%  {b["prec"]*100:>6.1f}%  '
                     f'{b["f1"]:>6.3f}')

    txt = '\n'.join(lines)
    out_txt = RESULTS / 'denoiser_threshold_sweep.txt'
    out_txt.write_text(txt)
    print(f'Summary → {out_txt}')
    print()
    print(txt)


if __name__ == '__main__':
    main()
