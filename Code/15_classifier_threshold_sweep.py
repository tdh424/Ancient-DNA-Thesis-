"""
15_classifier_threshold_sweep.py — Threshold sweep for the aDNA classifier.

For each trained classifier variant, computes precision, recall, F1, MCC and
related metrics across a dense grid of probability thresholds. Equivalent to
Code/8.4_denoiser_threshold_sweep.py but for the read-level classifier.

Reads outputs/results/classifier_probs.npz produced by Code/9_classifier.py.

Outputs:
    outputs/figures/classifier_threshold_sweep.png   four-panel sweep figure
    outputs/results/classifier_threshold_sweep.txt   per-variant tables
"""

from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

OUT_DIR    = Path('outputs')
PROBS_FILE = OUT_DIR / 'results' / 'classifier_probs.npz'

VARIANTS = {
    'probs_seq':      'Seq-only',
    'probs_evo_base': 'Evo2 (per-base)',
    'probs_evo_full': 'Evo2 (per-base + read LL)',
}
COLORS = {
    'Seq-only':                  '#3498db',
    'Evo2 (per-base)':           '#2ecc71',
    'Evo2 (per-base + read LL)': '#1a7a49',
}

GRID = np.linspace(0.02, 0.98, 49)


def metrics_at(probs, labels, t):
    pred = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1   = 2 * prec * sens / max(prec + sens, 1e-9)
    den  = np.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
    mcc  = (float(tp)*float(tn) - float(fp)*float(fn)) / den if den > 0 else 0.0
    return dict(sens=sens, spec=spec, prec=prec, f1=f1, mcc=mcc, fpr=1-spec)


def main():
    if not PROBS_FILE.exists():
        raise SystemExit(f'{PROBS_FILE} not found — train the classifier first.')

    d      = np.load(PROBS_FILE)
    labels = d['labels']
    print(f'Test set: {len(labels):,} reads ({labels.mean()*100:.1f}% ancient)')

    sweeps = {}
    summary_rows = []
    for key, name in VARIANTS.items():
        if key not in d:
            continue
        probs = d[key]
        rows = [{'t': t, **metrics_at(probs, labels, t)} for t in GRID]
        sweeps[name] = rows
        roc  = roc_auc_score(labels, probs)
        pr   = average_precision_score(labels, probs)
        best_f1  = max(rows, key=lambda r: r['f1'])
        best_mcc = max(rows, key=lambda r: r['mcc'])
        summary_rows.append({
            'name': name, 'roc_auc': roc, 'pr_auc': pr,
            'best_f1': best_f1, 'best_mcc': best_mcc,
        })

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Classifier — threshold sweep', fontsize=13, fontweight='bold')

    def plot_metric(ax, key, ylabel):
        for name, rows in sweeps.items():
            ax.plot([r['t'] for r in rows], [r[key] for r in rows],
                    color=COLORS[name], lw=2, label=name)
        ax.set_xlabel('Threshold'); ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.25); ax.legend(fontsize=8)

    plot_metric(axes[0, 0], 'f1',   'F1')
    axes[0, 0].set_title('(A) F1 score')
    plot_metric(axes[0, 1], 'mcc',  'MCC')
    axes[0, 1].set_title('(B) Matthews correlation coefficient')
    plot_metric(axes[1, 0], 'sens', 'Recall (sensitivity)')
    axes[1, 0].set_title('(C) Recall')
    plot_metric(axes[1, 1], 'prec', 'Precision')
    axes[1, 1].set_title('(D) Precision')

    plt.tight_layout()
    out_png = OUT_DIR / 'figures' / 'classifier_threshold_sweep.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure  → {out_png}')

    # ── Text summary ─────────────────────────────────────────────────────────
    lines = ['Classifier threshold sweep', '=' * 60, '']
    for r in summary_rows:
        b1 = r['best_f1']; bm = r['best_mcc']
        lines += [
            f'── {r["name"]} ──',
            f'  ROC-AUC = {r["roc_auc"]:.4f}   PR-AUC = {r["pr_auc"]:.4f}',
            f'  Best F1  : {b1["f1"]:.4f} @ thr {b1["t"]:.2f}  '
            f'(sens {b1["sens"]:.3f}, prec {b1["prec"]:.3f})',
            f'  Best MCC : {bm["mcc"]:.4f} @ thr {bm["t"]:.2f}  '
            f'(sens {bm["sens"]:.3f}, prec {bm["prec"]:.3f})',
            '',
        ]

    out_txt = OUT_DIR / 'results' / 'classifier_threshold_sweep.txt'
    out_txt.write_text('\n'.join(lines))
    print(f'Summary → {out_txt}')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
