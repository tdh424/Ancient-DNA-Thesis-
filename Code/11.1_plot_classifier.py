"""11.1_plot_classifier.py — Re-plot classifier results from saved probabilities."""

from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)

OUT_DIR    = Path('outputs')
PROBS_FILE = OUT_DIR / 'results' / 'classifier_probs.npz'
DATA_DIR   = Path('data')

# Keys in classifier_probs.npz → display names
PROB_KEYS = {
    'probs_evo_full': 'Evo2 (per-base + read LL)',
    'probs_evo_base': 'Evo2 (per-base)',
    'probs_seq':      'Seq-only',
}

# Briggs LLR parameters — must match Code/3_simulate.sh and Code/11_baselines_compare.py
# (Briggs et al. 2007, Vi-33.16 Neanderthal fit).
BRIGGS_S = 0.6815; BRIGGS_L = 0.3590; BRIGGS_D = 0.00937; SEQ_ERR = 0.01

COLORS = {
    'Evo2 (per-base + read LL)': '#1a7a49',
    'Evo2 (per-base)':           '#2ecc71',
    'Seq-only':                  '#3498db',
    'Briggs LLR':                '#e74c3c',
}

LINE_STYLES = {
    'Evo2 (per-base + read LL)': '-',
    'Evo2 (per-base)':           '--',
    'Seq-only':                  '-.',
    'Briggs LLR':                ':',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def youden_threshold(fpr_arr, tpr_arr, thresholds):
    return float(thresholds[np.argmax(tpr_arr - fpr_arr)])


def best_f1_threshold(probs, labels):
    prec, rec, thr = precision_recall_curve(labels, probs)
    with np.errstate(invalid='ignore'):
        f1 = 2 * prec * rec / (prec + rec)
    return float(thr[np.argmax(np.nan_to_num(f1[:-1]))])


def best_mcc_threshold(probs, labels, n_steps=201):
    thresholds = np.linspace(0.01, 0.99, n_steps)
    mccs = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
        denom = np.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
        mccs.append((float(tp)*float(tn) - float(fp)*float(fn)) / denom if denom > 0 else 0.0)
    return float(thresholds[np.argmax(mccs)])


def stats_at(probs, labels, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    n    = tn + fp + fn + tp
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1   = 2*ppv*sens / (ppv+sens) if (ppv+sens) > 0 else 0.0
    acc  = (tp + tn) / n
    bal  = (sens + spec) / 2
    fpr  = 1 - spec
    denom = np.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
    mcc   = (float(tp)*float(tn) - float(fp)*float(fn)) / denom if denom > 0 else 0.0
    return dict(sens=sens, spec=spec, fpr=fpr, ppv=ppv, npv=npv,
                f1=f1, acc=acc, bal_acc=bal, mcc=mcc,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def compute_briggs_scores(damaged_arr, lengths_arr):
    N, MAX_LEN = damaged_arr.shape
    pos    = np.arange(MAX_LEN)
    scores = np.zeros(N, dtype=np.float64)
    for k in range(MAX_LEN):
        p_ct  = BRIGGS_S * (BRIGGS_L ** k) + BRIGGS_D
        valid = pos[k] < lengths_arr
        p_anc = p_ct + (1 - p_ct) * SEQ_ERR
        scores += np.where((damaged_arr[:, k] == 4) & valid, np.log(p_anc / SEQ_ERR), 0.0)
    for k in range(MAX_LEN):
        dist3 = lengths_arr - 1 - k
        valid = (dist3 >= 0) & (k < lengths_arr)
        p_ga  = BRIGGS_S * (BRIGGS_L ** np.maximum(dist3, 0)) + BRIGGS_D
        p_anc = p_ga + (1 - p_ga) * SEQ_ERR
        scores += np.where((damaged_arr[:, k] == 1) & valid, np.log(p_anc / SEQ_ERR), 0.0)
    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not PROBS_FILE.exists():
        print(f'ERROR: {PROBS_FILE} not found — run 10_classifier.py first.')
        return

    d      = np.load(PROBS_FILE)
    labels = d['labels']
    N      = len(labels)
    prev   = labels.mean()

    # ── Collect all models: ML variants + Briggs baseline ─────────────────────
    models = {}
    for key, name in PROB_KEYS.items():
        if key in d:
            models[name] = d[key]

    # Recompute Briggs LLR if test data is available
    test_npz = DATA_DIR / 'test.npz'
    if test_npz.exists():
        td = np.load(test_npz)
        if len(td['damaged']) == N:
            llr = compute_briggs_scores(
                td['damaged'].astype(np.int32), td['lengths'].astype(np.int32))
            models['Briggs LLR'] = 1.0 / (1.0 + np.exp(-llr * 0.1))
    if 'Briggs LLR' not in models:
        print('  Note: test.npz not found or size mismatch — Briggs LLR omitted.')

    order = [n for n in [*PROB_KEYS.values(), 'Briggs LLR'] if n in models]

    # ── Compute curves and threshold stats ────────────────────────────────────
    curves = {}
    stats  = {}
    for name, probs in models.items():
        fpr, tpr, thr = roc_curve(labels, probs)
        t_y  = youden_threshold(fpr, tpr, thr)
        t_f  = best_f1_threshold(probs, labels)
        t_m  = best_mcc_threshold(probs, labels)
        curves[name] = (fpr, tpr, thr)
        stats[name]  = {
            'roc_auc':   roc_auc_score(labels, probs),
            'pr_auc':    average_precision_score(labels, probs),
            't_youden': t_y, 't_f1': t_f, 't_mcc': t_m,
            'at_youden': stats_at(probs, labels, t_y),
            'at_f1':     stats_at(probs, labels, t_f),
            'at_mcc':    stats_at(probs, labels, t_m),
        }

    # ── Figure: 3 panels ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle('aDNA Read Classifier — ML Models vs Briggs LLR Baseline',
                 fontsize=13, fontweight='bold', y=1.01)

    # Panel 1 — ROC (dot at Youden threshold)
    ax = axes[0]
    for name in order:
        fpr, tpr, _ = curves[name]
        auc = stats[name]['roc_auc']
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, ls=LINE_STYLES[name],
                label=f'{name}  (AUC={auc:.3f})')
        s = stats[name]['at_youden']
        ax.scatter([s['fpr']], [s['sens']], color=COLORS[name], s=90,
                   zorder=5, edgecolors='black', linewidths=0.8)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.25, lw=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    ax.set_title('(A) ROC Curve  (● = Youden threshold)', fontsize=11)
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel 2 — Precision-Recall (dot at best-F1 threshold)
    ax = axes[1]
    ax.axhline(prev, color='k', ls='--', alpha=0.3, lw=1,
               label=f'Random  (prevalence = {prev:.2f})')
    for name in order:
        prec, rec, _ = precision_recall_curve(labels, models[name])
        ap = stats[name]['pr_auc']
        ax.plot(rec, prec, color=COLORS[name], lw=2, ls=LINE_STYLES[name],
                label=f'{name}  (AP={ap:.3f})')
        s = stats[name]['at_f1']
        ax.scatter([s['sens']], [s['ppv']], color=COLORS[name], s=90,
                   zorder=5, edgecolors='black', linewidths=0.8)
    ax.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax.set_ylabel('Precision (PPV)', fontsize=11)
    ax.set_title('(B) Precision-Recall Curve  (● = best-F1 threshold)', fontsize=11)
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel 3 — Bar chart at Youden threshold
    ax    = axes[2]
    mkeys = ['roc_auc', 'pr_auc', 'sens', 'spec', 'ppv', 'npv', 'f1', 'bal_acc', 'mcc']
    mlbls = ['ROC-AUC', 'PR-AUC', 'Sens', 'Spec', 'PPV', 'NPV', 'F1', 'BalAcc', 'MCC']
    x     = np.arange(len(mkeys))
    width = 0.8 / len(order)
    for i, name in enumerate(order):
        s    = stats[name]
        vals = [s['roc_auc'], s['pr_auc']] + [s['at_youden'][k] for k in mkeys[2:]]
        bars = ax.bar(x + (i - len(order)/2 + 0.5)*width, vals,
                      width, label=name, color=COLORS[name], alpha=0.85)
        for b in bars:
            v = b.get_height()
            if v > 0.04:
                ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(mlbls, fontsize=8.5, rotation=25, ha='right')
    ax.set_ylim(0, 1.15); ax.set_ylabel('Score', fontsize=11)
    ax.set_title('(C) All metrics at Youden threshold', fontsize=11)
    ax.legend(fontsize=7.5)
    ax.axhline(0.5, color='grey', ls=':', alpha=0.4, lw=1)

    plt.tight_layout()
    out_fig = OUT_DIR / 'figures' / 'classifier_comparison.png'
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure → {out_fig}')

    # ── Text summary ──────────────────────────────────────────────────────────
    W   = 8
    col = ['ROC-AUC', 'PR-AUC', 'Sens', 'Spec', 'PPV', 'NPV', 'F1', 'Acc', 'BalAcc', 'MCC', 'Thr']
    hdr = f'{"Model":<32}' + ''.join(f'  {c:>{W}}' for c in col)
    sep = '-' * len(hdr)

    def fmt_row(name, stat_key, thr_key):
        s  = stats[name]
        ay = s[stat_key]
        vals = [s['roc_auc'], s['pr_auc'],
                ay['sens'], ay['spec'], ay['ppv'], ay['npv'],
                ay['f1'], ay['acc'], ay['bal_acc'], ay['mcc'], s[thr_key]]
        return f'  {name:<30}' + ''.join(f'  {v:>{W}.4f}' for v in vals)

    lines = [
        'aDNA Read Classifier — Model Comparison',
        '=' * len(hdr),
        f'Test set: {N:,} reads  ({prev*100:.1f}% ancient / {(1-prev)*100:.1f}% modern)',
        f'Random PR-AUC baseline = {prev:.4f}  (class prevalence)',
        '',
        'Threshold selection: three operating points are reported.',
        '  Youden J  = maximises Sensitivity + Specificity − 1',
        '  Best-MCC  = maximises Matthews Correlation Coefficient (recommended)',
        '  Best-F1   = maximises F1 score',
        '',
        '── At Youden J threshold ──', hdr, sep,
    ] + [fmt_row(n, 'at_youden', 't_youden') for n in order]

    lines += ['', '── At best-MCC threshold (recommended) ──', hdr, sep]
    lines += [fmt_row(n, 'at_mcc', 't_mcc') for n in order]

    lines += ['', '── At best-F1 threshold ──', hdr, sep]
    lines += [fmt_row(n, 'at_f1', 't_f1') for n in order]

    txt = '\n'.join(lines)
    out_txt = OUT_DIR / 'results' / 'classifier_comparison.txt'
    out_txt.write_text(txt)
    print(f'Summary → {out_txt}')
    print()
    print(txt)


if __name__ == '__main__':
    main()
