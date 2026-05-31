"""12_per_source_auc.py — Per-source classifier AUC (bacterial vs human vs environmental)."""

from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

DATA_DIR = Path('data')
OUT_DIR  = Path('outputs')

SRC = {0: 'bact', 1: 'human', 2: 'env'}

PROB_KEYS = {
    'probs_seq':      'Seq-only',
    'probs_evo_base': 'Evo2 (per-base)',
    'probs_evo_full': 'Evo2 (per-base + read LL)',
}
COLORS = {
    'Seq-only':                  '#3498db',
    'Evo2 (per-base)':           '#2ecc71',
    'Evo2 (per-base + read LL)': '#1a7a49',
}


def auc_safe(y, p):
    if y.sum() == 0 or y.sum() == len(y) or len(y) < 50:
        return float('nan')
    return roc_auc_score(y, p)


def main():
    test_npz = DATA_DIR / 'test.npz'
    probs_npz = OUT_DIR / 'results' / 'classifier_probs.npz'
    test  = np.load(test_npz)
    probs = np.load(probs_npz)

    sources = test['sources']
    labels  = probs['labels']
    n_total = len(labels)
    print(f'Test set: {n_total:,} reads')
    print(f'  Source counts: bact={int((sources==0).sum()):,}  '
          f'human={int((sources==1).sum()):,}  env={int((sources==2).sum()):,}')
    print(f'  Ancient prevalence: {labels.mean()*100:.1f}%')
    print()

    # Per-source masks for the modern class
    is_ancient   = labels.astype(bool)
    is_modern    = ~is_ancient
    is_bact      = sources == 0
    is_human     = sources == 1
    is_env       = sources == 2

    contrasts = {
        'overall':        np.ones(n_total, dtype=bool),
        'bact-vs-bact':   is_ancient | (is_modern & is_bact),
        'bact-vs-human':  is_ancient | (is_modern & is_human),
        'bact-vs-env':    is_ancient | (is_modern & is_env),
    }

    rows = []
    for key, name in PROB_KEYS.items():
        if key not in probs:
            continue
        p = probs[key]
        row = {'model': name}
        for c_name, mask in contrasts.items():
            y_sub = labels[mask]
            p_sub = p[mask]
            row[c_name] = auc_safe(y_sub, p_sub)
            row[f'{c_name}_n_modern']  = int(mask.sum() - y_sub.sum())
        rows.append(row)

    # ── Text summary ─────────────────────────────────────────────────────────
    lines = [
        'Per-source classifier evaluation',
        '=' * 78,
        f'Test set: {n_total:,} reads  ({labels.mean()*100:.1f}% ancient)',
        f'Source counts: bact {int((sources==0).sum()):,} | '
        f'human {int((sources==1).sum()):,} | env {int((sources==2).sum()):,}',
        '',
        'Interpretation:',
        '  overall          — full test set (ancient vs all modern)',
        '  bact-vs-bact     — same composition as ancient (only damage can drive score)',
        '  bact-vs-human    — large composition gap (length, GC, kmers all differ)',
        '  bact-vs-env      — medium composition gap',
        '',
        f'  {"Model":<30}  {"overall":>9}  {"bact":>9}  {"human":>9}  {"env":>9}',
        '  ' + '-' * 75,
    ]
    for r in rows:
        lines.append(
            f'  {r["model"]:<30}  '
            f'{r["overall"]:>9.4f}  {r["bact-vs-bact"]:>9.4f}  '
            f'{r["bact-vs-human"]:>9.4f}  {r["bact-vs-env"]:>9.4f}'
        )
    lines += [
        '',
        'Reading guide:',
        '  - If `bact-vs-bact` ≈ `overall`, the model is using genuine damage signal,',
        '    not composition (good — damage detection is what we want).',
        '  - If `bact-vs-human` ≫ `bact-vs-bact`, the model is exploiting compositional',
        '    differences (length, k-mer freq, GC) rather than damage. The gap quantifies',
        '    how much classification credit comes from composition vs damage.',
        '  - The Evo2 variants should ideally show a smaller composition-vs-damage gap',
        '    than seq-only, since the Evo2 reference is supposed to add damage-specific',
        '    rather than composition-specific signal.',
    ]
    txt = '\n'.join(lines)
    out_txt = OUT_DIR / 'results' / 'per_source_auc.txt'
    out_txt.write_text(txt)
    print(txt)
    print(f'\nSummary → {out_txt}')

    # ── Figure: grouped bar chart ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    contrast_labels = ['overall', 'bact-vs-bact', 'bact-vs-human', 'bact-vs-env']
    contrast_keys   = ['overall', 'bact-vs-bact', 'bact-vs-human', 'bact-vs-env']
    x = np.arange(len(contrast_labels))
    width = 0.8 / max(len(rows), 1)

    for i, r in enumerate(rows):
        vals = [r[k] for k in contrast_keys]
        bars = ax.bar(x + (i - len(rows)/2 + 0.5)*width, vals, width,
                      label=r['model'], color=COLORS.get(r['model'], '#888'),
                      alpha=0.9)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(contrast_labels)
    ax.axhline(0.5, color='grey', ls=':', alpha=0.5, label='Random (0.5)')
    ax.set_ylabel('ROC-AUC')
    ax.set_ylim(0.45, 1.0)
    ax.set_title('Classifier per-source ROC-AUC  '
                 '(does the model use damage or composition?)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_png = OUT_DIR / 'figures' / 'per_source_auc.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure  → {out_png}')


if __name__ == '__main__':
    main()
