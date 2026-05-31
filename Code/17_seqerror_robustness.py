"""17_seqerror_robustness.py — Sequencing-error robustness of the classifier and denoiser."""

from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

DATA_DIR = Path('data')
OUT_DIR  = Path('outputs')

A, C, G, T = 1, 2, 3, 4
VOCAB_SIZE = 6
EXTRA_ERROR_RATES = [0.000, 0.005, 0.010, 0.020, 0.030]

# Import the classifier model class
import importlib.util
spec = importlib.util.spec_from_file_location('cls9', 'Code/9_classifier.py')
cls9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cls9)


def inject_errors(damaged, lengths, rate, seed=0):
    """Add uniform random substitutions among {A,C,G,T} at the given per-base rate.
    Modifies and returns a copy. Sequencer noise is symmetric across sources."""
    if rate <= 0:
        return damaged.copy()
    rng = np.random.default_rng(seed)
    out = damaged.copy()
    N, L = out.shape
    pos = np.arange(L)[None, :]
    valid = pos < lengths[:, None]

    pick = (rng.random(out.shape) < rate) & valid & (out != 0) & (out != 5)
    # For positions selected for an error, pick a random alternative base
    alternatives = rng.integers(1, 4, size=out.shape)
    # If original base is X, replace with ((X - 1 + alt) mod 4) + 1 to ensure different
    new_base = ((out - 1 + alternatives) % 4) + 1
    out = np.where(pick, new_base, out)
    return out


def make_labels(damaged, clean):
    return ((damaged != clean) & (damaged != 0)).any(axis=1).astype(np.int32)


def evaluate_classifier_at_rate(rate, device, seeds=(42,)):
    """Inject extra errors, run ensemble of classifier checkpoints, return AUCs."""
    d       = np.load(DATA_DIR / 'test.npz')
    # Labels reflect the *original* Briggs damage state of the read, not the
    # post-noise state. Deriving labels from the noise-injected array shifts
    # the positive prevalence with the noise level (random substitutions get
    # counted as damage) and inflates PR-AUC — exactly the artefact that made
    # the right-hand panel rise instead of fall.
    labels  = make_labels(d['damaged'].astype(np.int64), d['clean'])
    damaged = inject_errors(d['damaged'].astype(np.int64), d['lengths'], rate)
    lengths = d['lengths'].astype(np.int64)
    has_evo2 = 'ref_evo2' in d
    ref_evo2 = d['ref_evo2'].astype(np.float32) if has_evo2 else np.zeros(
        (len(damaged), 100, VOCAB_SIZE), np.float32)

    ds     = cls9.ReadDataset(damaged, lengths, labels, ref_evo2)
    loader = torch.utils.data.DataLoader(ds, batch_size=cls9.BATCH_SIZE,
                                          shuffle=False, num_workers=2)

    results = {}
    for variant_key in ('seq', 'evo_base', 'evo_full'):
        cfg = cls9.VARIANT_CONFIG[variant_key]
        seed_probs = []
        for seed in seeds:
            ckpt = OUT_DIR / 'models' / f'classifier_{variant_key}_seed{seed}.pt'
            if not ckpt.exists():
                continue
            model = cls9.aDNAClassifier(use_evo2=cfg['use_evo2'],
                                        n_dmg_features=cfg['n_feat']).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            p, _ = cls9.evaluate(model, loader, device)
            seed_probs.append(p)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        if not seed_probs:
            continue
        avg = np.mean(np.stack(seed_probs), axis=0)
        results[variant_key] = {
            'roc_auc': roc_auc_score(labels, avg),
            'pr_auc':  average_precision_score(labels, avg),
        }
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    rows = []
    for rate in EXTRA_ERROR_RATES:
        print(f'\nExtra error rate: {rate*100:.1f}%')
        res = evaluate_classifier_at_rate(rate, device, seeds=(42, 43, 44))
        for variant_key, metrics in res.items():
            rows.append({'rate': rate, 'variant': variant_key, **metrics})
            print(f'  {cls9.MODEL_NAMES[variant_key]:<30}  '
                  f'ROC-AUC = {metrics["roc_auc"]:.4f}   PR-AUC = {metrics["pr_auc"]:.4f}')

    # Plot
    variants = sorted({r['variant'] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle('Classifier robustness to additional sequencing error',
                 fontsize=13, fontweight='bold')
    for v in variants:
        sub = sorted([r for r in rows if r['variant'] == v], key=lambda r: r['rate'])
        xs  = [r['rate'] * 100 for r in sub]
        axes[0].plot(xs, [r['roc_auc'] for r in sub], 'o-',
                     label=cls9.MODEL_NAMES[v])
        axes[1].plot(xs, [r['pr_auc']  for r in sub], 'o-',
                     label=cls9.MODEL_NAMES[v])
    for ax, ylab, letter in zip(axes, ['ROC-AUC', 'PR-AUC'], ['A', 'B']):
        ax.set_xlabel('Extra sequencing error (%/base)')
        ax.set_ylabel(ylab); ax.legend(fontsize=9); ax.grid(alpha=0.25)
        ax.set_title(f'({letter}) {ylab} vs. extra sequencing error')

    plt.tight_layout()
    out_png = OUT_DIR / 'figures' / 'seqerror_robustness.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nFigure  -> {out_png}')

    # Text summary
    lines = ['Sequencing-error robustness', '=' * 60, '',
             f'  {"Variant":<28} {"Extra err":>10}  {"ROC-AUC":>8}  {"PR-AUC":>8}',
             '  ' + '-' * 60]
    for r in rows:
        lines.append(f'  {cls9.MODEL_NAMES[r["variant"]]:<28} '
                     f'{r["rate"]*100:>9.1f}%  {r["roc_auc"]:>8.4f}  {r["pr_auc"]:>8.4f}')
    out_txt = OUT_DIR / 'results' / 'seqerror_robustness.txt'
    out_txt.write_text('\n'.join(lines))
    print(f'Summary -> {out_txt}')


if __name__ == '__main__':
    main()
