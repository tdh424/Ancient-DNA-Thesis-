"""
7_evaluate.py — Denoiser evaluation: all trained variants vs baselines.

Loads all available test_denoised_{variant}.npz files and compares them on:
  1. Base reconstruction accuracy at true-C positions (C→T correction quality)
  2. Spurious stop codon rate (CDS proxy; C→T creates TAA/TAG/TGA stop codons)
  3. Precision-Recall curves for damage correction (P(C) at T positions)
  4. PR-AUC by read length

Reference lines on every plot:
  Damaged (no correction)  — lower bound (red dashed)
  Clean (ground truth)     — upper bound (green dashed)

Outputs → outputs/figures/evaluation.png
Outputs → outputs/results/evaluation_summary.txt
"""

from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, precision_recall_curve

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR = Path('outputs')
RES_DIR = OUT_DIR / 'results'

# Display names and colours for each denoiser variant
VARIANT_META = {
    'seq_only': {'label': 'Seq-only',    'color': '#3498db', 'ls': '-'},
    'evo2':     {'label': 'Evo2',        'color': '#9b59b6', 'ls': '-'},
    'bwa':      {'label': 'BWA',         'color': '#f39c12', 'ls': '-'},
}

DAMAGED_COLOR = '#e74c3c'   # red
CLEAN_COLOR   = '#2ecc71'   # green

STOP_CODONS = {(4, 1, 1), (4, 1, 3), (4, 3, 1)}   # TAA TAG TGA (vocab: A=1 C=2 G=3 T=4)
# ─────────────────────────────────────────────────────────────────────────────


def log(msg=''):
    print(msg, flush=True)


# ── Metrics ───────────────────────────────────────────────────────────────────

def reconstruction_accuracy(reads, clean):
    """Fraction of positions matching clean — overall and at true-C positions."""
    overall = float((reads == clean).mean())
    ct_mask = (clean == 2)
    at_c    = float((reads[ct_mask] == clean[ct_mask]).mean()) if ct_mask.sum() > 0 else float('nan')
    return overall, at_c


def count_stop_codons(reads):
    """Mean stop codons per read across all three reading frames."""
    N, L = reads.shape
    total = 0
    for frame in range(3):
        for k in range(frame, L - 2, 3):
            c = reads[:, k], reads[:, k+1], reads[:, k+2]
            for stop in STOP_CODONS:
                total += int(((c[0] == stop[0]) & (c[1] == stop[1]) & (c[2] == stop[2])).sum())
    return float(total) / N


def pr_auc(damaged, clean, prob_c):
    """PR-AUC for C→T correction: P(C) at T positions vs true label (clean == C)."""
    t_mask   = (damaged == 4).flatten()
    dmg_mask = ((damaged == 4) & (clean == 2)).flatten().astype(np.int32)
    if t_mask.sum() == 0 or dmg_mask[t_mask].sum() == 0:
        return float('nan')
    return float(average_precision_score(dmg_mask[t_mask], prob_c.flatten()[t_mask]))


def prauc_by_length(damaged, clean, prob_c, lengths):
    """PR-AUC broken down by read length bin."""
    bins       = list(range(30, 101, 10))
    bin_labels, bin_praucs, bin_ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (lengths >= lo) & (lengths < hi)
        if mask.sum() < 10:
            continue
        bd, bc, bp = damaged[mask], clean[mask], prob_c[mask]
        t_mask   = (bd == 4).flatten()
        dmg_mask = ((bd == 4) & (bc == 2)).flatten().astype(np.int32)
        if t_mask.sum() == 0 or dmg_mask[t_mask].sum() == 0:
            continue
        pa = average_precision_score(dmg_mask[t_mask], bp.flatten()[t_mask])
        bin_labels.append(f'{lo}–{hi-1} bp')
        bin_praucs.append(float(pa))
        bin_ns.append(int(mask.sum()))
    return bin_labels, bin_praucs, bin_ns


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    (OUT_DIR / 'figures').mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Discover available variant results ────────────────────────────────────
    available = {}
    for variant, meta in VARIANT_META.items():
        path = RES_DIR / f'test_denoised_{variant}.npz'
        if path.exists():
            available[variant] = path
    if not available:
        log('ERROR: No test_denoised_{variant}.npz found. Run 6_denoise.py first.')
        return
    log(f'Found denoised results for variants: {list(available.keys())}')

    # ── Load a reference set (damaged + clean) from the first available file ─
    d_ref    = np.load(next(iter(available.values())))
    damaged  = d_ref['damaged'].astype(np.int64)
    clean    = d_ref['clean'].astype(np.int64)
    lengths  = d_ref['lengths'].astype(np.int64)
    N        = len(damaged)

    # Mask padded positions
    L       = damaged.shape[1]
    pos_idx = np.arange(L)[None, :]
    valid   = pos_idx < lengths[:, None]
    damaged = np.where(valid, damaged, 0)
    clean   = np.where(valid, clean,   0)

    log(f'  {N:,} test reads  (lengths {lengths.min()}–{lengths.max()} bp)')

    # ── Compute metrics per variant ───────────────────────────────────────────
    results = {}
    for variant, path in available.items():
        d        = np.load(path)
        denoised = np.where(valid, d['denoised'].astype(np.int64), 0)
        prob_c   = np.where(valid, d['prob_c'].astype(np.float32), 0.0)

        _, at_c  = reconstruction_accuracy(denoised, clean)
        stops    = count_stop_codons(denoised)
        pa       = pr_auc(damaged, clean, prob_c)
        lbls, praucs, ns = prauc_by_length(damaged, clean, prob_c, lengths)

        results[variant] = {
            'denoised': denoised, 'prob_c': prob_c,
            'at_c': at_c, 'stops': stops,
            'pr_auc': pa,
            'bin_labels': lbls, 'bin_praucs': praucs, 'bin_ns': ns,
        }

    # Damaged + clean baseline metrics
    _, dmg_at_c = reconstruction_accuracy(damaged, clean)
    _, cln_at_c = reconstruction_accuracy(clean, clean)
    dmg_stops   = count_stop_codons(damaged)
    cln_stops   = count_stop_codons(clean)

    # ── Build text report ─────────────────────────────────────────────────────
    lines = []
    def record(msg):
        log(msg); lines.append(msg)

    record('\n' + '=' * 60)
    record('  METRIC 1 — Base Reconstruction Accuracy at True-C Positions')
    record('=' * 60)
    record(f'  (% of C→T-damaged positions correctly restored to C)\n')
    record(f'  {"Variant":<30}  {"At true-C positions":>20}')
    record(f'  {"-"*30}  {"-"*20}')
    record(f'  {"Damaged (no correction)":<30}  {dmg_at_c*100:>19.2f}%')
    for v, r in results.items():
        record(f'  {VARIANT_META[v]["label"]:<30}  {r["at_c"]*100:>19.2f}%')
    record(f'  {"Clean (upper bound)":<30}  {cln_at_c*100:>19.2f}%')

    record('\n' + '=' * 60)
    record('  METRIC 2 — Spurious Stop Codon Rate')
    record('=' * 60)
    record(f'  (C→T creates TAA/TAG/TGA stops; lower = better)\n')
    record(f'  {"Variant":<30}  {"Stop codons/read":>18}')
    record(f'  {"-"*30}  {"-"*18}')
    record(f'  {"Damaged (no correction)":<30}  {dmg_stops:>17.4f}')
    for v, r in results.items():
        record(f'  {VARIANT_META[v]["label"]:<30}  {r["stops"]:>17.4f}')
    record(f'  {"Clean (upper bound)":<30}  {cln_stops:>17.4f}')

    if dmg_stops > cln_stops:
        for v, r in results.items():
            reduction = (dmg_stops - r['stops']) / (dmg_stops - cln_stops) * 100
            record(f'  {VARIANT_META[v]["label"]}: {reduction:.1f}% of damaged→clean reduction')

    record('\n' + '=' * 60)
    record('  METRIC 3 — Damage Correction PR-AUC')
    record('=' * 60)
    record(f'  (PR-AUC for P(C) at T-positions where clean=C)\n')
    for v, r in results.items():
        record(f'  {VARIANT_META[v]["label"]:<30}  PR-AUC = {r["pr_auc"]:.4f}')

    (RES_DIR / 'evaluation_summary.txt').write_text('\n'.join(lines))

    # ── Figure ────────────────────────────────────────────────────────────────
    n_panels = 4
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.5))
    fig.suptitle('Denoiser Evaluation — All Variants vs Baselines',
                 fontsize=13, fontweight='bold')

    # Panel 1: Reconstruction accuracy at C positions
    ax   = axes[0]
    bar_names  = ['Damaged\n(no fix)'] + [VARIANT_META[v]['label'] for v in results] + ['Clean\n(ideal)']
    bar_vals   = [dmg_at_c*100] + [r['at_c']*100 for r in results.values()] + [cln_at_c*100]
    bar_colors = [DAMAGED_COLOR] + [VARIANT_META[v]['color'] for v in results] + [CLEAN_COLOR]
    bars = ax.bar(bar_names, bar_vals, color=bar_colors, width=0.55)
    ax.set_ylim(0, 105)
    ax.set_ylabel('% correct at true-C positions')
    ax.set_title('Reconstruction accuracy\n(higher = better)')
    for bar, v in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=9)
    # Reference lines
    ax.axhline(dmg_at_c*100, color=DAMAGED_COLOR, ls='--', alpha=0.5, lw=1)
    ax.axhline(cln_at_c*100, color=CLEAN_COLOR,   ls='--', alpha=0.5, lw=1)

    # Panel 2: Stop codon rates
    ax   = axes[1]
    bars2 = ax.bar(bar_names, [dmg_stops] + [r['stops'] for r in results.values()] + [cln_stops],
                   color=bar_colors, width=0.55)
    ax.set_ylabel('Stop codons per read')
    ax.set_title('Spurious stop codons\n(lower = better for CDS)')
    for bar, v in zip(bars2, [dmg_stops] + [r['stops'] for r in results.values()] + [cln_stops]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.001, f'{v:.3f}',
                ha='center', va='bottom', fontsize=9)
    ax.axhline(dmg_stops, color=DAMAGED_COLOR, ls='--', alpha=0.5, lw=1,
               label='Damaged baseline')
    ax.axhline(cln_stops, color=CLEAN_COLOR,   ls='--', alpha=0.5, lw=1,
               label='Clean upper bound')
    ax.legend(fontsize=7)

    # Panel 3: PR curves for damage correction
    ax = axes[2]
    t_mask   = (damaged == 4).flatten()
    dmg_mask = ((damaged == 4) & (clean == 2)).flatten().astype(np.int32)
    random_b = dmg_mask[t_mask].mean() if t_mask.sum() > 0 else float('nan')
    ax.axhline(random_b, color='grey', ls='--', lw=1, alpha=0.5,
               label=f'Random ({random_b:.3f})')
    for v, r in results.items():
        prec, rec, _ = precision_recall_curve(
            dmg_mask[t_mask], r['prob_c'].flatten()[t_mask])
        ax.plot(rec, prec, color=VARIANT_META[v]['color'],
                ls=VARIANT_META[v]['ls'], lw=2,
                label=f'{VARIANT_META[v]["label"]} (AP={r["pr_auc"]:.4f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall (damage correction)')
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel 4: PR-AUC by read length
    ax = axes[3]
    # Use first variant for bin labels
    first_v  = next(iter(results))
    bin_lbls = results[first_v]['bin_labels']
    x_pos    = np.arange(len(bin_lbls))
    w        = 0.8 / max(len(results), 1)
    for i, (v, r) in enumerate(results.items()):
        praucs = r['bin_praucs']
        if len(praucs) == len(bin_lbls):
            ax.bar(x_pos + (i - len(results)/2 + 0.5)*w, praucs, w,
                   label=VARIANT_META[v]['label'], color=VARIANT_META[v]['color'], alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_lbls, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('PR-AUC'); ax.set_title('PR-AUC by read length')
    ax.legend(fontsize=8)
    if bin_lbls:
        all_praucs = [p for r in results.values() for p in r['bin_praucs']]
        ax.axhline(np.mean(all_praucs), color='red', ls='--', alpha=0.5, lw=1, label='Mean')

    plt.tight_layout()
    out_fig = OUT_DIR / 'figures' / 'evaluation.png'
    fig.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    log(f'\nFigure saved → {out_fig}')
    log(f'Summary saved → {RES_DIR}/evaluation_summary.txt')


if __name__ == '__main__':
    main()
