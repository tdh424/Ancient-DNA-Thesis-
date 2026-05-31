"""8.3_denoiser_errors.py — Error analysis of the denoiser's predicted changes."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = Path('outputs')

# Encoding: PAD=0 A=1 C=2 G=3 T=4 N=5
PAD, A, C, G, T = 0, 1, 2, 3, 4


def load(variant):
    p = OUT_DIR / 'results' / f'test_denoised_{variant}.npz'
    if not p.exists():
        raise SystemExit(f'ERROR: {p} not found — run 6_denoise.py first.')
    return np.load(p)


def recompute_denoised(damaged, prob_c, prob_g, threshold):
    """Re-apply the threshold to prob_c / prob_g without re-running the model.

    Mirrors the rule in 7_denoise.py:174-176:
      flip T->C where damaged==T and prob_c > threshold
      flip A->G where damaged==A and prob_g > threshold
    """
    A_, C_, G_, T_ = 1, 2, 3, 4
    denoised = damaged.copy()
    denoised[(damaged == T_) & (prob_c > threshold)] = C_
    denoised[(damaged == A_) & (prob_g > threshold)] = G_
    return denoised


def summarise(damaged, clean, denoised, lengths):
    """Compute the high-level stats Line asked about."""
    L = damaged.shape[1]
    pos = np.arange(L)[None, :]
    valid = pos < lengths[:, None]   # mask out padding

    # Where the model changed something
    changed       = (denoised != damaged) & valid
    # The two damage events the model is supposed to undo
    is_ct_dmg     = (damaged == T) & (clean == C) & valid
    is_ga_dmg     = (damaged == A) & (clean == G) & valid
    # A "modern" position is one where damaged == clean (no damage to undo)
    no_damage     = (damaged == clean) & valid

    n_pos         = valid.sum()
    n_changed     = changed.sum()
    n_ct_dmg      = is_ct_dmg.sum()
    n_ga_dmg      = is_ga_dmg.sum()
    n_no_damage   = no_damage.sum()

    # Of predicted changes: split into correct vs spurious
    correct_change = changed & (denoised == clean)
    wrong_change   = changed & (denoised != clean)

    # Damage events — recovered vs missed
    ct_recovered = (is_ct_dmg & (denoised == C)).sum()
    ct_missed    = (is_ct_dmg & (denoised == T)).sum()
    ga_recovered = (is_ga_dmg & (denoised == G)).sum()
    ga_missed    = (is_ga_dmg & (denoised == A)).sum()

    # False flips — model changed a position that did not need changing
    false_flip_T_to_C = ((damaged == T) & (clean == T) & (denoised == C) & valid).sum()
    false_flip_A_to_G = ((damaged == A) & (clean == A) & (denoised == G) & valid).sum()

    # Reads with at least one prediction
    reads_with_change = (changed.any(axis=1)).sum()
    n_reads           = damaged.shape[0]

    return dict(
        n_reads=n_reads, reads_with_change=int(reads_with_change),
        n_pos=int(n_pos), n_changed=int(n_changed),
        n_correct_change=int(correct_change.sum()),
        n_wrong_change=int(wrong_change.sum()),
        n_ct_dmg=int(n_ct_dmg), ct_recovered=int(ct_recovered), ct_missed=int(ct_missed),
        n_ga_dmg=int(n_ga_dmg), ga_recovered=int(ga_recovered), ga_missed=int(ga_missed),
        n_no_damage=int(n_no_damage),
        false_flip_T_to_C=int(false_flip_T_to_C),
        false_flip_A_to_G=int(false_flip_A_to_G),
    )


def position_curves(damaged, clean, denoised, lengths):
    L = damaged.shape[1]
    pos = np.arange(L)
    valid = pos[None, :] < lengths[:, None]

    # Index from 5'
    is_ct = (damaged == T) & (clean == C) & valid
    rec5  = is_ct & (denoised == C)
    ct_per_pos      = is_ct.sum(axis=0)
    ct_rec_per_pos  = rec5.sum(axis=0)

    # Index from 3' — reverse per-read
    rev_idx = (lengths[:, None] - 1 - pos[None, :]).clip(0, L - 1)
    is_ga    = (damaged == A) & (clean == G) & valid
    rec_ga   = is_ga & (denoised == G)
    ga_per_pos     = np.zeros(L, dtype=np.int64)
    ga_rec_per_pos = np.zeros(L, dtype=np.int64)
    for k in range(L):
        m = (rev_idx == k) & valid
        ga_per_pos[k]     = (is_ga & m).sum()
        ga_rec_per_pos[k] = (rec_ga & m).sum()

    # False flips per 5' position (T -> C on a genuine T)
    ff_T = ((damaged == T) & (clean == T) & (denoised == C) & valid).sum(axis=0)

    # False flips per 3' position (A -> G on a genuine A), indexed from 3' end
    is_ff_A = (damaged == A) & (clean == A) & (denoised == G) & valid
    ff_A = np.zeros(L, dtype=np.int64)
    for k in range(L):
        m = (rev_idx == k) & valid
        ff_A[k] = (is_ff_A & m).sum()
    return ct_per_pos, ct_rec_per_pos, ga_per_pos, ga_rec_per_pos, ff_T, ff_A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='evo2',
                    help="Denoiser variant: 'evo2' (default) or 'baseline'.")
    ap.add_argument('--threshold', type=float, default=None,
                    help='Optional probability threshold. If given, denoised is '
                         'recomputed from prob_c/prob_g at this threshold. '
                         'If omitted, uses the denoised array stored in the NPZ '
                         '(produced by 7_denoise.py at THRESHOLD=0.5).')
    ap.add_argument('--threshold2', type=float, default=None,
                    help='Optional second threshold to overlay on the same '
                         'figure (e.g. --threshold 0.34 --threshold2 0.30 to '
                         'compare F1-optimal and MCC-optimal operating points). '
                         'The text report uses --threshold only.')
    args = ap.parse_args()

    d = load(args.variant)
    damaged  = d['damaged'].astype(np.int32)
    clean    = d['clean'].astype(np.int32)
    lengths  = d['lengths'].astype(np.int32)
    if args.threshold is not None:
        if 'prob_c' not in d or 'prob_g' not in d:
            raise SystemExit('ERROR: --threshold requires prob_c and prob_g in NPZ.')
        denoised = recompute_denoised(
            damaged, d['prob_c'], d['prob_g'], args.threshold
        ).astype(np.int32)
        thr_label = f'threshold={args.threshold:.2f}'
    else:
        denoised = d['denoised'].astype(np.int32)
        thr_label = 'threshold=0.50 (default)'

    s = summarise(damaged, clean, denoised, lengths)
    ct5, ct5r, ga3, ga3r, ff5, ff3 = position_curves(damaged, clean, denoised, lengths)

    # Optional second threshold for overlay
    s2 = None
    if args.threshold2 is not None:
        denoised2 = recompute_denoised(
            damaged, d['prob_c'], d['prob_g'], args.threshold2
        ).astype(np.int32)
        s2 = summarise(damaged, clean, denoised2, lengths)
        ct5_2, ct5r_2, ga3_2, ga3r_2, ff5_2, ff3_2 = position_curves(
            damaged, clean, denoised2, lengths)
        thr_label = (f'F1-opt $\\tau={args.threshold:.2f}$ vs. '
                     f'MCC-opt $\\tau={args.threshold2:.2f}$')

    # ── Text report ──────────────────────────────────────────────────────────
    pct = lambda a, b: 100 * a / max(b, 1)
    rep = [
        f'Denoiser error analysis  —  variant: {args.variant}',
        '=' * 72,
        f'Reads: {s["n_reads"]:>10,}    Reads where model predicted a change: '
        f'{s["reads_with_change"]:>10,} ({pct(s["reads_with_change"], s["n_reads"]):.1f}%)',
        f'Valid positions analysed: {s["n_pos"]:,}',
        '',
        '── Behaviour: when does the model intervene? ────────────────────────',
        f'  Positions changed by model:        {s["n_changed"]:>10,} '
        f'({pct(s["n_changed"], s["n_pos"]):.2f}% of all positions)',
        f'    of which correct (→ clean):      {s["n_correct_change"]:>10,} '
        f'({pct(s["n_correct_change"], max(s["n_changed"],1)):.1f}% of changes)',
        f'    of which wrong (≠ clean):        {s["n_wrong_change"]:>10,} '
        f'({pct(s["n_wrong_change"], max(s["n_changed"],1)):.1f}% of changes)',
        '',
        '── C→T damage (model should flip T → C at 5\' end) ───────────────────',
        f'  True C→T damage events:            {s["n_ct_dmg"]:>10,}',
        f'    recovered (T→C correctly):       {s["ct_recovered"]:>10,} '
        f'({pct(s["ct_recovered"], s["n_ct_dmg"]):.2f}%  =  recall on C→T)',
        f'    missed   (left as T):            {s["ct_missed"]:>10,}',
        '',
        '── G→A damage (model should flip A → G at 3\' end) ───────────────────',
        f'  True G→A damage events:            {s["n_ga_dmg"]:>10,}',
        f'    recovered (A→G correctly):       {s["ga_recovered"]:>10,} '
        f'({pct(s["ga_recovered"], s["n_ga_dmg"]):.2f}%  =  recall on G→A)',
        f'    missed   (left as A):            {s["ga_missed"]:>10,}',
        '',
        '── False flips (model damaged a healthy base) ───────────────────────',
        f'  True T flipped to C (5\'):          {s["false_flip_T_to_C"]:>10,}',
        f'  True A flipped to G (3\'):          {s["false_flip_A_to_G"]:>10,}',
        f'  Total false flips:                 '
        f'{s["false_flip_T_to_C"]+s["false_flip_A_to_G"]:>10,}',
        f'  False-flip rate / true-flip rate:  '
        f'{(s["false_flip_T_to_C"]+s["false_flip_A_to_G"]) / max(s["ct_recovered"]+s["ga_recovered"],1):.2f}',
        '',
        '── Diagnosis ─────────────────────────────────────────────────────────',
    ]

    # Heuristic interpretation
    ct_recall = pct(s['ct_recovered'], s['n_ct_dmg'])
    ga_recall = pct(s['ga_recovered'], s['n_ga_dmg'])
    if s['n_changed'] / max(s['n_pos'], 1) > 0.10:
        rep.append('  ⚠  Model is over-active: changes >10% of all bases.')
    if ct_recall > 5 * ga_recall + 1e-6:
        rep.append(f'  ⚠  G→A is severely under-learned: '
                   f'C→T recall {ct_recall:.1f}% vs G→A recall {ga_recall:.1f}%.')
    if (s['false_flip_T_to_C'] + s['false_flip_A_to_G']) > (s['ct_recovered'] + s['ga_recovered']):
        rep.append('  ⚠  More false flips than correct recoveries — '
                   'precision is below 50%.')

    txt = '\n'.join(rep)
    out_txt = OUT_DIR / 'results' / f'denoiser_errors_{args.variant}.txt'
    out_txt.write_text(txt)
    print(txt)
    print(f'\nSummary → {out_txt}')

    # ── Figure: 4 panels ─────────────────────────────────────────────────────
    figsize = (14, 10) if s2 is not None else (12, 9)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Denoiser error analysis  —  {args.variant}  ({thr_label})',
                 fontsize=13, fontweight='bold')

    def fmt_count(v):
        """Human-readable count with appropriate unit (no '0.00M' surprises)."""
        if v >= 1_000_000:
            return f'{v/1e6:.2f}M'
        if v >= 10_000:
            return f'{v/1e3:.1f}k'
        if v >= 1_000:
            return f'{v/1e3:.2f}k'
        return f'{v:,}'

    # Labels used when overlaying two thresholds
    overlay = s2 is not None
    lbl1 = f'$\\tau={args.threshold:.2f}$' if args.threshold is not None else 'default'
    lbl2 = f'$\\tau={args.threshold2:.2f}$' if overlay else None

    # Panel A — share of behaviour bins
    ax = axes[0, 0]
    cats = ['Correct\nchange', 'Wrong\nchange', 'Missed C→T', 'Missed G→A',
            'No-op (was\nalready clean)']
    n_unchanged_ok = (s['n_no_damage'] - s['false_flip_T_to_C'] - s['false_flip_A_to_G'])
    vals = [s['n_correct_change'], s['n_wrong_change'],
            s['ct_missed'], s['ga_missed'], n_unchanged_ok]
    cols = ['#27ae60', '#e74c3c', '#e67e22', '#d35400', '#95a5a6']
    xpos = np.arange(len(cats))
    if overlay:
        n_unchanged_ok2 = (s2['n_no_damage'] - s2['false_flip_T_to_C']
                           - s2['false_flip_A_to_G'])
        vals2 = [s2['n_correct_change'], s2['n_wrong_change'],
                 s2['ct_missed'], s2['ga_missed'], n_unchanged_ok2]
        w = 0.4
        bars1 = ax.bar(xpos - w/2, vals, w, color=cols,
                       edgecolor='black', linewidth=0.5, label=lbl1)
        bars2 = ax.bar(xpos + w/2, vals2, w, color=cols,
                       edgecolor='black', linewidth=0.5,
                       hatch='///', alpha=0.7, label=lbl2)
        for b, v in zip(bars1, vals):
            ax.text(b.get_x() + b.get_width()/2, v, fmt_count(v),
                    ha='center', va='bottom', fontsize=7)
        for b, v in zip(bars2, vals2):
            ax.text(b.get_x() + b.get_width()/2, v, fmt_count(v),
                    ha='center', va='bottom', fontsize=7)
        # Legend for the hatch pattern only (the colours are category-specific)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='white', edgecolor='black', label=lbl1),
            Patch(facecolor='white', edgecolor='black', hatch='///',
                  alpha=0.7, label=lbl2),
        ], loc='lower right', fontsize=8)
    else:
        bars = ax.bar(xpos, vals, color=cols)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v, fmt_count(v),
                    ha='center', va='bottom', fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(cats)
    # Log-scale only if there are strictly positive values in *all* series
    # plotted; otherwise matplotlib produces an undefined axis range that
    # `bbox_inches='tight'` later bakes into a 100k-pixel-tall PNG.
    pos_vals = [v for v in vals if v > 0]
    if overlay:
        pos_vals += [v for v in vals2 if v > 0]
    if pos_vals:
        ax.set_yscale('log')
        ax.set_ylim(bottom=max(1, min(pos_vals) * 0.5))
    ax.set_ylabel('# positions (log)' if pos_vals else '# positions')
    ax.set_title('(A) Where do model decisions land?')

    # Panel B — recall per damage class
    ax = axes[0, 1]
    if overlay:
        ct_recall_2 = pct(s2['ct_recovered'], s2['n_ct_dmg'])
        ga_recall_2 = pct(s2['ga_recovered'], s2['n_ga_dmg'])
        xb = np.arange(2)
        w = 0.35
        b1 = ax.bar(xb - w/2, [ct_recall, ga_recall], w,
                    color=['#3498db', '#9b59b6'],
                    edgecolor='black', linewidth=0.5)
        b2 = ax.bar(xb + w/2, [ct_recall_2, ga_recall_2], w,
                    color=['#3498db', '#9b59b6'],
                    edgecolor='black', linewidth=0.5,
                    hatch='///', alpha=0.7)
        ymax = max(100.0, max(ct_recall, ga_recall,
                              ct_recall_2, ga_recall_2) * 1.15)
        ax.set_ylim(0, ymax)
        ax.set_xticks(xb); ax.set_xticklabels(['C→T recall', 'G→A recall'])
        for xi, v in zip(xb - w/2, [ct_recall, ga_recall]):
            ax.text(xi, v + ymax * 0.01, f'{v:.1f}%',
                    ha='center', fontsize=8)
        for xi, v in zip(xb + w/2, [ct_recall_2, ga_recall_2]):
            ax.text(xi, v + ymax * 0.01, f'{v:.1f}%',
                    ha='center', fontsize=8)
        # Legend uses neutral grey patches so blue/purple in the bars only
        # encode damage class (not threshold).
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#bdc3c7', edgecolor='black', label=lbl1),
            Patch(facecolor='#bdc3c7', edgecolor='black',
                  hatch='///', alpha=0.7, label=lbl2),
        ], loc='upper right', fontsize=8)
    else:
        ax.bar(['C→T recall', 'G→A recall'], [ct_recall, ga_recall],
               color=['#3498db', '#9b59b6'])
        ymax = max(100.0, max(ct_recall, ga_recall) * 1.15)
        ax.set_ylim(0, ymax)
        for i, v in enumerate([ct_recall, ga_recall]):
            ax.text(i, v + ymax * 0.01, f'{v:.2f}%',
                    ha='center', fontsize=10)
    ax.set_ylabel('Recall (%)')
    ax.set_title('(B) Damage-class recovery')

    # Panel C — C→T recall per 5' position
    ax = axes[1, 0]
    L = len(ct5)
    show = min(L, 30)
    rec  = ct5r[:show] / np.maximum(ct5[:show], 1)
    ff   = ff5[:show]
    xp = np.arange(show)
    if overlay:
        rec_2 = ct5r_2[:show] / np.maximum(ct5_2[:show], 1)
        ff_2  = ff5_2[:show]
        w = 0.4
        ax.bar(xp - w/2, rec * 100, w, color='#3498db',
               edgecolor='black', linewidth=0.3, label=f'C→T recall {lbl1}')
        ax.bar(xp + w/2, rec_2 * 100, w, color='#3498db',
               alpha=0.5, hatch='///', edgecolor='black', linewidth=0.3,
               label=f'C→T recall {lbl2}')
        ax2 = ax.twinx()
        ax2.plot(xp, ff, color='#e74c3c', lw=1.5, label=f'False flips {lbl1}')
        ax2.plot(xp, ff_2, color='#e74c3c', lw=1.5, ls='--',
                 label=f'False flips {lbl2}')
    else:
        ax.bar(xp, rec * 100, color='#3498db', label='C→T recall (%)')
        ax2 = ax.twinx()
        ax2.plot(xp, ff, color='#e74c3c', label='False T→C flips', lw=1.5)
    ax.set_xlabel("Position from 5' end")
    ax.set_ylabel('C→T recall (%)')
    ax2.set_ylabel('False flips')
    ax.set_title("(C) C→T recovery vs false flips (5')")
    ax.set_ylim(0, 100)
    if overlay:
        # combined legend
        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, loc='upper right', fontsize=7)

    # Panel D — G→A recall per 3' position (symmetric to Panel C)
    ax = axes[1, 1]
    show3 = min(len(ga3), 30)
    rec3 = ga3r[:show3] / np.maximum(ga3[:show3], 1)
    ff3s = ff3[:show3]
    xp3 = np.arange(show3)
    if overlay:
        rec3_2 = ga3r_2[:show3] / np.maximum(ga3_2[:show3], 1)
        ff3_2s = ff3_2[:show3]
        w = 0.4
        ax.bar(xp3 - w/2, rec3 * 100, w, color='#9b59b6',
               edgecolor='black', linewidth=0.3, label=f'G→A recall {lbl1}')
        ax.bar(xp3 + w/2, rec3_2 * 100, w, color='#9b59b6',
               alpha=0.5, hatch='///', edgecolor='black', linewidth=0.3,
               label=f'G→A recall {lbl2}')
        ax2 = ax.twinx()
        ax2.plot(xp3, ff3s, color='#e74c3c', lw=1.5, label=f'False flips {lbl1}')
        ax2.plot(xp3, ff3_2s, color='#e74c3c', lw=1.5, ls='--',
                 label=f'False flips {lbl2}')
    else:
        ax.bar(xp3, rec3 * 100, color='#9b59b6', label='G→A recall (%)')
        ax2 = ax.twinx()
        ax2.plot(xp3, ff3s, color='#e74c3c', label='False A→G flips', lw=1.5)
    ax.set_xlabel("Position from 3' end")
    ax.set_ylabel('G→A recall (%)')
    ax2.set_ylabel('False flips')
    ax.set_title("(D) G→A recovery vs false flips (3')")
    ax.set_ylim(0, 100)
    if overlay:
        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, loc='upper right', fontsize=7)

    plt.tight_layout()
    out_png = OUT_DIR / 'figures' / f'denoiser_errors_{args.variant}.png'
    # Drop `bbox_inches='tight'` — it included off-axis ticks/annotations in
    # the bounding box for all-zero data (seq_only, udg) and rendered the
    # PNG at 169k px tall.
    fig.savefig(out_png, dpi=150)
    plt.close()
    print(f'Figure  → {out_png}')


if __name__ == '__main__':
    main()
