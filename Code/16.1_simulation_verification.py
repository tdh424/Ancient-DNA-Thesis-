"""16.1_simulation_verification.py — Verification figure of the final simulated test set (length + damage)."""
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from project_root import project_root

ROOT = project_root()
NPZ  = ROOT / 'data/test.npz'
OUT_FIG  = ROOT / 'outputs/figures/simulation_verification.png'
THESIS_FIG = ROOT / 'thesis/figures/simulation_verification.png'

BRIGGS_S = 0.6815
BRIGGS_L = 0.359
BRIGGS_D = 0.00937
SHOW_K   = 20  # positions to plot from each end

A, C, G, T = 1, 2, 3, 4


def per_position_rate(damaged, clean, lengths, from_b, to_b, end):
    """Per-position substitution frequency (from_b -> to_b) along the read."""
    N, L = damaged.shape
    rate  = np.zeros(L, dtype=np.float64)
    denom = np.zeros(L, dtype=np.float64)
    pos = np.arange(L)
    for i in range(N):
        ln = lengths[i]
        if ln == 0:
            continue
        if end == '5p':
            idx = pos[:ln]
        else:  # 3'
            idx = ln - 1 - pos[:ln]
        for k in range(ln):
            p = int(idx[k])
            if 0 <= p < L:
                if clean[i, k] == from_b:
                    denom[p] += 1
                    if damaged[i, k] == to_b:
                        rate[p] += 1
    return rate / np.maximum(denom, 1)


def main():
    d = np.load(NPZ)
    damaged = d['damaged'].astype(np.int32)
    clean   = d['clean'].astype(np.int32)
    lengths = d['lengths'].astype(np.int32)
    sources = d['sources']

    bact = sources == 0
    # "Ancient" = bacterial reads that retained realised damage (the damaged
    # branch kept by prevalence control), i.e. damaged differs from clean.
    ancient = bact & ((damaged != clean).sum(axis=1) > 0)
    print(f'Bacterial test reads: {bact.sum():,} (mean length {lengths[bact].mean():.1f} bp)')
    print(f'  of which damaged (ancient): {ancient.sum():,}')

    # After prevalence control: rate averaged over ALL bacterial reads.
    ct_5p_all = per_position_rate(damaged[bact], clean[bact], lengths[bact], C, T, '5p')
    ga_3p_all = per_position_rate(damaged[bact], clean[bact], lengths[bact], G, A, '3p')
    # Before prevalence control: rate over the damaged reads only (close to Briggs).
    ct_5p_anc = per_position_rate(damaged[ancient], clean[ancient], lengths[ancient], C, T, '5p')
    ga_3p_anc = per_position_rate(damaged[ancient], clean[ancient], lengths[ancient], G, A, '3p')

    pos = np.arange(SHOW_K)
    theoretical = BRIGGS_S * BRIGGS_L ** pos + BRIGGS_D

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ── Panel 1: fragment length distribution ───────────────────────────
    ax = axes[0]
    ax.hist(lengths[bact], bins=np.arange(30, 101, 2),
            color='#5b9bd5', edgecolor='white', linewidth=0.5)
    ax.axvline(100, color='red', linestyle='--', linewidth=1.2,
               label='100 bp cap')
    ax.set_xlabel('Encoded fragment length (bp)')
    ax.set_ylabel('Number of bacterial reads')
    ax.set_title('Fragment length (after 30–100 bp filter)')
    ax.set_xlim(20, 105)
    ax.legend(loc='upper right', fontsize=9)

    # ── Panel 2: 5' C→T ─────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(pos, theoretical, '--', color='orange',
            label='Briggs $f(k)=s\\lambda^k+d$', linewidth=1.5)
    ax.plot(pos, ct_5p_anc[:SHOW_K], 'o-', color='#2ca02c',
            label='before prevalence control (damaged reads)', markersize=4)
    ax.plot(pos, ct_5p_all[:SHOW_K], 'o-', color='#1f77b4',
            label='after prevalence control (full test set)', markersize=4)
    ax.set_xlabel("Position from 5' end")
    ax.set_ylabel("C $\\to$ T frequency")
    ax.set_title("5$'$ end: C $\\to$ T deamination")
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=9)

    # ── Panel 3: 3' G→A ─────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(pos, theoretical, '--', color='orange',
            label='Briggs $f(k)=s\\lambda^k+d$', linewidth=1.5)
    ax.plot(pos, ga_3p_anc[:SHOW_K], 'o-', color='#2ca02c',
            label='before prevalence control (damaged reads)', markersize=4)
    ax.plot(pos, ga_3p_all[:SHOW_K], 'o-', color='#d62728',
            label='after prevalence control (full test set)', markersize=4)
    ax.set_xlabel("Distance from 3' end")
    ax.set_ylabel("G $\\to$ A frequency")
    ax.set_title("3$'$ end: G $\\to$ A deamination")
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure → {OUT_FIG}')

    if THESIS_FIG.parent.exists():
        shutil.copy(OUT_FIG, THESIS_FIG)
        print(f'Copied → {THESIS_FIG}')

    print(f'\nKey numbers for the caption:')
    print(f'  Length distribution: bacterial mean = {lengths[bact].mean():.1f} bp, '
          f'cap = 100 bp ({(lengths[bact] == 100).mean()*100:.1f} % at cap)')
    print(f'  5\' C→T position 0: before = {ct_5p_anc[0]:.3f}, '
          f'after = {ct_5p_all[0]:.3f}, theoretical = {theoretical[0]:.3f}')
    print(f'  3\' G→A position 0: before = {ga_3p_anc[0]:.3f}, '
          f'after = {ga_3p_all[0]:.3f}, theoretical = {theoretical[0]:.3f}')


if __name__ == '__main__':
    main()
