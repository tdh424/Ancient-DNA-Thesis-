"""16_damage_stats.py — Verify the simulated dataset's damage profile against the Briggs curve."""

from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path('data')
OUT_DIR  = Path('outputs')

# Briggs parameters — must match Code/3_simulate.sh
BRIGGS_S = 0.6815
BRIGGS_L = 0.3590
BRIGGS_D = 0.00937

A, C, G, T = 1, 2, 3, 4


def theoretical_briggs(positions, s=BRIGGS_S, lam=BRIGGS_L, d=BRIGGS_D):
    """Briggs deamination frequency as a function of distance from end."""
    return s * (lam ** positions) + d


def per_position_rate(damaged, clean, lengths, from_base, to_base, end='5p'):
    """Compute per-position substitution frequency (from->to) along the read."""
    N, L  = damaged.shape
    rate  = np.zeros(L, dtype=np.float64)
    denom = np.zeros(L, dtype=np.float64)
    pos   = np.arange(L)
    for i in range(N):
        ln = lengths[i]
        if ln == 0:
            continue
        if end == '5p':
            row_pos = pos[:ln]
        else:
            row_pos = (ln - 1 - pos[:ln])
        for k in range(ln):
            p = int(row_pos[k])
            if 0 <= p < L:
                if clean[i, k] == from_base:
                    denom[p] += 1
                    if damaged[i, k] == to_base:
                        rate[p] += 1
    return rate / np.maximum(denom, 1)


def main():
    test_npz = DATA_DIR / 'test.npz'
    d = np.load(test_npz)
    damaged = d['damaged'].astype(np.int32)
    clean   = d['clean'].astype(np.int32)
    lengths = d['lengths'].astype(np.int32)
    sources = d['sources'] if 'sources' in d else np.zeros(len(damaged), dtype=np.uint8)

    N = len(damaged)
    is_bact  = sources == 0
    has_dmg  = (damaged != clean).any(axis=1)

    print(f'Test set: {N:,} reads ({has_dmg.sum():,} with damage = '
          f'{100*has_dmg.mean():.1f}%)')

    # Position-wise C->T (from 5') and G->A (from 3') over BACTERIAL reads only
    # (the only source with simulated damage).
    bact_idx  = np.where(is_bact)[0]
    print(f'Computing per-position rates on {len(bact_idx):,} bacterial reads...')
    ct_5p = per_position_rate(damaged[bact_idx], clean[bact_idx],
                              lengths[bact_idx], C, T, end='5p')
    ga_3p = per_position_rate(damaged[bact_idx], clean[bact_idx],
                              lengths[bact_idx], G, A, end='3p')

    # Plot
    show = 20
    pos = np.arange(show)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Simulated damage profile — bacterial test reads',
                 fontsize=13, fontweight='bold')

    theoretical = theoretical_briggs(pos)
    axes[0].plot(pos, ct_5p[:show], 'o-', color='#3498db', label='measured')
    axes[0].plot(pos, theoretical,  '--', color='orange',  label='Briggs theoretical')
    axes[0].set_xlabel("Position from 5' end")
    axes[0].set_ylabel('C -> T frequency')
    axes[0].set_title("(A) 5' end")
    axes[0].legend()
    axes[0].set_ylim(bottom=0)

    axes[1].plot(pos, ga_3p[:show], 'o-', color='#e74c3c', label='measured')
    axes[1].plot(pos, theoretical,  '--', color='orange',  label='Briggs theoretical')
    axes[1].set_xlabel("Position from 3' end")
    axes[1].set_ylabel('G -> A frequency')
    axes[1].set_title("(B) 3' end")
    axes[1].legend()
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    out_png = OUT_DIR / 'figures' / 'damage_stats.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure  -> {out_png}')

    # Text summary
    lines = [
        'Damage profile verification',
        '=' * 60,
        f'Test reads: {N:,}',
        f'  bacterial: {is_bact.sum():,}',
        f'  human:     {(sources == 1).sum():,}',
        f'  env:       {(sources == 2).sum():,}',
        '',
        f'Reads with any C->T or G->A event: {has_dmg.sum():,} '
        f'({100*has_dmg.mean():.2f}%)',
        '',
        "5' C->T frequency by position (bacterial reads, first 10 positions):",
    ]
    for k in range(10):
        lines.append(f'  pos {k:>2}: measured {ct_5p[k]:.4f}   '
                     f'theoretical {theoretical_briggs(np.array([k]))[0]:.4f}')

    lines.append('')
    lines.append("3' G->A frequency by distance from 3' end (first 10 positions):")
    for k in range(10):
        lines.append(f'  dist {k:>2}: measured {ga_3p[k]:.4f}   '
                     f'theoretical {theoretical_briggs(np.array([k]))[0]:.4f}')

    out_txt = OUT_DIR / 'results' / 'damage_stats.txt'
    out_txt.write_text('\n'.join(lines))
    print(f'Summary -> {out_txt}')
    print()
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
