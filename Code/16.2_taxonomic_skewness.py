"""16.2_taxonomic_skewness.py — Figure of the genus-level distribution of the bacterial genomes."""
from collections import Counter
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from project_root import project_root

ROOT       = project_root()
GENOME_DIR = ROOT / 'data/genomes/ncbi_dataset/data'
OUT_FIG    = ROOT / 'outputs/figures/taxonomic_skewness_comparison.png'
THESIS_FIG = ROOT / 'thesis/figures/taxonomic_skewness_comparison.png'


def extract_genus(fna_path):
    """Pick the genus from the FASTA header organism name."""
    with open(fna_path) as f:
        header = f.readline().strip()
    if not header.startswith('>'):
        return None
    parts = header[1:].split()
    if len(parts) < 2:
        return None
    return parts[1]


def main():
    genera = Counter()
    for fna in sorted(GENOME_DIR.glob('*/*.fna')):
        g = extract_genus(fna)
        if g is not None:
            genera[g] += 1

    n_total = sum(genera.values())
    n_genera = len(genera)
    counts = np.array(sorted(genera.values(), reverse=True))

    print(f'Total genomes : {n_total}')
    print(f'Distinct genera: {n_genera}')
    print(f'Max per genus  : {counts[0]}')
    print(f'Median per genus: {int(np.median(counts))}')
    print()
    print('Top 10:')
    for g, n in genera.most_common(10):
        print(f'  {n:>3}  {g}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── Panel 1: ranked count per genus ─────────────────────────────────
    ax = axes[0]
    x = np.arange(len(counts))
    ax.bar(x, counts, color='#5b9bd5', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Genus (ranked by genome count)')
    ax.set_ylabel('Number of genomes')
    ax.set_title('Genome count per genus (ranked)')
    # Annotate the top few
    for i in range(min(3, len(counts))):
        ax.text(i, counts[i] + 1, str(counts[i]),
                ha='center', va='bottom', fontsize=8, color='#1f4e79')
    ax.set_xlim(-0.5, len(counts) - 0.5)
    ax.text(0.97, 0.95,
            f'{n_total} genomes\n{n_genera} genera\nmax = {counts[0]}, median = {int(np.median(counts))}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color='#444',
            bbox=dict(facecolor='white', edgecolor='#ccc', boxstyle='round,pad=0.4'))

    # ── Panel 2: histogram of genus sizes ───────────────────────────────
    ax = axes[1]
    max_count = counts.max()
    bin_edges = np.arange(0.5, max_count + 1.5, 1)
    ax.hist(counts, bins=bin_edges, color='#ed7d31',
            edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Genomes per genus')
    ax.set_ylabel('Number of genera')
    ax.set_title('Distribution of genus sizes')
    ax.set_yscale('log')
    ax.set_xlim(0, max_count + 1)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure → {OUT_FIG}')

    if THESIS_FIG.parent.exists():
        shutil.copy(OUT_FIG, THESIS_FIG)
        print(f'Copied → {THESIS_FIG}')


if __name__ == '__main__':
    main()
