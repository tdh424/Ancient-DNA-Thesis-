"""2_cluster_genomes.py — Cluster genomes by Mash distance and assign whole clusters to train/val/test."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

GENOME_DIR = Path('data/genomes/ncbi_dataset/data')
OUT_TSV    = Path('data/genomes/split_assignment.tsv')
TMP_DIR    = Path('data/genomes/_mash_tmp')

TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10   # test gets the remaining 0.10


def find_genomes():
    paths = sorted(GENOME_DIR.glob('*/*.fna'))
    if not paths:
        sys.exit(f'ERROR: no .fna files in {GENOME_DIR}')
    return paths


def genome_size(path):
    n = 0
    with open(path) as f:
        for line in f:
            if not line.startswith('>'):
                n += len(line.strip())
    return n


def run_mash(paths, threshold, kmer=21, sketch=1000):
    if shutil.which('mash') is None:
        sys.exit('ERROR: `mash` not on PATH. Install via:\n'
                 '    conda install -n ancient-dna -c bioconda mash')

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    sketch_path = TMP_DIR / 'all.msh'

    print(f'Mash sketch: k={kmer}, sketch size={sketch}, '
          f'{len(paths)} genomes...', flush=True)
    cmd = ['mash', 'sketch', '-k', str(kmer), '-s', str(sketch),
           '-o', str(sketch_path.with_suffix(''))]
    cmd.extend(str(p) for p in paths)
    subprocess.run(cmd, check=True)   # let mash log to stderr — useful on failure

    print('Mash dist (pairwise)...', flush=True)
    out = subprocess.run(
        ['mash', 'dist', str(sketch_path), str(sketch_path)],
        check=True, capture_output=True, text=True,
    ).stdout

    # Build distance matrix
    idx = {str(p): i for i, p in enumerate(paths)}
    n   = len(paths)
    D   = np.zeros((n, n), dtype=np.float32)
    for line in out.splitlines():
        a, b, dist, *_ = line.split('\t')
        i, j = idx.get(a), idx.get(b)
        if i is None or j is None:
            continue
        D[i, j] = float(dist)
    D = np.minimum(D, D.T)   # ensure symmetry
    return D


def single_linkage_clusters(D, threshold):
    """Union-find: connect any pair with distance < threshold."""
    n = D.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < threshold:
                union(i, j)

    roots = [find(i) for i in range(n)]
    uniq  = sorted(set(roots))
    cid   = {r: k for k, r in enumerate(uniq)}
    return np.array([cid[r] for r in roots], dtype=np.int32)


def greedy_split(cluster_ids, sizes, train_frac=0.80, val_frac=0.10, seed=42):
    """Greedy bin-packing: assign clusters (sorted by size desc) to whichever
    split is most below its target."""
    rng = np.random.default_rng(seed)
    n_clusters = cluster_ids.max() + 1
    cluster_sizes = np.zeros(n_clusters, dtype=np.int64)
    for ci, sz in zip(cluster_ids, sizes):
        cluster_sizes[ci] += sz
    total = cluster_sizes.sum()

    target = {'train': train_frac * total,
              'val'  : val_frac   * total,
              'test' : (1 - train_frac - val_frac) * total}
    have   = {'train': 0, 'val': 0, 'test': 0}
    cluster_split = {}

    # Big clusters first — they constrain layout most
    order = np.argsort(-cluster_sizes)
    # Tie-break with rng to avoid deterministic pathological packing
    rng.shuffle(order[cluster_sizes[order] == cluster_sizes[order[0]]])
    for k in order:
        deficit = {sp: target[sp] - have[sp] for sp in target}
        sp      = max(deficit, key=deficit.get)
        cluster_split[int(k)] = sp
        have[sp] += int(cluster_sizes[k])

    splits = np.array([cluster_split[int(c)] for c in cluster_ids])
    return splits, have, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.05,
                    help='Mash distance threshold (default 0.05 ≈ 95%% ANI)')
    ap.add_argument('--train-frac', type=float, default=TRAIN_FRAC)
    ap.add_argument('--val-frac',   type=float, default=VAL_FRAC)
    args = ap.parse_args()

    paths = find_genomes()
    print(f'Found {len(paths)} genomes')

    sizes = np.array([genome_size(p) for p in paths], dtype=np.int64)
    print(f'Total bases: {sizes.sum() / 1e9:.2f} Gb')

    D = run_mash(paths, args.threshold)
    print(f'Distance matrix: {D.shape}, '
          f'min={D[D>0].min():.4f}, mean={D[D>0].mean():.4f}')

    clusters = single_linkage_clusters(D, args.threshold)
    n_clusters = clusters.max() + 1
    print(f'Clusters at threshold {args.threshold}: {n_clusters} '
          f'(largest = {np.bincount(clusters).max()} genomes)')

    splits, have, target = greedy_split(clusters, sizes,
                                        args.train_frac, args.val_frac)

    print('\nSplit summary (target → actual):')
    for sp in ['train', 'val', 'test']:
        n = (splits == sp).sum()
        print(f'  {sp:5s}: {n:>4} genomes  '
              f'{have[sp] / 1e9:>6.2f} Gb '
              f'(target {target[sp] / 1e9:>5.2f} Gb)')

    # Write split assignment
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TSV, 'w') as f:
        f.write('accession\tsplit\tcluster_id\tgenome_size_bp\n')
        for p, sp, ci, sz in zip(paths, splits, clusters, sizes):
            acc = p.parent.name
            f.write(f'{acc}\t{sp}\t{ci}\t{sz}\n')
    print(f'\nWritten: {OUT_TSV}')

    # Cleanup mash sketch
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)


if __name__ == '__main__':
    main()
