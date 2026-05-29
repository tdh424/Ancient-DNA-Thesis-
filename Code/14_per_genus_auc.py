"""
14_per_genus_auc.py — Per-cluster classifier performance.

Cross-references test-set predictions with the Mash cluster assignments in
data/genomes/split_assignment.tsv to measure whether the classifier performs
uniformly across phylogenetic groups, or whether some clusters are easier or
harder than others.

Mapping reads back to source genomes uses the BAM file produced by the
oracle-mode pyDamage benchmark (data/pydamage/oracle/test_aligned.bam) —
each aligned read's reference contig name resolves to an accession,
which maps to a cluster.

Outputs:
    outputs/results/per_cluster_auc.txt
    outputs/figures/per_cluster_auc.png
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

DATA_DIR  = Path('data')
OUT_DIR   = Path('outputs')
SPLIT_TSV  = DATA_DIR / 'genomes' / 'split_assignment.tsv'
GENOME_DIR = DATA_DIR / 'genomes' / 'ncbi_dataset' / 'data'
BAM_FILE   = DATA_DIR / 'pydamage' / 'oracle' / 'test_aligned.bam'
PROBS_NPZ  = OUT_DIR / 'results' / 'classifier_probs.npz'


def load_accession_to_cluster():
    """Parse split_assignment.tsv → dict mapping accession → (cluster_id, split)."""
    mapping = {}
    with open(SPLIT_TSV) as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            acc, split, cluster_id = parts[0], parts[1], int(parts[2])
            mapping[acc] = (cluster_id, split)
    return mapping


def build_seqid_to_gcf():
    """BAM references reads by NCBI sequence ID (e.g. NC_000917.1), but
    split_assignment.tsv keys on GCF assembly ID. Build the mapping by parsing
    FASTA headers in each genome dir."""
    seqid_to_gcf = {}
    for gcf_dir in GENOME_DIR.iterdir():
        if not gcf_dir.is_dir():
            continue
        gcf = gcf_dir.name
        fna = gcf_dir / f'{gcf}.fna'
        if not fna.exists():
            continue
        with open(fna) as f:
            for line in f:
                if line.startswith('>'):
                    seqid = line[1:].split()[0]
                    seqid_to_gcf[seqid] = gcf
    return seqid_to_gcf


def map_reads_to_clusters(bam_path, n_reads, acc_to_cluster, seqid_to_gcf):
    """For each aligned read, return its source genome's cluster id (-1 if unmapped)."""
    try:
        import pysam
    except ImportError:
        raise SystemExit('pysam required: pip install pysam')

    cluster_per_read = np.full(n_reads, -1, dtype=np.int32)
    with pysam.AlignmentFile(str(bam_path), 'rb') as bam:
        for read_idx, aln in enumerate(bam.fetch(until_eof=True)):
            if aln.is_unmapped or read_idx >= n_reads:
                continue
            seqid = aln.reference_name or ''
            gcf = seqid_to_gcf.get(seqid)
            if gcf and gcf in acc_to_cluster:
                cluster_per_read[read_idx] = acc_to_cluster[gcf][0]
    return cluster_per_read


def main():
    if not SPLIT_TSV.exists():
        raise SystemExit(f'{SPLIT_TSV} not found — run Code/2_cluster_genomes.py first.')
    if not BAM_FILE.exists():
        raise SystemExit(f'{BAM_FILE} not found — run Code/10.1_run_pydamage.sh first.')
    if not PROBS_NPZ.exists():
        raise SystemExit(f'{PROBS_NPZ} not found — train the classifier first.')

    probs    = np.load(PROBS_NPZ)
    labels   = probs['labels']
    N        = len(labels)
    acc_map  = load_accession_to_cluster()
    print(f'Loaded {len(acc_map):,} GCF → cluster mappings')

    seqid_to_gcf = build_seqid_to_gcf()
    print(f'Built {len(seqid_to_gcf):,} sequence-ID → GCF lookup from FASTA headers')

    print(f'Mapping {N:,} test reads to clusters via BAM...')
    cluster = map_reads_to_clusters(BAM_FILE, N, acc_map, seqid_to_gcf)
    n_mapped = int((cluster >= 0).sum())
    print(f'  {n_mapped:,} of {N:,} reads mapped ({100*n_mapped/N:.1f}%)')

    # Group reads by cluster and compute AUC where the cluster has both classes
    by_cluster = defaultdict(list)
    for idx, cid in enumerate(cluster):
        if cid >= 0:
            by_cluster[int(cid)].append(idx)

    # Use the best classifier variant for the per-cluster breakdown
    score_key = 'probs_evo_full' if 'probs_evo_full' in probs else (
        'probs_evo_base' if 'probs_evo_base' in probs else 'probs_seq'
    )
    scores = probs[score_key]
    print(f'Using {score_key} for per-cluster AUC')

    rows = []
    for cid, indices in by_cluster.items():
        idx = np.array(indices)
        y, p = labels[idx], scores[idx]
        if len(y) < 50 or y.sum() == 0 or y.sum() == len(y):
            continue
        auc = roc_auc_score(y, p)
        rows.append(dict(cluster=cid, n_reads=len(y), n_ancient=int(y.sum()),
                         auc=auc))

    rows.sort(key=lambda r: -r['auc'])
    print(f'\nEvaluated {len(rows)} clusters with mixed classes\n')

    lines = [
        'Per-cluster classifier AUC',
        '=' * 60,
        f'Scoring model: {score_key}',
        f'Reads mapped: {n_mapped:,} of {N:,}',
        f'Clusters evaluated: {len(rows)} (need >=50 reads with both classes)',
        '',
        f'  {"Cluster":>8} {"Reads":>8} {"Ancient":>8}  {"ROC-AUC":>8}',
        '  ' + '-' * 40,
    ]
    for r in rows:
        lines.append(f'  {r["cluster"]:>8} {r["n_reads"]:>8,} {r["n_ancient"]:>8,}'
                     f'  {r["auc"]:>8.4f}')

    aucs = np.array([r['auc'] for r in rows])
    if len(aucs) == 0:
        lines += ['', 'No clusters had >=50 reads with both classes — nothing to summarize.']
        out_txt = OUT_DIR / 'results' / 'per_cluster_auc.txt'
        out_txt.write_text('\n'.join(lines))
        print('\n'.join(lines))
        raise SystemExit('No clusters to plot.')
    lines += [
        '',
        f'AUC summary: mean = {aucs.mean():.4f}, median = {np.median(aucs):.4f}, '
        f'range = [{aucs.min():.4f}, {aucs.max():.4f}], '
        f'std = {aucs.std():.4f}',
    ]

    out_txt = OUT_DIR / 'results' / 'per_cluster_auc.txt'
    out_txt.write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f'\nWritten: {out_txt}')

    # Histogram + scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Classifier performance per Mash cluster')

    axes[0].hist(aucs, bins=20, color='#3498db', edgecolor='white')
    axes[0].axvline(aucs.mean(), color='red', ls='--', label=f'mean = {aucs.mean():.3f}')
    axes[0].set_xlabel('ROC-AUC')
    axes[0].set_ylabel('Number of clusters')
    axes[0].set_title('(A) Distribution of per-cluster AUC')
    axes[0].legend()

    axes[1].scatter([r['n_reads'] for r in rows], aucs, alpha=0.6, color='#2ecc71')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Reads in cluster (log)')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].set_title('(B) AUC vs cluster size')
    axes[1].axhline(0.5, color='grey', ls=':', alpha=0.5, label='random')
    axes[1].legend()

    plt.tight_layout()
    out_png = OUT_DIR / 'figures' / 'per_cluster_auc.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure : {out_png}')


if __name__ == '__main__':
    main()
