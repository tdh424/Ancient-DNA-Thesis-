"""11_baselines_compare.py — Compare reference-free baselines against the ML classifier variants."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_auc_score, roc_curve,
                             average_precision_score, precision_recall_curve,
                             confusion_matrix)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path('data')
OUT_DIR    = Path('outputs')
ML_PROBS   = OUT_DIR / 'results' / 'classifier_probs.npz'
TEST_FASTA = DATA_DIR / 'raw' / 'test' / 'damaged.fasta'

# ── Briggs damage parameters — must match Code/3_simulate.sh exactly ──
BRIGGS_V = 0.0241    # nick frequency (ν)
BRIGGS_L = 0.3590    # geometric decay constant (λ)
BRIGGS_D = 0.00937   # double-strand interior deamination rate (δ_d)
BRIGGS_S = 0.6815    # single-strand overhang deamination rate (δ_s)
SEQ_ERR  = 0.01      # sequencing error rate assumed by the LLR (Briggs default)

# Vocab: PAD=0 A=1 C=2 G=3 T=4 N=5
IDX_TO_BASE = {0: 'N', 1: 'A', 2: 'C', 3: 'G', 4: 'T', 5: 'N'}

# ML model display names — must match keys saved by 9_classifier.py
ML_MODELS = {
    'probs_evo_full': 'Evo2 (per-base + read LL)',
    'probs_evo_base': 'Evo2 (per-base)',
    'probs_seq':      'Seq-only',
}

COLORS = {
    'Evo2 (per-base + read LL)': '#1a7a49',   # dark green
    'Evo2 (per-base)':           '#2ecc71',   # green
    'Seq-only':                  '#3498db',   # blue
    'Briggs LLR':                '#e74c3c',   # red
    'pyDamage':                  '#f39c12',   # orange
}

LINE_STYLES = {
    'Evo2 (per-base + read LL)': '-',
    'Evo2 (per-base)':           '--',
    'Seq-only':                  '-.',
    'Briggs LLR':                ':',
    'pyDamage':                  ':',
}


# ── Briggs LLR ────────────────────────────────────────────────────────────────

def briggs_damage_prob(k, s=BRIGGS_S, lam=BRIGGS_L, d=BRIGGS_D):
    """
    P(deamination at distance k from read end).
    p(k) = s·λ^k + d   (k=0 is the terminal position).
    """
    return s * (lam ** k) + d


def compute_briggs_scores(damaged_arr, lengths_arr):
    """
    Vectorised reference-free Briggs LLR for all reads.
    LLR = Σ_k log[P(T_k | ancient) / P(T_k | modern)]
        + Σ_k log[P(A_k | ancient) / P(A_k | modern)]
    where P(T_k | ancient) = p_CT(k) + (1-p_CT(k))·ε
          P(T_k | modern)  = ε
    """
    N, MAX_LEN = damaged_arr.shape
    pos    = np.arange(MAX_LEN)
    scores = np.zeros(N, dtype=np.float64)

    # C→T at 5' end
    for k in range(MAX_LEN):
        p_ct  = briggs_damage_prob(k)
        valid = pos[k] < lengths_arr
        p_anc = p_ct + (1 - p_ct) * SEQ_ERR
        mask  = (damaged_arr[:, k] == 4) & valid
        scores += np.where(mask, np.log(p_anc / SEQ_ERR), 0.0)

    # G→A at 3' end
    for k in range(MAX_LEN):
        dist3 = lengths_arr - 1 - k
        valid = (dist3 >= 0) & (k < lengths_arr)
        p_ga  = BRIGGS_S * (BRIGGS_L ** np.maximum(dist3, 0)) + BRIGGS_D
        p_anc = p_ga + (1 - p_ga) * SEQ_ERR
        mask  = (damaged_arr[:, k] == 1) & valid
        scores += np.where(mask, np.log(p_anc / SEQ_ERR), 0.0)

    return scores


# ── pyDamage: parse CSV and match reads via FASTA headers ─────────────────────

def parse_fasta_headers(fasta_path):
    """Returns dict: sequence_string → contig_accession."""
    seq_to_contig = {}
    with open(fasta_path) as f:
        header, seq_parts = None, []
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if header and seq_parts:
                    seq_to_contig[''.join(seq_parts)] = header
                header    = line[1:].split(':')[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header and seq_parts:
            seq_to_contig[''.join(seq_parts)] = header
    return seq_to_contig


def tokens_to_seq(row, length):
    return ''.join(IDX_TO_BASE.get(int(t), 'N') for t in row[:length])


def match_npz_to_contigs(damaged_arr, lengths_arr, fasta_path):
    """Match each NPZ read to its source contig by sequence identity."""
    print('Parsing FASTA headers...')
    seq_to_contig = parse_fasta_headers(fasta_path)
    print(f'  {len(seq_to_contig):,} reads in FASTA')
    N          = len(lengths_arr)
    contig_ids = [None] * N
    misses     = 0
    for i in range(N):
        seq = tokens_to_seq(damaged_arr[i], lengths_arr[i])
        c   = seq_to_contig.get(seq)
        if c is None:
            misses += 1
        contig_ids[i] = c
    print(f'  Matched {N-misses:,} / {N:,} reads ({misses:,} unmatched)')
    return contig_ids


def load_pydamage_results(csv_path, contig_ids):
    """Load pyDamage CSV, assign per-contig predicted_accuracy to each read."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    col = 'predicted_accuracy'
    if col not in df.columns:
        candidates = [c for c in df.columns if 'accuracy' in c.lower() or 'damage' in c.lower()]
        if not candidates:
            raise ValueError(f'Cannot find accuracy column. Columns: {list(df.columns)}')
        col = candidates[0]
    contig_to_prob = dict(zip(df['reference'].astype(str), df[col].astype(float)))
    print(f'  pyDamage: {len(df):,} contigs scored')
    N     = len(contig_ids)
    probs = np.full(N, np.nan)
    for i, cid in enumerate(contig_ids):
        if cid and cid in contig_to_prob:
            probs[i] = contig_to_prob[cid]
    valid = ~np.isnan(probs)
    print(f'  {valid.sum():,} / {N:,} reads have a pyDamage score')
    return probs, valid


# ── Metrics ───────────────────────────────────────────────────────────────────

def best_mcc_threshold(probs, labels, n_steps=201):
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t, best_mcc = 0.5, -2.0
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
        denom = np.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
        mcc   = (float(tp)*float(tn) - float(fp)*float(fn)) / denom if denom > 0 else 0.0
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return float(best_t)


def stats_at(probs, labels, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2*ppv*sens / (ppv+sens) if (ppv+sens) > 0 else 0.0
    bal  = (sens + spec) / 2
    denom = np.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
    mcc   = (float(tp)*float(tn) - float(fp)*float(fn)) / denom if denom > 0 else 0.0
    return dict(sens=sens, spec=spec, ppv=ppv, f1=f1, bal_acc=bal, mcc=mcc,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pydamage',  type=Path, default=None,
                        help='pydamage_results.csv from MODE=oracle')
    parser.add_argument('--pydamage-denovo', dest='pydamage_denovo', type=Path, default=None,
                        help='pydamage_results.csv from MODE=denovo (MetaSPAdes)')
    parser.add_argument('--pmdtools',  type=Path, default=None,
                        help='Path to pmdtools_scores.csv (Code/10.2_run_read_baselines.sh)')
    args = parser.parse_args()

    (OUT_DIR / 'figures').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'results').mkdir(parents=True, exist_ok=True)

    # ── Load test data ────────────────────────────────────────────────────────
    test_npz = DATA_DIR / 'test.npz'
    d       = np.load(test_npz)
    damaged = d['damaged'].astype(np.int32)
    clean   = d['clean'].astype(np.int32)
    lengths = d['lengths'].astype(np.int32)
    labels  = ((damaged != clean) & (clean != 0)).any(axis=1).astype(np.int32)
    N       = len(labels)
    prev    = labels.mean()
    print(f'Test set: {N:,} reads  ({prev*100:.1f}% ancient)')

    # ── Briggs LLR ────────────────────────────────────────────────────────────
    print('\nComputing Briggs LLR scores...')
    llr_raw      = compute_briggs_scores(damaged, lengths)
    probs_briggs = 1.0 / (1.0 + np.exp(-llr_raw * 0.1))   # sigmoid scaling
    roc_b = roc_auc_score(labels, probs_briggs)
    pr_b  = average_precision_score(labels, probs_briggs)
    print(f'  Briggs LLR — ROC-AUC: {roc_b:.4f}  PR-AUC: {pr_b:.4f}')

    # ── Load ML classifier probabilities ──────────────────────────────────────
    # Order: baselines first (will render behind ML models), then ML models on top
    models = {}   # display_name → probs array
    if ML_PROBS.exists():
        ml = np.load(ML_PROBS)
        if len(ml['labels']) != N:
            print(f'  WARNING: ML probs have {len(ml["labels"]):,} reads but test.npz has {N:,}.'
                  f' Skipping ML comparison.')
        else:
            for key, display_name in ML_MODELS.items():
                if key in ml:
                    models[display_name] = ml[key]
    else:
        print(f'  No ML probs found at {ML_PROBS}. Run 9_classifier.py first.')

    models['Briggs LLR'] = probs_briggs

    # ── pyDamage (optional) ───────────────────────────────────────────────────
    if args.pydamage and args.pydamage.exists():
        print(f'\nLoading pyDamage results from {args.pydamage}...')
        if not TEST_FASTA.exists():
            print(f'  WARNING: {TEST_FASTA} not found — cannot match reads to contigs.')
        else:
            contig_ids = match_npz_to_contigs(damaged, lengths, TEST_FASTA)
            pd_probs, valid = load_pydamage_results(args.pydamage, contig_ids)
            if valid.sum() > 100:
                p_s, l_s = pd_probs[valid], labels[valid]
                print(f'  pyDamage — ROC-AUC: {roc_auc_score(l_s, p_s):.4f}'
                      f'  PR-AUC: {average_precision_score(l_s, p_s):.4f}'
                      f'  ({valid.sum():,} reads)')
                models['pyDamage'] = pd_probs
    elif args.pydamage:
        print(f'  pyDamage CSV not found at {args.pydamage}. Run run_pydamage.sh first.')

    if args.pydamage_denovo and args.pydamage_denovo.exists():
        print(f'\nLoading de novo pyDamage results from {args.pydamage_denovo}...')
        if TEST_FASTA.exists():
            contig_ids = match_npz_to_contigs(damaged, lengths, TEST_FASTA)
            pd_probs, valid = load_pydamage_results(args.pydamage_denovo, contig_ids)
            if valid.sum() > 100:
                p_s, l_s = pd_probs[valid], labels[valid]
                print(f'  pyDamage (de novo) — ROC-AUC: {roc_auc_score(l_s, p_s):.4f}'
                      f'  PR-AUC: {average_precision_score(l_s, p_s):.4f}'
                      f'  ({valid.sum():,} reads)')
                models['pyDamage (de novo)'] = pd_probs

    # ── PMDtools (read-level, reference-based) ───────────────────────────────
    def load_read_csv(csv_path, score_col, fasta_path):
        """Match per-read scores to NPZ rows by sequence identity."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        if score_col not in df.columns:
            cands = [c for c in df.columns if c.lower() != 'read_id']
            if not cands:
                raise ValueError(f'No score column in {csv_path}')
            score_col = cands[0]
        rid_to_score = dict(zip(df['read_id'].astype(str),
                                df[score_col].astype(float)))
        seq_to_rid   = parse_fasta_headers(fasta_path)
        N = len(damaged); probs = np.full(N, np.nan)
        for i in range(N):
            seq = tokens_to_seq(damaged[i], lengths[i])
            rid = seq_to_rid.get(seq)
            if rid and rid in rid_to_score:
                probs[i] = rid_to_score[rid]
        valid = ~np.isnan(probs)
        return probs, valid

    for name, path, score_col, color, ls in [
        ('PMDtools',  args.pmdtools,  'pmds', '#8e44ad', ':'),
    ]:
        if path and path.exists():
            print(f'\nLoading {name} results from {path}...')
            if not TEST_FASTA.exists():
                print(f'  WARNING: {TEST_FASTA} not found — cannot match reads.')
                continue
            p, valid = load_read_csv(path, score_col, TEST_FASTA)
            if valid.sum() > 100:
                p_s, l_s = p[valid], labels[valid]
                print(f'  {name} — ROC-AUC: {roc_auc_score(l_s, p_s):.4f}'
                      f'  PR-AUC: {average_precision_score(l_s, p_s):.4f}'
                      f'  ({valid.sum():,} reads)')
                models[name]      = p
                COLORS[name]      = color
                LINE_STYLES[name] = ls
        elif path:
            print(f'  {name} CSV not found at {path}.')

    # Display order: best ML models first, baselines last
    order = [n for n in [*ML_MODELS.values(),
                         'Briggs LLR', 'pyDamage', 'PMDtools']
             if n in models]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle('aDNA Classifier — ML Models vs Baselines',
                 fontsize=13, fontweight='bold', y=1.01)

    # Panel 1 — ROC
    ax = axes[0]
    for name in order:
        p = models[name]
        mask = ~np.isnan(p) if np.isnan(p).any() else np.ones(N, bool)
        fpr, tpr, _ = roc_curve(labels[mask], p[mask])
        auc = roc_auc_score(labels[mask], p[mask])
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, ls=LINE_STYLES[name],
                label=f'{name}  (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.25, lw=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=11)
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel 2 — Precision-Recall
    ax = axes[1]
    ax.axhline(prev, color='k', ls='--', alpha=0.25, lw=1,
               label=f'Random (prev={prev:.2f})')
    for name in order:
        p = models[name]
        mask = ~np.isnan(p) if np.isnan(p).any() else np.ones(N, bool)
        prec, rec, _ = precision_recall_curve(labels[mask], p[mask])
        ap = average_precision_score(labels[mask], p[mask])
        ax.plot(rec, prec, color=COLORS[name], lw=2, ls=LINE_STYLES[name],
                label=f'{name}  (AP={ap:.3f})')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve', fontsize=11)
    ax.legend(fontsize=7.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Panel 3 — Bar chart at best-MCC threshold
    ax    = axes[2]
    mkeys = ['roc_auc', 'pr_auc', 'sens', 'spec', 'ppv', 'f1', 'bal_acc', 'mcc']
    mlbls = ['ROC-AUC', 'PR-AUC', 'Sens', 'Spec', 'PPV', 'F1', 'BalAcc', 'MCC']
    x     = np.arange(len(mkeys))
    width = 0.8 / len(order)
    for i, name in enumerate(order):
        p    = models[name]
        mask = ~np.isnan(p) if np.isnan(p).any() else np.ones(N, bool)
        p_s, l_s = p[mask], labels[mask]
        t    = best_mcc_threshold(p_s, l_s)
        s    = stats_at(p_s, l_s, t)
        roc  = roc_auc_score(l_s, p_s)
        pra  = average_precision_score(l_s, p_s)
        vals = [roc, pra, s['sens'], s['spec'], s['ppv'], s['f1'], s['bal_acc'], s['mcc']]
        bars = ax.bar(x + (i - len(order)/2 + 0.5)*width, vals,
                      width, label=name, color=COLORS[name], alpha=0.85,
                      linestyle='-')
        for b in bars:
            v = b.get_height()
            if v > 0.04:
                ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(mlbls, fontsize=8.5, rotation=25, ha='right')
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Metrics at best-MCC threshold', fontsize=11)
    ax.legend(fontsize=7)
    ax.axhline(0.5, color='grey', ls=':', alpha=0.4, lw=1)

    plt.tight_layout()
    out_fig = OUT_DIR / 'figures' / 'baselines_comparison.png'
    fig.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nFigure → {out_fig}')

    # ── Text summary ──────────────────────────────────────────────────────────
    W   = 9
    col = ['ROC-AUC', 'PR-AUC', 'Sens', 'Spec', 'PPV', 'F1', 'BalAcc', 'MCC', 'MCC-thr']
    hdr = f'{"Model":<35}' + ''.join(f'  {c:>{W}}' for c in col)
    sep = '-' * len(hdr)

    def row(name, p, l):
        t   = best_mcc_threshold(p, l)
        s   = stats_at(p, l, t)
        roc = roc_auc_score(l, p)
        pra = average_precision_score(l, p)
        vals = [roc, pra, s['sens'], s['spec'], s['ppv'], s['f1'], s['bal_acc'], s['mcc'], t]
        return f'  {name:<33}' + ''.join(f'  {v:>{W}.4f}' for v in vals)

    lines = [
        'aDNA Classifier — ML Models vs Baselines',
        '=' * len(hdr),
        f'Test set: {N:,} reads  ({prev*100:.1f}% ancient / {(1-prev)*100:.1f}% modern)',
        f'Random PR-AUC baseline = {prev:.4f}  (class prevalence)',
        '',
        'Briggs LLR formula (reference-free, Briggs 2007):',
        '  p_CT(k) = s·λ^k + d  (C→T rate at 5\'-position k)',
        '  p_GA(k) = s·λ^k + d  (G→A rate at 3\'-position k)',
        f'  Parameters: s={BRIGGS_S}, λ={BRIGGS_L}, d={BRIGGS_D}, v={BRIGGS_V}',
        '',
        'Note: PMDtools conditions on the reference base (C or G) at each',
        '  aligned position. The reference-free version used here scores ALL T/A',
        '  positions, which is noisier but requires no prior mapping step.',
        '',
        '── At best-MCC threshold (per model) ──',
        hdr, sep,
    ]
    for name in order:
        p    = models[name]
        mask = ~np.isnan(p) if np.isnan(p).any() else np.ones(N, bool)
        if mask.sum() < N:
            sub_prev = labels[mask].mean()
            suffix = (f'  [evaluated on aligned subset: {mask.sum():,}/{N:,} '
                      f'reads, prevalence {sub_prev*100:.1f}%]')
        else:
            suffix = ''
        lines.append(row(name, p[mask], labels[mask]) + suffix)
    lines += [
        '',
        'Note on [evaluated on aligned subset] rows:',
        '  PMDtools only scores reads that align to the oracle',
        '  reference (~63 % of the test set). The ROC/PR-AUC for those rows',
        '  is therefore on the aligned subset, which has a different ancient',
        '  prevalence (~39 %) than the full test set (25 %). Use the standalone',
        '  pmdtools_oracle.txt for the authoritative PMDtools numbers and',
        '  apples-to-apples comparison via the per-read subset evaluation in',
        '  Section 5.5 of the thesis.',
    ]

    txt = '\n'.join(lines)
    out_txt = OUT_DIR / 'results' / 'baselines_comparison.txt'
    out_txt.write_text(txt)
    print(f'Summary → {out_txt}')
    print()
    print(txt)


if __name__ == '__main__':
    main()
