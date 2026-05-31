"""11.3_contig_level_eval.py — Contig-level damage detection: pyDamage vs aggregated classifier."""
import numpy as np
import pandas as pd
import pysam
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
from project_root import project_root

ROOT = project_root()
NPZ  = ROOT / 'data/test.npz'
FASTA = ROOT / 'data/raw/test/damaged.fasta'
OUT_FILE = ROOT / 'outputs/results/contig_level_eval.txt'

MODES = {
    'oracle': {
        'bam': ROOT / 'data/pydamage/oracle/test_aligned.bam',
        'pyd': ROOT / 'data/pydamage/oracle/results/pydamage_results.csv',
    },
    'denovo': {
        'bam': ROOT / 'data/pydamage/denovo/test_aligned.bam',
        'pyd': ROOT / 'data/pydamage/denovo/results/pydamage_results.csv',
    },
}
# Two label definitions: "any" (contig has any ancient reads) and "enrich"
# (ancient fraction >= 25 %, the population prevalence).
LABEL_THRESH = 0.25


def read_name_to_idx_and_labels():
    """Map read_id → index in NPZ; labels[i]=1 iff read i carries damage."""
    d = np.load(NPZ)
    clean, damaged = d['clean'], d['damaged']
    ancient = ((clean != damaged) & (clean != 0)).any(axis=1).astype(np.int8)
    name_to_idx: dict[str, int] = {}
    with open(FASTA) as f:
        i = 0
        for line in f:
            if line.startswith('>'):
                name_to_idx[line[1:].split()[0]] = i
                i += 1
    return name_to_idx, ancient


def load_classifier_probs():
    """evo_full per-read ancient probability for every read in test.npz."""
    p = np.load(ROOT / 'outputs/results/classifier_probs_evo_full.npz')
    return p['probs_evo_full']  # shape (311995,)


def aggregate_per_contig(bam_path, name_to_idx, labels, probs):
    """Walk the BAM, group reads by contig, return per-contig stats."""
    per_contig: dict[str, dict] = {}
    bam = pysam.AlignmentFile(str(bam_path), 'rb')
    for rec in bam.fetch(until_eof=True):
        if rec.is_unmapped or rec.is_secondary or rec.is_supplementary:
            continue
        i = name_to_idx.get(rec.query_name)
        if i is None:
            continue
        contig = bam.get_reference_name(rec.reference_id)
        c = per_contig.setdefault(contig, dict(n=0, anc=0, prob_sum=0.0,
                                                prob_max=-1.0))
        c['n'] += 1
        c['anc'] += int(labels[i])
        c['prob_sum'] += float(probs[i])
        c['prob_max'] = max(c['prob_max'], float(probs[i]))
    bam.close()
    rows = []
    for contig, c in per_contig.items():
        if c['n'] == 0:
            continue
        rows.append({
            'contig': contig,
            'n_reads': c['n'],
            'ancient_frac': c['anc'] / c['n'],
            'prob_mean': c['prob_sum'] / c['n'],
            'prob_max': c['prob_max'],
        })
    return pd.DataFrame(rows)


def metrics(y_true, y_score, n_grid=200):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None
    roc = roc_auc_score(y_true, y_score)
    prc = average_precision_score(y_true, y_score)
    grid = np.linspace(y_score.min(), y_score.max(), n_grid)
    best_f1 = best_mcc = -1
    f1_thr = mcc_thr = None
    for tau in grid:
        pred = (y_score >= tau).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        p, r, f, _ = precision_recall_fscore_support(
            y_true, pred, average='binary', zero_division=0)
        mcc = matthews_corrcoef(y_true, pred)
        if f > best_f1:
            best_f1 = f; f1_thr = tau
        if mcc > best_mcc:
            best_mcc = mcc; mcc_thr = tau
    return dict(ROC=roc, PR=prc, F1=best_f1, F1_thr=f1_thr,
                MCC=best_mcc, MCC_thr=mcc_thr)


def eval_mode(mode: str, name_to_idx, labels, probs):
    cfg = MODES[mode]
    print(f'\n── mode={mode} ─────────────────────────────────────────')
    print(f'  Walking BAM: {cfg["bam"].relative_to(ROOT)}')
    per = aggregate_per_contig(cfg['bam'], name_to_idx, labels, probs)
    print(f'  Contigs with ≥1 aligned read: {len(per)}')

    pyd = pd.read_csv(cfg['pyd'])
    # Keep qvalue and pdj alongside predicted_accuracy and pvalue for the
    # binary call threshold (q ≤ 0.05, pdj ≤ 0.6, predicted_accuracy ≥ 0.67).
    keep_cols = ['reference', 'predicted_accuracy', 'pvalue', 'qvalue']
    if 'pdj' in pyd.columns:
        keep_cols.append('pdj')
    pyd = pyd[keep_cols].rename(columns={'reference': 'contig'})
    merged = per.merge(pyd, on='contig', how='inner')
    print(f'  Contigs with pyDamage prediction: {len(merged)}')

    # Drop contigs with too few reads — pyDamage itself drops these too;
    # too few reads makes the per-contig label noisy.
    merged = merged[merged['n_reads'] >= 10].reset_index(drop=True)
    print(f'  Contigs with ≥10 reads (kept):    {len(merged)}')
    if len(merged) < 5:
        print('  Not enough contigs for reliable metrics — skipping.')
        return None

    merged['label'] = (merged['ancient_frac'] >= LABEL_THRESH).astype(int)
    y = merged['label'].values
    print(f'  Ancient contigs (>= {LABEL_THRESH*100:.0f}% reads ancient): '
          f'{int(y.sum())} / {len(y)} ({y.mean()*100:.1f} %)')

    results = {}
    for name, score in [
        ('pyDamage predicted_accuracy', merged['predicted_accuracy'].values),
        ('pyDamage 1 - pvalue',          1 - merged['pvalue'].values),
        ('evo_full (mean per contig)',  merged['prob_mean'].values),
        ('evo_full (max per contig)',   merged['prob_max'].values),
    ]:
        r = metrics(y, score)
        if r is None:
            continue
        results[name] = r
        print(f'  {name:<32}  ROC={r["ROC"]:.3f}  PR={r["PR"]:.3f}  '
              f'MCC={r["MCC"]:.3f}  F1={r["F1"]:.3f}')

    # ── pyDamage binary call ────────────────────────────────────────────
    # Apply pyDamage's recommended filter (q ≤ 0.05, pdj ≤ 0.6,
    # predicted_accuracy ≥ 0.67) for its own "damaged" call.
    if 'qvalue' in merged.columns:
        borry_call = (
            (merged['qvalue'] <= 0.05) &
            (merged.get('pdj', 0.0) <= 0.6) &
            (merged['predicted_accuracy'] >= 0.67)
        ).astype(int).values
        tp = int(((borry_call == 1) & (y == 1)).sum())
        fp = int(((borry_call == 1) & (y == 0)).sum())
        fn = int(((borry_call == 0) & (y == 1)).sum())
        tn = int(((borry_call == 0) & (y == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        mcc_num = tp * tn - fp * fn
        mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0.0
        results['pyDamage (Borry 2021 call)'] = dict(
            ROC=float('nan'), PR=float('nan'),
            F1=f1, F1_thr=float('nan'),
            MCC=mcc, MCC_thr=float('nan'),
            precision=prec, recall=rec,
            n_called=int(borry_call.sum()),
            tp=tp, fp=fp, fn=fn, tn=tn,
        )
        print(f'  {"pyDamage (Borry 2021 call)":<32}  '
              f'P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  MCC={mcc:.3f}  '
              f'(called {borry_call.sum()} contigs ancient)')
    return dict(mode=mode, n_contigs=len(merged), n_ancient=int(y.sum()),
                results=results)


def write_summary(all_results):
    lines = ['Contig-level evaluation: pyDamage vs evo_full classifier',
             '=' * 72,
             '',
             'Setup',
             '-----',
             '  Goal: evaluate pyDamage on the level it was designed for',
             '  (one prediction per contig) and aggregate our per-read',
             '  classifier to the same level via mean-pool.',
             '',
             '  Contig label : "ancient" if >= 25 % of the reads aligning',
             '                 to the contig carry a realised damage event',
             '                 (matches the population prevalence; a',
             '                 labelled-ancient contig is therefore',
             '                 enriched over the background).',
             '  Reads/contig : minimum 10 (pyDamage drops sparse contigs).',
             '']
    for r in all_results:
        if r is None:
            continue
        lines.append(f'Mode = {r["mode"]}')
        lines.append('-' * 72)
        lines.append(f'  Contigs scored: {r["n_contigs"]}   '
                     f'Ancient contigs: {r["n_ancient"]}')
        lines.append(f'  {"Method":<34} {"ROC-AUC":>8} {"PR-AUC":>8} '
                     f'{"MCC":>7} {"F1":>7}')
        lines.append('  ' + '-' * 70)
        for name, m in r['results'].items():
            lines.append(f'  {name:<34} {m["ROC"]:>8.3f} {m["PR"]:>8.3f} '
                         f'{m["MCC"]:>7.3f} {m["F1"]:>7.3f}')
        lines.append('')
    lines += [
        'Interpretation',
        '--------------',
        '  At the contig level, pyDamage is given the format it was',
        '  designed for. Aggregating evo_full to contig level via',
        '  mean-pool puts our classifier on the same footing. The',
        '  read-level evaluation in Table 5.5 understated pyDamage',
        '  because it forced a contig-level tool into per-read output.',
    ]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nWrote → {OUT_FILE}')


def main():
    print('Loading labels & classifier probabilities...')
    name_to_idx, labels = read_name_to_idx_and_labels()
    probs = load_classifier_probs()
    assert probs.shape[0] == labels.shape[0]
    print(f'  {len(name_to_idx):,} reads, {int(labels.sum()):,} ancient '
          f'({labels.mean()*100:.1f} %)')

    out = []
    for mode in MODES:
        if not MODES[mode]['bam'].exists():
            print(f'  Skipping {mode}: BAM missing')
            continue
        if not MODES[mode]['pyd'].exists():
            print(f'  Skipping {mode}: pyDamage CSV missing')
            continue
        out.append(eval_mode(mode, name_to_idx, labels, probs))

    write_summary(out)


if __name__ == '__main__':
    main()
