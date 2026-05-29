"""
8.1_cds_eval.py — Validate that bacterial reads carry a protein-coding signal.

Bacterial genomes are ~85 % coding sequence (CDS); reads sampled from them
should show a clear "is-coding" signature. The cleanest such signature is the
**best-frame stop-codon rate**: a random DNA sequence has 3/64 ≈ 4.7 % stop
codons in any frame, but a real CDS read in its correct reading frame has far
fewer. Three sanity checks make this concrete:

  1. **Per-frame stop rate** (bacterial clean reads). If the reads come from
     coding regions, frame 0, 1 or 2 — whichever matches the original CDS —
     should be much lower than the random baseline (~4.7 %), while the other
     two should sit near it.
  2. **Best-frame stop rate by source** (bact vs human vs random-shuffled).
     Bact reads come from genomes that are mostly CDS; human reads come from
     chr21, which is non-coding-dominated; random-shuffled reads are the
     pure-noise baseline. If our metric is real, the order should be
     bact < human < random.
  3. **Best-frame stop rate, bact reads only**: clean vs damaged vs
     random-shuffled. Damage pushes the rate up because C→T converts
     CAA / CAG / CGA to the stop codons TAA / TAG / TGA. If clean << damaged
     < random, the damage signal we are trying to detect is *causing* the
     coding signal to degrade.

Output → outputs/figures/cds_evaluation.png
         outputs/results/cds_summary.txt
"""

from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path('outputs/results')
DATA_DIR    = Path('data')
OUT_DIR     = Path('outputs')

# Base encoding: PAD=0 A=1 C=2 G=3 T=4 N=5
A, C, G, T = 1, 2, 3, 4
RANDOM_EXPECTED = 3.0 / 64.0   # 3 stop codons (TAA, TAG, TGA) out of 64

# Codon → AA byte lookup (used only to count stops here)
def _stop_mask():
    """Boolean (5,5,5) array — True at the three stop codons."""
    m = np.zeros((5, 5, 5), dtype=bool)
    for b1, b2, b3 in [(T, A, A), (T, A, G), (T, G, A)]:
        m[b1, b2, b3] = True
    return m

STOP_MASK = _stop_mask()


def log(msg=''):
    print(msg, flush=True)


def per_frame_stop_rate(reads, lengths):
    """Return (N, 3) matrix: per-read stop-codon rate in each forward frame."""
    N, L = reads.shape
    out = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        rlen = int(lengths[i])
        for f in range(3):
            n_codons = (rlen - f) // 3
            if n_codons == 0:
                continue
            end = f + n_codons * 3
            seg = np.clip(reads[i, f:end], 0, 4).reshape(n_codons, 3)
            stops = STOP_MASK[seg[:, 0], seg[:, 1], seg[:, 2]].sum()
            out[i, f] = stops / n_codons
    return out


def best_frame_stop_rate(per_frame):
    """For each read, pick the frame with the FEWEST stops; return that rate."""
    return per_frame.min(axis=1)


def shuffle_bases_within_read(reads, lengths, seed=42):
    """Per-read base shuffle — keeps composition, destroys codon structure.
    Used as a 'random DNA with same composition' negative control."""
    rng = np.random.default_rng(seed)
    out = reads.copy()
    for i in range(len(reads)):
        rlen = int(lengths[i])
        if rlen < 3:
            continue
        out[i, :rlen] = rng.permutation(out[i, :rlen])
    return out


def main():
    (OUT_DIR / 'figures').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'results').mkdir(parents=True, exist_ok=True)

    log('Loading test data...')
    # Need source labels — those live on the raw test NPZ, not on
    # outputs/results/test_denoised.npz.
    d_test = np.load(DATA_DIR / 'test.npz')
    clean    = d_test['clean'].astype(np.int32)
    damaged  = d_test['damaged'].astype(np.int32)
    lengths  = d_test['lengths'].astype(np.int32)
    sources  = d_test['sources']  # 0=bact, 1=human, 2=env
    N = len(clean)
    log(f'  {N:,} test reads  '
        f'(bact={int((sources==0).sum()):,}, '
        f'human={int((sources==1).sum()):,}, '
        f'env={int((sources==2).sum()):,})')

    bact_idx  = np.where(sources == 0)[0]
    human_idx = np.where(sources == 1)[0]

    log('Computing per-frame stop rates...')
    pf_bact_clean  = per_frame_stop_rate(clean[bact_idx],   lengths[bact_idx])
    pf_bact_dmg    = per_frame_stop_rate(damaged[bact_idx], lengths[bact_idx])
    pf_human_clean = per_frame_stop_rate(clean[human_idx],  lengths[human_idx])

    log('Building shuffled (random-DNA) baseline...')
    bact_shuf = shuffle_bases_within_read(
        clean[bact_idx], lengths[bact_idx], seed=42
    )
    pf_bact_shuf = per_frame_stop_rate(bact_shuf, lengths[bact_idx])

    # ── Headline numbers ─────────────────────────────────────────────────────
    bf_bact_clean  = best_frame_stop_rate(pf_bact_clean)
    bf_bact_dmg    = best_frame_stop_rate(pf_bact_dmg)
    bf_human_clean = best_frame_stop_rate(pf_human_clean)
    bf_bact_shuf   = best_frame_stop_rate(pf_bact_shuf)

    log('')
    log(f'Random-DNA expected stop rate:   {RANDOM_EXPECTED:.4f}')
    log(f'Bact clean   best-frame mean:    {bf_bact_clean.mean():.4f}')
    log(f'Bact damaged best-frame mean:    {bf_bact_dmg.mean():.4f}')
    log(f'Human clean  best-frame mean:    {bf_human_clean.mean():.4f}')
    log(f'Bact shuffled best-frame mean:   {bf_bact_shuf.mean():.4f}')

    # Per-frame breakdown (bact clean): if reads are CDS, the lowest-stop
    # frame should be far below the other two.
    pf_bact_clean_mean = pf_bact_clean.mean(axis=0)
    # For each read, rank the frames by stop count; report mean stop rate at
    # rank 0 (best), rank 1, rank 2.
    pf_bact_clean_sorted = np.sort(pf_bact_clean, axis=1).mean(axis=0)
    log('\nBact clean — mean stop rate per frame (sorted within read):')
    log(f'  best={pf_bact_clean_sorted[0]:.4f}  '
        f'mid={pf_bact_clean_sorted[1]:.4f}  '
        f'worst={pf_bact_clean_sorted[2]:.4f}')

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1 — Per-frame stop rate, bact clean
    # The lowest bar should be far below the random expectation if the reads
    # are protein-coding.
    ax = axes[0]
    pf_clean_sorted = np.sort(pf_bact_clean, axis=1).mean(axis=0)
    pf_dmg_sorted   = np.sort(pf_bact_dmg,   axis=1).mean(axis=0)
    pf_shuf_sorted  = np.sort(pf_bact_shuf,  axis=1).mean(axis=0)
    xp = np.arange(3)
    w  = 0.27
    ax.bar(xp - w,  pf_clean_sorted * 100, w, color='#2ecc71',
           edgecolor='black', linewidth=0.5, label='Bact clean')
    ax.bar(xp,      pf_dmg_sorted   * 100, w, color='#e67e22',
           edgecolor='black', linewidth=0.5, label='Bact damaged')
    ax.bar(xp + w,  pf_shuf_sorted  * 100, w, color='#7f8c8d',
           edgecolor='black', linewidth=0.5, label='Bact shuffled (random)')
    ax.axhline(RANDOM_EXPECTED * 100, color='red', ls='--', lw=1.2,
               label=f'Random expectation ({RANDOM_EXPECTED*100:.1f} %)')
    ax.set_xticks(xp)
    ax.set_xticklabels(['Best frame', 'Middle frame', 'Worst frame'])
    ax.set_ylabel('Mean stop-codon rate (%)')
    ax.set_title('Per-frame stop-codon rate (bact reads)\n'
                 '(best frame << random ⇒ reads are CDS)')
    ax.legend(fontsize=8, loc='upper left')
    for xi, v in zip(xp - w, pf_clean_sorted):
        ax.text(xi, v * 100 + 0.1, f'{v*100:.2f}',
                ha='center', va='bottom', fontsize=7)

    # Panel 2 — Survival curve (P(stop rate > x)) by source, log y-scale
    # Plain histograms are dominated by a single tall spike at 0, so use a
    # complementary CDF to see the whole distribution.
    def survival(x, grid):
        x = np.sort(x)
        # 1 - empirical CDF
        return 1.0 - np.searchsorted(x, grid, side='right') / len(x)

    ax = axes[1]
    grid = np.linspace(0, 0.20, 200)
    ax.plot(grid * 100, survival(bf_bact_clean,  grid), color='#2ecc71',
            lw=2.0, label=f'Bact clean (μ={bf_bact_clean.mean()*100:.2f} %)')
    ax.plot(grid * 100, survival(bf_human_clean, grid), color='#e74c3c',
            lw=2.0, label=f'Human chr21 clean (μ={bf_human_clean.mean()*100:.2f} %)')
    ax.plot(grid * 100, survival(bf_bact_shuf,   grid), color='#7f8c8d',
            lw=2.0, label=f'Random shuffled (μ={bf_bact_shuf.mean()*100:.2f} %)')
    ax.axvline(RANDOM_EXPECTED * 100, color='red', ls='--', lw=1.2,
               label=f'Random expectation ({RANDOM_EXPECTED*100:.1f} %)')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel('Best-frame stop-codon rate (%)')
    ax.set_ylabel('P(read has stop rate > x)  [log scale]')
    ax.set_title('Source comparison (survival curve)\n'
                 '(bact below human ⇒ metric tracks CDS density)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.25)

    # Panel 3 — Same view, bact only: clean vs damaged vs random
    ax = axes[2]
    ax.plot(grid * 100, survival(bf_bact_clean, grid), color='#2ecc71',
            lw=2.0, label=f'Bact clean (μ={bf_bact_clean.mean()*100:.2f} %)')
    ax.plot(grid * 100, survival(bf_bact_dmg,   grid), color='#e67e22',
            lw=2.0, label=f'Bact damaged (μ={bf_bact_dmg.mean()*100:.2f} %)')
    ax.plot(grid * 100, survival(bf_bact_shuf,  grid), color='#7f8c8d',
            lw=2.0, label=f'Bact shuffled (μ={bf_bact_shuf.mean()*100:.2f} %)')
    ax.axvline(RANDOM_EXPECTED * 100, color='red', ls='--', lw=1.2,
               label=f'Random expectation ({RANDOM_EXPECTED*100:.1f} %)')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel('Best-frame stop-codon rate (%)')
    ax.set_ylabel('P(read has stop rate > x)  [log scale]')
    ax.set_title('Damage degrades the coding signal\n'
                 '(damaged shifted right of clean by C→T stops)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.25)

    fig.suptitle('Coding-sequence signal in the test set',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'cds_evaluation.png', dpi=150)
    plt.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    lines = [
        'Coding-sequence (CDS) signal in the test set',
        '=' * 60,
        f'Test reads: {N:,}',
        '',
        f'Random-DNA expectation:        {RANDOM_EXPECTED*100:.3f} % stops per codon',
        '',
        'Best-frame stop-codon rate (mean across reads):',
        f'  Bact clean       {bf_bact_clean.mean()*100:>8.3f} %    '
        f'(n={len(bact_idx):,})',
        f'  Bact damaged     {bf_bact_dmg.mean()*100:>8.3f} %    '
        f'(n={len(bact_idx):,})',
        f'  Bact shuffled    {bf_bact_shuf.mean()*100:>8.3f} %    '
        f'(random-DNA control on same reads)',
        f'  Human chr21      {bf_human_clean.mean()*100:>8.3f} %    '
        f'(n={len(human_idx):,})',
        '',
        'Bact clean — per-frame stop rate (sorted within each read):',
        f'  Best frame  : {pf_bact_clean_sorted[0]*100:>7.3f} %',
        f'  Middle frame: {pf_bact_clean_sorted[1]*100:>7.3f} %',
        f'  Worst frame : {pf_bact_clean_sorted[2]*100:>7.3f} %',
        '',
        'Interpretation:',
        '  • Bact clean << random ⇒ reads are enriched for CDS regions.',
        '  • Best-frame gap between clean and damaged ⇒ damage shows up as',
        '    extra stop codons exactly because C→T converts Q/E/W codons',
        '    (CAA, CAG, CGA) into TAA / TAG / TGA.',
        '  • Bact clean < human chr21 ⇒ the metric tracks CDS density, not',
        '    just GC content (human chr21 is non-coding-dominated).',
    ]
    (OUT_DIR / 'results' / 'cds_summary.txt').write_text('\n'.join(lines))
    log(f'\nFigure → outputs/figures/cds_evaluation.png')
    log(f'Summary → outputs/results/cds_summary.txt')


if __name__ == '__main__':
    main()
