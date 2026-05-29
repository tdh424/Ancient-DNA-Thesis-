# Ancient DNA — Reference-free Single-Read Damage Detection with Evo 2

Code for the MSc thesis *Reference-free Single-Read Ancient DNA Damage Detection with Genomic
Language Models*.

We simulate Briggs-model post-mortem damage (C→T at the 5' end, G→A at the 3' end) on
bacterial reference genomes and train Transformer models to detect and correct that damage
from a **single read**, without aligning to a reference at inference time. The frozen Evo 2
genomic language model is used as a soft sequence prior.

Two tasks, evaluated as several input variants:

| Task | Variant | External signal | Notes |
|------|---------|-----------------|-------|
| Denoiser (per base)   | `seq_only` | none           | sequence + position only |
| Denoiser (per base)   | `evo2`     | Evo 2 (7B)     | per-position probabilities + Shannon entropy channel |
| Denoiser (per base)   | `bwa`      | BWA alignment  | one-hot reference base (oracle alignment) |
| Denoiser (per base)   | `udg`      | none           | trained on UDG-treated data |
| Classifier (per read) | `seq`      | none           | 2 damage-count features |
| Classifier (per read) | `evo_base` | Evo 2 (7B)     | + local Evo 2 damage-fraction features |
| Classifier (per read) | `evo_full` | Evo 2 (7B)     | + whole-read Evo 2 log-likelihood |

External baselines: **Briggs LLR** (reference-free positional rule), **PMDtools** (reference-based,
oracle alignment) and **pyDamage** (contig level, oracle and de novo MetaSPAdes assembly).

All models are trained with a single random seed (42). The code supports seed ensembling via
`--seeds`, but the thesis results use one seed.

---

## Running from scratch

The pipeline was developed for the Hendrix SLURM cluster (University of Copenhagen). `run.sh`
submits every stage as SLURM jobs with the right dependencies. Each script can also be run
directly without SLURM (see *Pipeline* below).

### 1. Prerequisites

A CUDA GPU with compute capability `sm_80`+ (A100, L40S, A40, H100) is required for Evo 2.
The following command-line tools must be on `PATH` (cluster modules or conda):

- `datasets` (NCBI datasets CLI) — genome download
- `mash` — genome clustering
- `bwa`, `samtools` — alignment and BAM handling
- `metaspades.py` (SPAdes) — de novo assembly for the pyDamage baseline
- `pmdtools`, `pydamage` — external baselines

### 2. One-time setup

```bash
conda env create -f environment.yml      # Python 3.12, PyTorch (CUDA 12.1), Evo 2, scikit-learn, pysam
conda activate ancient-dna

bash Code/0_setup.sh                      # build Gargammel from Code/Gargammel/
bash Code/1_download_genomes.sh 300       # query N bacterial accessions, filter to <20 MB
bash Code/1.1_download_contam.sh          # human chr21 + environmental bacteria
```

### 3. Run the full pipeline

```bash
bash run.sh
```

This downloads nothing further. It clusters and splits the genomes, simulates the damaged and
UDG datasets, precomputes the Evo 2 and BWA references, trains the four denoiser and three
classifier variants, runs the external baselines, and produces every figure and table under
`outputs/`. Reuse already-completed stages with the skip flags:

```bash
bash run.sh --skip-cluster --skip-sim --skip-udg-sim   # reuse data/*.npz
bash run.sh --skip-evo2 --skip-bwa                      # reuse precomputed references
bash run.sh --skip-pydamage --skip-read-baselines       # skip the external baselines
```

### 4. What you get

Figures land in `outputs/figures/` and the matching numeric tables in `outputs/results/`
(see *Outputs* below). The full run is roughly 50 GPU-hours on A100 hardware, of which the
Evo 2 reference precomputation is about 30 hours.

---

## Pipeline

File numbering is `N.x`, where `N` is the stage and `.x` is a parallel sub-job. Run a script
directly to reproduce just its outputs (no SLURM needed for the single-GPU and CPU steps).

```
0    Code/0_setup.sh                 build Gargammel
1    Code/1_download_genomes.sh      endogenous bacterial genomes (NCBI RefSeq)
1.1  Code/1.1_download_contam.sh     human chr21 + environmental bacteria
2    Code/2_cluster_genomes.py       Mash 95% ANI clusters -> 80/10/10 split

     Code/3_simulate.sh              Gargammel fragSim + deamSim (UDG=true for the UDG set)
     Code/3.1_simulate_contam.sh     spike in modern human + environmental reads
     Code/4_build_dataset.py         FASTA pairs -> NPZ (encoded, 30-100 bp length filter)

     Code/5.1_evo2_refs.py           (GPU) add ref_evo2 channel to NPZ
     Code/5.2_bwa_refs.py            (CPU) add ref_bwa channel to NPZ

     Code/6_train_denoiser.py --variant {seq_only,evo2,bwa,udg} --seeds 42
     Code/7_denoise.py        --variant {seq_only,evo2,bwa,udg} --seeds 42

     Code/8_evaluate_denoiser.py          per-variant figure + AUC tables
     Code/8.1_cds_eval.py                 CDS / stop-codon proxy
     Code/8.2_bayesian_ceiling.py         position-only Bayesian baseline
     Code/8.3_denoiser_errors.py          per-variant error breakdown
     Code/8.4_denoiser_threshold_sweep.py threshold sensitivity
     Code/8.4_cds_per_threshold.py        CDS proxy across thresholds

     Code/9_classifier.py --variant {seq,evo_base,evo_full}    (then --merge)

7.1  Code/10.1_run_pydamage.sh  MODE=oracle   (source-genome contigs)
7.2  Code/10.1_run_pydamage.sh  MODE=denovo   (MetaSPAdes assembly)
7.3  Code/10.2_run_read_baselines.sh          PMDtools + Briggs LLR
     Code/11.2_pmdtools_eval.py               PMDtools per-read scoring on the oracle BAM
     Code/11.3_contig_level_eval.py           contig-level pyDamage vs evo_full

8    Code/11_baselines_compare.py    classifiers + Briggs LLR + pyDamage on one figure
     Code/11.1_plot_classifier.py    final classifier comparison
     Code/12_per_source_auc.py       bacterial / human / env breakdown
     Code/14_per_genus_auc.py        per-Mash-cluster AUC
     Code/15_classifier_threshold_sweep.py  F1/MCC threshold sweeps

     Code/16_damage_stats.py              empirical vs theoretical Briggs profile
     Code/16.1_simulation_verification.py simulated test-set verification figure
     Code/16.2_taxonomic_skewness.py      genus-distribution figure
     Code/17_seqerror_robustness.py       AUC under added sequencing error
     Code/18_extra_figures.py             extra analysis figures A-G

10   Code/13_run_udg_test.sh         build the UDG-treated test set
     Code/13.1_evaluate_udg.py       UDG cross-protocol evaluation
```

---

## Model architecture

### Denoiser (`Code/6_train_denoiser.py`)

Per-position Transformer that predicts the original base at each position of a damaged read.

| Component | Details |
|-----------|---------|
| Embedding | 128-d token embedding (vocab size 6) |
| Positional encoding | dual sinusoidal: forward (5'→3') + reverse (distance from 3' end) |
| Reference projection | `Linear(7 -> 128)` for Evo 2 / BWA + entropy (omitted for `seq_only` and `udg`) |
| Encoder | 4-layer Transformer, 8 heads, 256-d FFN, pre-LN |
| Loss | focal cross-entropy (gamma=2) with 200x upweight inside the Briggs window (positions 0-7, both ends) |
| Checkpoint metric | validation PR-AUC (averaged over C→T and G→A) |
| LR schedule | linear warmup (5 epochs) + cosine annealing |

Reference signals:
- `seq_only` — no external signal (the reference-free floor)
- `evo2` — Evo 2 (7B) per-position probabilities `P(A/C/G/T/N/PAD)` + a Shannon-entropy channel
- `bwa` — one-hot reference base from BWA alignment to the source genome. Reads that fail to
  align (or fall below MAPQ 20) get a zero vector, which is the realistic floor an
  alignment-based pipeline works with
- `udg` — same architecture as `seq_only`, trained on UDG-treated data (interior C→T removed,
  terminal damage retained)

### Classifier (`Code/9_classifier.py`)

Per-read binary classifier: ancient (damaged, label 1) vs modern (undamaged, label 0).

| Component | Details |
|-----------|---------|
| Embedding | 64-d token embedding |
| Positional encoding | dual sinusoidal |
| Encoder | 2-layer Transformer, 4 heads, 128-d FFN |
| Pooling | max + mean pooling concatenated, projected back to 64-d |
| Damage features | explicit per-read scalar features (below), projected and added before the head |
| Head | LayerNorm -> Linear -> sigmoid |
| Threshold | best-MCC operating point selected on the validation split |

Damage feature vectors:
- `seq` (2): `[n_ct_5, n_ga_3]`
- `evo_base` (4): `+ [dmg_frac_5, dmg_frac_3]`
- `evo_full` (5): `+ [evo2_ll]`

---

## Data

- **Genomes**: NCBI RefSeq bacteria (endogenous), plus human chr21 and environmental bacteria
  as modern contamination
- **Split**: 80/10/10 train/val/test, stratified by Mash 95% ANI cluster so no homologous
  fragment leaks across splits
- **Damage simulation**: Gargammel `fragSim` + `deamSim` with Briggs (2007) Vi-33.16
  parameters (`nu=0.0241, lambda=0.3590, delta_d=0.00937, delta_s=0.6815`); the
  single-stranded rate `s` is resampled per genome from a `Beta(4,2)`. The ART sequencing
  stage is skipped and a flat 0.5% per-base error is applied manually, so the clean/damaged
  difference is pure Briggs damage plus controllable noise
- **Fragment lengths**: log-normal, mean ~53 bp; reads outside 30-100 bp are filtered out
- **Prevalence**: 25% ancient bacterial / 75% modern (human + env + undamaged bacterial)
- **Encoding**: `PAD=0 A=1 C=2 G=3 T=4 N=5`
- **UDG variant**: same simulator with `UDG=true` (`delta_d=0`), interior C→T removed

Large intermediate data (`data/`, `outputs/models/`, `logs/`, `archive/`) is git-ignored and
regenerated by the pipeline.

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/models/denoiser_{variant}_seed{N}.pt` | trained denoiser weights |
| `outputs/models/classifier_{variant}_seed{N}.pt` | trained classifier weights |
| `outputs/results/test_denoised_{variant}.npz` | denoised test reads |
| `outputs/results/classifier_probs.npz` | per-read P(ancient) for all classifier variants |
| `outputs/figures/classifier_comparison.png` | classifiers + Briggs LLR + PMDtools |
| `outputs/figures/per_source_auc.png` | per-source (bact / human / env) AUC |
| `outputs/figures/per_cluster_auc.png` | per-Mash-cluster AUC distribution |
| `outputs/figures/damage_stats.png` | empirical vs theoretical Briggs profile |
| `outputs/figures/seqerror_robustness.png` | AUC under added sequencing error |
| `outputs/figures/udg_evaluation.png` | UDG cross-protocol evaluation |
| `outputs/figures/extra_*.png` | additional analysis figures (A-G) |
| `outputs/results/*.txt` | numeric tables behind every figure |

---

## Environment

```bash
conda env create -f environment.yml
conda activate ancient-dna
```

Evo 2 (7B) is loaded from the `evo2` package (`arcinstitute/evo2-7b`, Hugging Face). BWA,
samtools, SPAdes, Mash, PMDtools, pyDamage and the NCBI datasets CLI are expected on `PATH`.

---

## Re-running individual stages

```bash
# Reproduce a single figure (no retraining), once data and models exist
python Code/11_baselines_compare.py
python Code/15_classifier_threshold_sweep.py

# Force re-computation of the Evo 2 / BWA reference channels
python -c "
import numpy as np
for s in ['train','val','test']:
    d = dict(np.load(f'data/{s}.npz'))
    d.pop('ref_bwa', None); d.pop('ref_evo2', None)
    np.savez_compressed(f'data/{s}.npz', **d)
"
```

### SLURM cheatsheet

```bash
squeue -u $USER                     # queued / running jobs
sacct -u $USER -X --format=JobID,JobName,State,Elapsed   # recent history
tail -f logs/<job>_<jobid>.log      # live-tail a job log
srun --gres=gpu:1 --mem=32G --time=04:00:00 --pty bash   # interactive GPU node
```
