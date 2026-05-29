#!/bin/bash
# run.sh — submit the full Ancient DNA pipeline to SLURM.
#
# Trains and evaluates 7 model checkpoints (single seed 42):
#   Denoisers   : seq_only | evo2 | bwa | udg
#   Classifiers : Seq-only | Evo2 (per-base) | Evo2 (per-base + read LL)
#
# Numbering convention
#   N.x  — stage N, parallel sub-job x (runs on its own SLURM allocation)
#   N    — single sequential stage
#
# One-time setup before the first run:
#   bash Code/0_setup.sh                 # build Gargammel
#   bash Code/1_download_genomes.sh 250  # endogenous bacterial genomes
#   bash Code/1.1_download_contam.sh     # human chr21 + env bacteria
#
# Full pipeline:
#   bash run.sh
#
# Skip already-completed stages:
#   bash run.sh --skip-cluster         # reuse split_assignment.tsv
#   bash run.sh --skip-sim             # reuse existing data/
#   bash run.sh --skip-udg-sim         # reuse data/*_udg.npz
#   bash run.sh --skip-evo2            # ref_evo2 already in NPZ files
#   bash run.sh --skip-bwa             # ref_bwa already in NPZ files
#   bash run.sh --skip-pydamage        # skip pyDamage benchmark
#   bash run.sh --skip-read-baselines  # skip PMDtools / ngsBriggs

set -e

SKIP_SIM=false
SKIP_UDG_SIM=false
SKIP_EVO2=false
SKIP_BWA=false
SKIP_CLUSTER=false
SKIP_PYDAMAGE=false
SKIP_READ_BASELINES=false
for arg in "$@"; do
    [[ "$arg" == "--skip-sim"             ]] && SKIP_SIM=true
    [[ "$arg" == "--skip-udg-sim"         ]] && SKIP_UDG_SIM=true
    [[ "$arg" == "--skip-evo2"            ]] && SKIP_EVO2=true
    [[ "$arg" == "--skip-bwa"             ]] && SKIP_BWA=true
    [[ "$arg" == "--skip-cluster"         ]] && SKIP_CLUSTER=true
    [[ "$arg" == "--skip-pydamage"        ]] && SKIP_PYDAMAGE=true
    [[ "$arg" == "--skip-read-baselines"  ]] && SKIP_READ_BASELINES=true
done

EXCLUDE="hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl"

WRAP_HEADER='module load miniconda/24.5.0 gcc/11.2.0
eval "$(conda shell.bash hook)"
conda activate ancient-dna
export PYTHONNOUSERSITE=1
unset CUDA_VISIBLE_DEVICES_OVERRIDE
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then nvidia-smi -L 2>/dev/null | head -2 || true; fi'

mkdir -p outputs/{models,results,figures} logs

echo "================================================"
echo "  Ancient DNA Pipeline"
echo "  4 denoiser variants + 3 classifier variants (seed 42)"
echo "================================================"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0: Homology-aware genome clustering (CPU, minutes).
# Produces data/genomes/split_assignment.tsv.
# ─────────────────────────────────────────────────────────────────────────────
DEP_CLUSTER=""
if [ "$SKIP_CLUSTER" = false ]; then
    JOB_CLUSTER=$(sbatch --parsable \
        --job-name=adna-cluster \
        --output=logs/cluster_%j.log --error=logs/cluster_%j.err \
        --time=2-00:00:00 --mem=8G --cpus-per-task=4 \
        --wrap="$WRAP_HEADER
python Code/2_cluster_genomes.py --threshold 0.05")
    echo "Submitted: Cluster genomes (Mash)    → job $JOB_CLUSTER"
    DEP_CLUSTER="--dependency=afterok:$JOB_CLUSTER"
else
    echo "Skipping : Genome clustering (--skip-cluster)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1.1: Simulate the main dataset (regular Briggs damage), CPU.
# STAGE 1.2: Simulate the UDG-treated dataset in parallel (terminal-only damage).
# ─────────────────────────────────────────────────────────────────────────────
DEP_SIM=""
if [ "$SKIP_SIM" = false ]; then
    JOB_SIM=$(sbatch --parsable \
        --job-name=adna-sim \
        --output=logs/sim_%j.log --error=logs/sim_%j.err \
        --time=2-00:00:00 --mem=16G --cpus-per-task=4 \
        $DEP_CLUSTER \
        --wrap="$WRAP_HEADER
bash Code/3_simulate.sh && bash Code/3.1_simulate_contam.sh && python Code/4_build_dataset.py")
    echo "Submitted: 1.1 Simulate + dataset     → job $JOB_SIM"
    DEP_SIM="--dependency=afterok:$JOB_SIM"
else
    echo "Skipping : 1.1 Simulate (--skip-sim)"
fi

DEP_UDG_SIM=""
if [ "$SKIP_UDG_SIM" = false ]; then
    JOB_UDG_SIM=$(sbatch --parsable \
        --job-name=adna-sim-udg \
        --output=logs/sim_udg_%j.log --error=logs/sim_udg_%j.err \
        --time=2-00:00:00 --mem=16G --cpus-per-task=4 \
        $DEP_CLUSTER \
        --wrap="$WRAP_HEADER
UDG=true OUT_DIR=data/raw_udg bash Code/3_simulate.sh
OUT_DIR=data/raw_udg bash Code/3.1_simulate_contam.sh
RAW_DIR=data/raw_udg OUT_DIR=data OUT_SUFFIX=_udg python Code/4_build_dataset.py")
    echo "Submitted: 1.2 Simulate UDG + dataset → job $JOB_UDG_SIM"
    DEP_UDG_SIM="--dependency=afterok:$JOB_UDG_SIM"
else
    echo "Skipping : 1.2 UDG simulation (--skip-udg-sim)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2.1: Evo2 soft references (GPU) — parallel with BWA below.
# STAGE 2.2: BWA hard references (CPU).
# Both run on the main (non-UDG) dataset.
# ─────────────────────────────────────────────────────────────────────────────
DEP_EVO2=""
if [ "$SKIP_EVO2" = false ]; then
    JOB_EVO2=$(sbatch --parsable \
        --job-name=adna-evo2 \
        --output=logs/evo2_%j.log --error=logs/evo2_%j.err \
        --time=2-00:00:00 --gres=gpu:a100:1 --mem=64G --cpus-per-task=4 \
        --exclude=$EXCLUDE \
        $DEP_SIM \
        --wrap="$WRAP_HEADER
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python Code/5.1_evo2_refs.py --batch-size 64 --dtype bf16")
    echo "Submitted: 2.1 Evo2 soft references   → job $JOB_EVO2"
    DEP_EVO2="--dependency=afterok:$JOB_EVO2"
else
    echo "Skipping : 2.1 Evo2 (--skip-evo2)"
fi

DEP_BWA=""
if [ "$SKIP_BWA" = false ]; then
    JOB_BWA=$(sbatch --parsable \
        --job-name=adna-bwa \
        --output=logs/bwa_%j.log --error=logs/bwa_%j.err \
        --time=2-00:00:00 --mem=32G --cpus-per-task=8 \
        $DEP_SIM \
        --wrap="$WRAP_HEADER
module load bwa/0.7.17 samtools/1.17 2>/dev/null || true
python Code/5.2_bwa_refs.py")
    echo "Submitted: 2.2 BWA hard references    → job $JOB_BWA"
    DEP_BWA="--dependency=afterok:$JOB_BWA"
else
    echo "Skipping : 2.2 BWA (--skip-bwa)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3.1–3.4: Train denoisers — 4 variants in parallel, one GPU job each.
#   seq_only depends on main sim, evo2 depends on Evo2 refs, bwa depends on BWA
#   refs, udg depends on UDG sim.
# ─────────────────────────────────────────────────────────────────────────────
SEEDS="42"

submit_denoiser () {
    local tag="$1" variant="$2" dep="$3"
    sbatch --parsable \
        --job-name=adna-train-$tag \
        --output=logs/train_${tag}_%j.log --error=logs/train_${tag}_%j.err \
        --time=2-00:00:00 --gres=gpu:1 --mem=32G --cpus-per-task=4 \
        --exclude=$EXCLUDE \
        $dep \
        --wrap="$WRAP_HEADER
python Code/6_train_denoiser.py --variant $variant --seeds $SEEDS"
}

JOB_TRAIN_SEQ=$(submit_denoiser seq seq_only "$DEP_SIM")
echo "Submitted: 3.1 Train denoiser seq_only → job $JOB_TRAIN_SEQ"
JOB_TRAIN_EVO=$(submit_denoiser evo evo2    "$DEP_EVO2")
echo "Submitted: 3.2 Train denoiser evo2    → job $JOB_TRAIN_EVO"
JOB_TRAIN_BWA=$(submit_denoiser bwa bwa     "$DEP_BWA")
echo "Submitted: 3.3 Train denoiser bwa     → job $JOB_TRAIN_BWA"
JOB_TRAIN_UDG=$(submit_denoiser udg udg     "$DEP_UDG_SIM")
echo "Submitted: 3.4 Train denoiser udg     → job $JOB_TRAIN_UDG"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4.1–4.4: Denoise the test set — all 4 variants in parallel (GPU).
# ─────────────────────────────────────────────────────────────────────────────
submit_denoise () {
    local tag="$1" variant="$2" train_job="$3"
    sbatch --parsable \
        --job-name=adna-denoise-$tag \
        --output=logs/denoise_${tag}_%j.log --error=logs/denoise_${tag}_%j.err \
        --time=2-00:00:00 --gres=gpu:1 --mem=16G --cpus-per-task=2 \
        --exclude=$EXCLUDE \
        --dependency=afterok:$train_job \
        --wrap="$WRAP_HEADER
python Code/7_denoise.py --variant $variant --seeds $SEEDS"
}

JOB_DEN_SEQ=$(submit_denoise seq seq_only $JOB_TRAIN_SEQ)
echo "Submitted: 4.1 Denoise seq_only      → job $JOB_DEN_SEQ"
JOB_DEN_EVO=$(submit_denoise evo evo2    $JOB_TRAIN_EVO)
echo "Submitted: 4.2 Denoise evo2          → job $JOB_DEN_EVO"
JOB_DEN_BWA=$(submit_denoise bwa bwa     $JOB_TRAIN_BWA)
echo "Submitted: 4.3 Denoise bwa           → job $JOB_DEN_BWA"
JOB_DEN_UDG=$(submit_denoise udg udg     $JOB_TRAIN_UDG)
echo "Submitted: 4.4 Denoise udg           → job $JOB_DEN_UDG"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: Denoiser evaluation (CPU) — runs after all four denoise jobs finish.
# Combines per-variant probs, produces error-source breakdowns and threshold
# sweep, plus the CDS-aware evaluation and Bayesian ceiling.
# ─────────────────────────────────────────────────────────────────────────────
JOB_EVAL=$(sbatch --parsable \
    --job-name=adna-eval \
    --output=logs/eval_%j.log --error=logs/eval_%j.err \
    --time=2-00:00:00 --mem=8G --cpus-per-task=4 \
    --dependency=afterok:$JOB_DEN_SEQ:$JOB_DEN_EVO:$JOB_DEN_BWA:$JOB_DEN_UDG \
    --wrap="$WRAP_HEADER
python Code/8_evaluate_denoiser.py
python Code/8.1_cds_eval.py
python Code/8.2_bayesian_ceiling.py
python Code/8.3_denoiser_errors.py --variant seq_only
python Code/8.3_denoiser_errors.py --variant evo2
python Code/8.3_denoiser_errors.py --variant bwa
python Code/8.3_denoiser_errors.py --variant udg
python Code/8.4_denoiser_threshold_sweep.py")
echo "Submitted: 5   Denoiser evaluation   → job $JOB_EVAL"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6.1–6.3: Classifier variants (GPU) — three in parallel.
# STAGE 6.4: merge step on CPU, produces classifier_probs.npz and plots.
# ─────────────────────────────────────────────────────────────────────────────
submit_classifier () {
    local tag="$1" variant="$2"
    sbatch --parsable \
        --job-name=adna-clf-$tag \
        --output=logs/classify_${tag}_%j.log --error=logs/classify_${tag}_%j.err \
        --time=2-00:00:00 --gres=gpu:1 --mem=16G --cpus-per-task=2 \
        --exclude=$EXCLUDE \
        $DEP_EVO2 \
        --wrap="$WRAP_HEADER
python Code/9_classifier.py --variant $variant"
}

JOB_CLF_SEQ=$(submit_classifier seq      seq)
echo "Submitted: 6.1 Classifier seq        → job $JOB_CLF_SEQ"
JOB_CLF_EVO_BASE=$(submit_classifier evobase evo_base)
echo "Submitted: 6.2 Classifier evo_base   → job $JOB_CLF_EVO_BASE"
JOB_CLF_EVO_FULL=$(submit_classifier evofull evo_full)
echo "Submitted: 6.3 Classifier evo_full   → job $JOB_CLF_EVO_FULL"

JOB_CLF=$(sbatch --parsable \
    --job-name=adna-clf-merge \
    --output=logs/classify_merge_%j.log --error=logs/classify_merge_%j.err \
    --time=2-00:00:00 --mem=8G --cpus-per-task=2 \
    --dependency=afterok:$JOB_CLF_SEQ:$JOB_CLF_EVO_BASE:$JOB_CLF_EVO_FULL \
    --wrap="$WRAP_HEADER
python Code/9_classifier.py --merge")
echo "Submitted: 6.4 Classifier merge      → job $JOB_CLF"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7.1: pyDamage benchmark — oracle reference (CPU).
# STAGE 7.2: pyDamage benchmark — de novo assembly with MetaSPAdes (CPU).
# STAGE 7.3: PMDtools + ngsBriggs (CPU) — runs in parallel with 7.1/7.2.
# ─────────────────────────────────────────────────────────────────────────────
DEP_PYD=""
PYD_FLAG=""
if [ "$SKIP_PYDAMAGE" = false ]; then
    JOB_PYD_ORACLE=$(sbatch --parsable \
        --job-name=adna-pyd-oracle \
        --output=logs/pyd_oracle_%j.log --error=logs/pyd_oracle_%j.err \
        --time=2-00:00:00 --mem=16G --cpus-per-task=4 \
        $DEP_SIM \
        --wrap="$WRAP_HEADER
module load bwa/0.7.17 samtools/1.20 2>/dev/null || true
MODE=oracle bash Code/10.1_run_pydamage.sh")
    echo "Submitted: 7.1 pyDamage (oracle)     → job $JOB_PYD_ORACLE"

    JOB_PYD_DENOVO=$(sbatch --parsable \
        --job-name=adna-pyd-denovo \
        --output=logs/pyd_denovo_%j.log --error=logs/pyd_denovo_%j.err \
        --time=2-00:00:00 --mem=32G --cpus-per-task=8 \
        $DEP_SIM \
        --wrap="$WRAP_HEADER
module load bwa/0.7.17 samtools/1.20 2>/dev/null || true
MODE=denovo THREADS=8 bash Code/10.1_run_pydamage.sh")
    echo "Submitted: 7.2 pyDamage (de novo)    → job $JOB_PYD_DENOVO"

    DEP_PYD="afterok:$JOB_PYD_ORACLE:$JOB_PYD_DENOVO"
    PYD_FLAG="--pydamage data/pydamage/oracle/results/pydamage_results.csv \
              --pydamage-denovo data/pydamage/denovo/results/pydamage_results.csv"
else
    echo "Skipping : 7.1/7.2 pyDamage (--skip-pydamage)"
fi

DEP_RB=""
RB_FLAG=""
if [ "$SKIP_READ_BASELINES" = false ]; then
    JOB_RB=$(sbatch --parsable \
        --job-name=adna-readbase \
        --output=logs/readbase_%j.log --error=logs/readbase_%j.err \
        --time=2-00:00:00 --mem=16G --cpus-per-task=4 \
        $DEP_SIM \
        --wrap="$WRAP_HEADER
module load bwa/0.7.17 samtools/1.20 2>/dev/null || true
bash Code/10.2_run_read_baselines.sh")
    echo "Submitted: 7.3 PMDtools + ngsBriggs  → job $JOB_RB"
    DEP_RB="afterok:$JOB_RB"
    RB_FLAG="--pmdtools data/read_baselines/pmdtools_scores.csv \
             --ngsbriggs data/read_baselines/ngsbriggs_scores.csv"
else
    echo "Skipping : 7.3 Read baselines (--skip-read-baselines)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8: Baseline comparison + classifier plots (CPU).
# Depends on the classifier merge job plus whatever 7.x benchmarks ran.
# ─────────────────────────────────────────────────────────────────────────────
DEP_FINAL="afterok:$JOB_CLF"
[ -n "$DEP_PYD" ] && DEP_FINAL="$DEP_FINAL,$DEP_PYD"
[ -n "$DEP_RB"  ] && DEP_FINAL="$DEP_FINAL,$DEP_RB"

JOB_BASE=$(sbatch --parsable \
    --job-name=adna-baselines \
    --output=logs/baselines_%j.log --error=logs/baselines_%j.err \
    --time=2-00:00:00 --mem=8G --cpus-per-task=4 \
    --dependency=$DEP_FINAL \
    --wrap="$WRAP_HEADER
python Code/11_baselines_compare.py $PYD_FLAG $RB_FLAG
python Code/11.1_plot_classifier.py
python Code/12_per_source_auc.py")
echo "Submitted: 8   Baselines + per-source → job $JOB_BASE"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 9.1: Per-cluster classifier AUC (needs BAM from 7.1).
# STAGE 9.2: Classifier threshold sweep.
# STAGE 9.3: Damage-profile verification.
# STAGE 9.4: Sequencing-error robustness (GPU — re-evaluates classifier).
# ─────────────────────────────────────────────────────────────────────────────
DEP_CLUSTER_AUC="afterok:$JOB_CLF"
[ -n "$DEP_PYD" ] && DEP_CLUSTER_AUC="$DEP_CLUSTER_AUC,afterok:$JOB_PYD_ORACLE"
JOB_PCAUC=$(sbatch --parsable \
    --job-name=adna-pcauc \
    --output=logs/pcauc_%j.log --error=logs/pcauc_%j.err \
    --time=2-00:00:00 --mem=8G --cpus-per-task=2 \
    --dependency=$DEP_CLUSTER_AUC \
    --wrap="$WRAP_HEADER
python Code/14_per_genus_auc.py")
echo "Submitted: 9.1 Per-cluster AUC       → job $JOB_PCAUC"

JOB_THRSWEEP=$(sbatch --parsable \
    --job-name=adna-thrsweep \
    --output=logs/thrsweep_%j.log --error=logs/thrsweep_%j.err \
    --time=2-00:00:00 --mem=8G --cpus-per-task=2 \
    --dependency=afterok:$JOB_CLF \
    --wrap="$WRAP_HEADER
python Code/15_classifier_threshold_sweep.py")
echo "Submitted: 9.2 Classifier threshold  → job $JOB_THRSWEEP"

JOB_DMGSTATS=$(sbatch --parsable \
    --job-name=adna-dmgstats \
    --output=logs/dmgstats_%j.log --error=logs/dmgstats_%j.err \
    --time=2-00:00:00 --mem=8G --cpus-per-task=2 \
    $DEP_SIM \
    --wrap="$WRAP_HEADER
python Code/16_damage_stats.py")
echo "Submitted: 9.3 Damage stats          → job $JOB_DMGSTATS"

JOB_SEQERR=$(sbatch --parsable \
    --job-name=adna-seqerr \
    --output=logs/seqerr_%j.log --error=logs/seqerr_%j.err \
    --time=2-00:00:00 --gres=gpu:1 --mem=16G --cpus-per-task=2 \
    --exclude=$EXCLUDE \
    --dependency=afterok:$JOB_CLF \
    --wrap="$WRAP_HEADER
python Code/17_seqerror_robustness.py")
echo "Submitted: 9.4 Seq-error robustness  → job $JOB_SEQERR"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 10: UDG cross-protocol evaluation — re-evaluates the trained
# regular-protocol classifier/denoiser checkpoints on the UDG test set.
# Depends on classifier merge (for checkpoints) and UDG simulation (for data).
# ─────────────────────────────────────────────────────────────────────────────
DEP_UDG="afterok:$JOB_CLF"
[ -n "$DEP_UDG_SIM" ] && DEP_UDG="$DEP_UDG,afterok:$JOB_UDG_SIM"
JOB_UDG=$(sbatch --parsable \
    --job-name=adna-udg-eval \
    --output=logs/udg_%j.log --error=logs/udg_%j.err \
    --time=2-00:00:00 --mem=16G --cpus-per-task=4 \
    --dependency=$DEP_UDG \
    --wrap="$WRAP_HEADER
python Code/13.1_evaluate_udg.py")
echo "Submitted: 10  UDG cross-protocol    → job $JOB_UDG"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "Monitor:  squeue -u \$USER"
echo ""
echo "Key output files:"
echo "  Denoiser comparison       → outputs/figures/evaluation.png"
echo "  Denoiser threshold sweep  → outputs/figures/denoiser_threshold_sweep.png"
echo "  Classifier ROC/PR         → outputs/figures/classifier.png"
echo "  Classifier threshold      → outputs/figures/classifier_threshold_sweep.png"
echo "  All models vs Briggs      → outputs/figures/baselines_comparison.png"
echo "  Per-source AUC            → outputs/figures/per_source_auc.png"
echo "  Per-cluster AUC           → outputs/figures/per_cluster_auc.png"
echo "  Damage stats              → outputs/figures/damage_stats.png"
echo "  Seq-error robustness      → outputs/figures/seqerror_robustness.png"
echo "  UDG cross-protocol        → outputs/figures/udg_evaluation.png"
echo "================================================"
