"""
10_classifier.py — ML-based aDNA read classifier.

Trains three Transformer classifiers to distinguish ancient (damaged) reads
from modern (undamaged) reads, without needing a reference genome.

Model variants trained in one run:
  A) Seq-only               — sequence and position only, no external signal
  B) Evo2 (per-base)        — + Evo2 P(C)/P(G) at damage-prone positions (4 features)
  C) Evo2 (per-base + LL)   — + per-read log-likelihood from Evo2 (5 features)

Architecture:
  - Token embedding + dual sinusoidal PE (forward + reverse distance from ends)
  - 2-layer Transformer encoder + max+mean pooling
  - Explicit damage features projected and added to pooled representation
  - Focal loss (γ=2) + PR-AUC checkpoint + LR warmup + cosine annealing

Labels:
  1 = ancient  (read has at least one C→T or G→A damage substitution)
  0 = modern   (read identical to reference — no damage)

Outputs → outputs/results/
    classifier_probs.npz     keys: labels, probs_seq, probs_evo_base, probs_evo_full

Outputs → outputs/figures/
    classifier.png
    classifier_threshold_sweep.png

Outputs → outputs/results/
    classifier_summary.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             precision_recall_curve, confusion_matrix)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path('data')
OUT_DIR       = Path('outputs')
MAX_LEN       = 100
VOCAB_SIZE    = 6   # PAD=0 A=1 C=2 G=3 T=4 N=5
D_MODEL       = 64
N_HEADS       = 4
N_LAYERS      = 2
DIM_FF        = 128
DROPOUT       = 0.1
BATCH_SIZE    = 2048
LR            = 1e-3
EPOCHS        = 60
PATIENCE      = 12
WARMUP_EPOCHS = 5
FOCAL_GAMMA   = 2.0
BRIGGS_WIN    = 8

# Performance and inference options
USE_BF16     = True       # autocast on CUDA
USE_RC_TTA   = True       # average forward + reverse-complement at test time
N_SEEDS      = 1          # ensemble size; set to 1 to disable
SEEDS        = [42, 43, 44][:N_SEEDS]
CHECKPOINT_METRIC = 'mcc' # 'pr_auc' or 'mcc' — Line prefers MCC
# ─────────────────────────────────────────────────────────────────────────────

# Model display names (used in plots and summary tables)
MODEL_NAMES = {
    'seq':      'Seq-only',
    'evo_base': 'Evo2 (per-base)',
    'evo_full': 'Evo2 (per-base + read LL)',
}

COLORS = {
    'Seq-only':                  '#3498db',   # blue
    'Evo2 (per-base)':           '#2ecc71',   # green
    'Evo2 (per-base + read LL)': '#1a7a49',   # dark green
}


def log(msg=''):
    print(msg, flush=True)


def focal_loss(logits, labels, gamma=FOCAL_GAMMA):
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    p_t = torch.sigmoid(logits) * labels + (1 - torch.sigmoid(logits)) * (1 - labels)
    return ((1 - p_t).pow(gamma) * bce).mean()


def _autocast_ctx(device):
    if USE_BF16 and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    from contextlib import nullcontext
    return nullcontext()


def _unwrap(model):
    return getattr(model, '_orig_mod', model)


# Reverse-complement helpers (encoding: PAD=0 A=1 C=2 G=3 T=4 N=5)
_RC_MAP = torch.tensor([0, 4, 3, 2, 1, 5], dtype=torch.long)
# Channel permutation for ref_evo2 distributions: [N, A, C, G, T, N] → after RC
# the A↔T and C↔G channels swap positions
_RC_CH  = torch.tensor([0, 4, 3, 2, 1, 5], dtype=torch.long)


def reverse_complement(reads, lengths, ref=None):
    """Vectorised reverse-complement on padded reads.
    reads: (B, L) long. lengths: (B,) long. ref: optional (B, L, 6) float.
    Returns rc_reads, rc_ref (or None) — both same shape as inputs.
    """
    B, L   = reads.shape
    pos    = torch.arange(L, device=reads.device).unsqueeze(0)         # (1, L)
    valid  = pos < lengths.unsqueeze(1)                                # (B, L)
    # Index that flips read[0..len-1] → read[len-1..0], padding stays 0
    flip   = (lengths.unsqueeze(1) - 1 - pos).clamp(min=0)
    rc_map = _RC_MAP.to(reads.device)
    rc_int = rc_map[reads]                                             # complement
    rc_reads = torch.gather(rc_int, 1, flip)                           # then reverse
    rc_reads = torch.where(valid, rc_reads, torch.zeros_like(rc_reads))

    rc_ref = None
    if ref is not None:
        rc_ch    = _RC_CH.to(ref.device)
        ref_perm = ref.index_select(-1, rc_ch)                          # channel swap
        idx      = flip.unsqueeze(-1).expand(-1, -1, ref.shape[-1])
        rc_ref   = torch.gather(ref_perm, 1, idx)                       # reverse
        rc_ref   = rc_ref * valid.unsqueeze(-1).float()
    return rc_reads, rc_ref


def mcc_at_best_threshold(probs, labels, n_steps=51):
    """Quick MCC sweep for use during training (cheap)."""
    best = -1.0
    for t in np.linspace(0.05, 0.95, n_steps):
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
        d = np.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
        m = (float(tp)*float(tn) - float(fp)*float(fn)) / d if d > 0 else 0.0
        if m > best: best = m
    return float(best)


def get_lr_scale(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1 + np.cos(np.pi * progress))


# ── Data ──────────────────────────────────────────────────────────────────────

def make_labels(damaged, clean):
    """1 if any position differs (ancient read with damage), 0 if identical (modern)."""
    return ((damaged != clean) & (damaged != 0)).any(axis=1).astype(np.int32)


def load_split(name, with_evo2=True):
    d       = np.load(DATA_DIR / f'{name}.npz')
    damaged = d['damaged'].astype(np.int64)
    clean   = d['clean'].astype(np.int64)
    lengths = d['lengths'].astype(np.int64)
    labels  = make_labels(damaged, clean)
    ref     = d['ref_evo2'].astype(np.float32) if (with_evo2 and 'ref_evo2' in d) else None
    return damaged, clean, lengths, labels, ref


class ReadDataset(torch.utils.data.Dataset):
    def __init__(self, reads, lengths, labels, ref=None):
        self.x = torch.from_numpy(reads)
        self.l = torch.from_numpy(lengths)
        self.y = torch.from_numpy(labels)
        self.r = torch.from_numpy(ref) if ref is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        r = self.r[i] if self.r is not None else torch.zeros(MAX_LEN, VOCAB_SIZE)
        return self.x[i], self.l[i], self.y[i], r


# ── Model ─────────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class aDNAClassifier(nn.Module):
    """
    Binary classifier: ancient (1) vs modern (0).

    n_dmg_features controls the explicit damage feature vector:
      2 → [n_ct_5, n_ga_3]                    (seq-only variant)
      4 → [dmg_frac_5, n_ct_5, dmg_frac_3, n_ga_3]    (evo2 per-base)
      5 → same + evo2 per-read log-likelihood  (evo2 full)
    """
    def __init__(self, use_evo2=True, n_dmg_features=5):
        super().__init__()
        self.use_evo2        = use_evo2
        self.n_dmg_features  = n_dmg_features
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL, padding_idx=0)
        self.pos_enc   = SinusoidalPE(D_MODEL)
        self.rev_pos   = SinusoidalPE(D_MODEL)
        if use_evo2:
            self.ref_proj = nn.Linear(VOCAB_SIZE + 1, D_MODEL, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=DIM_FF,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder   = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.pool_proj = nn.Linear(2 * D_MODEL, D_MODEL)
        self.dmg_proj  = nn.Linear(n_dmg_features, D_MODEL)
        self.head      = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, 1))

    def forward(self, x, lengths, ref=None):
        B, L = x.shape
        emb  = self.embedding(x)
        if self.use_evo2 and ref is not None:
            entropy = -(ref * torch.log(ref.clamp(min=1e-8))).sum(-1, keepdim=True)
            emb = emb + self.ref_proj(torch.cat([ref, entropy], dim=-1))
        emb = self.pos_enc(emb)

        pos      = torch.arange(L, device=x.device).unsqueeze(0)        # (1, L)
        dist_3p  = (lengths.unsqueeze(1) - 1 - pos).clamp(0, L - 1)    # (B, L)
        emb      = emb + self.rev_pos.pe[0][dist_3p]
        pad_mask = pos >= lengths.unsqueeze(1)                           # (B, L)
        enc      = self.encoder(emb, src_key_padding_mask=pad_mask)

        # Max + mean pooling over valid positions
        valid_f   = (~pad_mask).float().unsqueeze(-1)                    # (B, L, 1)
        mean_pool = (enc * valid_f).sum(1) / valid_f.sum(1).clamp(min=1)
        max_pool  = enc.masked_fill(pad_mask.unsqueeze(-1), float('-inf')).max(1).values
        pooled    = self.pool_proj(torch.cat([mean_pool, max_pool], dim=-1))

        # Damage features — counts always available, Evo2 features only when ref present
        win5 = pos < BRIGGS_WIN                                          # (1, L)
        win3 = (dist_3p < BRIGGS_WIN) & ~pad_mask                       # (B, L)
        t5   = (x == 4) & win5    # T in 5' window (C→T candidates)
        a3   = (x == 1) & win3    # A in 3' window (G→A candidates)
        n_ct_5 = t5.float().sum(1) / BRIGGS_WIN                         # (B,)
        n_ga_3 = a3.float().sum(1) / BRIGGS_WIN                         # (B,)

        if self.use_evo2 and ref is not None and self.n_dmg_features >= 4:
            p_c        = ref[:, :, 2]                                    # Evo2 P(C)
            p_g        = ref[:, :, 3]                                    # Evo2 P(G)
            dmg_frac_5 = (p_c * t5.float()).sum(1) / t5.float().sum(1).clamp(min=1)
            dmg_frac_3 = (p_g * a3.float()).sum(1) / a3.float().sum(1).clamp(min=1)

            if self.n_dmg_features == 5:
                # Per-read mean log-likelihood: low for damaged reads (Evo2 expected C, saw T)
                valid_mask = ~pad_mask
                x_clamped  = x.clamp(0, VOCAB_SIZE - 1)
                p_actual   = ref.gather(-1, x_clamped.unsqueeze(-1)).squeeze(-1)
                evo2_ll    = (torch.log(p_actual.clamp(min=1e-8)) * valid_mask.float()).sum(1) \
                             / valid_mask.float().sum(1).clamp(min=1)
                dmg_feat = torch.stack([dmg_frac_5, n_ct_5, dmg_frac_3, n_ga_3, evo2_ll], dim=-1)
            else:
                dmg_feat = torch.stack([dmg_frac_5, n_ct_5, dmg_frac_3, n_ga_3], dim=-1)
        else:
            dmg_feat = torch.stack([n_ct_5, n_ga_3], dim=-1)

        return self.head(pooled + self.dmg_proj(dmg_feat)).squeeze(-1)   # (B,) logits


# ── Training ──────────────────────────────────────────────────────────────────

def train_classifier(model, tr_loader, va_loader, device, display_name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    use_evo2  = _unwrap(model).use_evo2
    best_metric, best_epoch, patience_cnt = -1e9, 0, 0
    best_state = None

    log(f'\n  Training {display_name}  (checkpoint metric: {CHECKPOINT_METRIC.upper()})...')
    log(f'  {"Epoch":>5}  {"LR":>8}  {"TrainLoss":>10}  {"PR-AUC":>8}  {"MCC":>7}')
    log(f'  {"-"*48}')

    for epoch in range(EPOCHS):
        if hasattr(tr_loader, 'batch_sampler') and \
           hasattr(tr_loader.batch_sampler, 'set_epoch'):
            tr_loader.batch_sampler.set_epoch(epoch)
        lr_now = LR * get_lr_scale(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        model.train()
        total_loss = 0.0
        for xb, lb, yb, rb in tr_loader:
            xb, lb, yb, rb = xb.to(device), lb.to(device), yb.float().to(device), rb.to(device)
            with _autocast_ctx(device):
                logits = model(xb, lb, rb if use_evo2 else None)
                loss   = focal_loss(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(_unwrap(model).parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for xb, lb, yb, rb in va_loader:
                xb, lb, rb = xb.to(device), lb.to(device), rb.to(device)
                with _autocast_ctx(device):
                    logits = model(xb, lb, rb if use_evo2 else None)
                all_logits.append(torch.sigmoid(logits).float().cpu().numpy())
                all_labels.append(yb.numpy())
        probs  = np.concatenate(all_logits)
        labels = np.concatenate(all_labels)
        pr_auc = average_precision_score(labels, probs) if labels.sum() > 0 else 0.0
        mcc    = mcc_at_best_threshold(probs, labels)
        log(f'  {epoch+1:>5}  {lr_now:>8.2e}  '
            f'{total_loss/len(tr_loader.dataset):>10.4f}  {pr_auc:>8.4f}  {mcc:>7.4f}')

        metric = mcc if CHECKPOINT_METRIC == 'mcc' else pr_auc
        if metric > best_metric:
            best_metric, best_epoch, patience_cnt = metric, epoch, 0
            best_state = {k: v.clone() for k, v in _unwrap(model).state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                log(f'  Early stop at epoch {epoch+1} '
                    f'(best {best_epoch+1}, {CHECKPOINT_METRIC}={best_metric:.4f})')
                break

    _unwrap(model).load_state_dict(best_state)
    return best_metric


def evaluate(model, loader, device, rc_tta=USE_RC_TTA):
    """Test-time inference. If rc_tta=True, average forward + reverse-complement
    sigmoid probabilities — typically +0.5–1 AUC point at no extra training cost."""
    model.eval()
    use_evo2 = _unwrap(model).use_evo2
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, lb, yb, rb in loader:
            xb, lb, rb = xb.to(device), lb.to(device), rb.to(device)
            with _autocast_ctx(device):
                logits_fwd = model(xb, lb, rb if use_evo2 else None)
                p = torch.sigmoid(logits_fwd).float()
                if rc_tta:
                    rc_x, rc_r = reverse_complement(xb, lb, rb if use_evo2 else None)
                    logits_rc = model(rc_x, lb, rc_r if use_evo2 else None)
                    p = 0.5 * (p + torch.sigmoid(logits_rc).float())
            all_probs.append(p.cpu().numpy())
            all_labels.append(yb.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ── Statistics helpers ────────────────────────────────────────────────────────

def all_threshold_stats(probs, labels, threshold):
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    n    = tn + fp + fn + tp
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1   = (2*ppv*sens / (ppv + sens)) if (ppv + sens) > 0 else 0.0
    acc  = (tp + tn) / n if n > 0 else 0.0
    bal  = (sens + spec) / 2
    fpr  = 1 - spec
    denom = np.sqrt(float(tp+fp)*float(tp+fn)*float(tn+fp)*float(tn+fn))
    mcc   = (float(tp)*float(tn) - float(fp)*float(fn)) / denom if denom > 0 else 0.0
    return dict(sens=sens, spec=spec, fpr=fpr, ppv=ppv, npv=npv,
                f1=f1, acc=acc, bal_acc=bal, mcc=mcc,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def youden_threshold(probs, labels):
    fpr_a, tpr_a, thr = roc_curve(labels, probs)
    return float(thr[np.argmax(tpr_a - fpr_a)])


def best_f1_threshold(probs, labels):
    prec, rec, thr = precision_recall_curve(labels, probs)
    with np.errstate(invalid='ignore'):
        f1 = 2 * prec * rec / (prec + rec)
    return float(thr[np.argmax(np.nan_to_num(f1[:-1]))])


def best_mcc_threshold(probs, labels, steps=201):
    thresholds = np.linspace(0.01, 0.99, steps)
    mccs = [all_threshold_stats(probs, labels, t)['mcc'] for t in thresholds]
    return float(thresholds[np.argmax(mccs)])


def full_threshold_sweep(probs, labels, steps=37):
    thresholds = np.linspace(0.05, 0.95, steps)
    rows = [all_threshold_stats(probs, labels, t) for t in thresholds]
    sweep = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    sweep['threshold'] = thresholds
    return sweep


# ── Main ──────────────────────────────────────────────────────────────────────

VARIANT_CONFIG = {
    'seq':      dict(use_evo2=False, n_feat=2),
    'evo_base': dict(use_evo2=True,  n_feat=4),
    'evo_full': dict(use_evo2=True,  n_feat=5),
}


def merge_variant_files():
    """Combine per-variant probs files into the unified classifier_probs.npz."""
    out_path = OUT_DIR / 'results' / 'classifier_probs.npz'
    parts    = {}
    labels   = None
    for variant in VARIANT_CONFIG.keys():
        p = OUT_DIR / 'results' / f'classifier_probs_{variant}.npz'
        if not p.exists():
            log(f'  [skip] {p.name} not found — variant {variant} not trained')
            continue
        d = np.load(p)
        parts[f'probs_{variant}'] = d[f'probs_{variant}']
        if labels is None:
            labels = d['labels']
        elif len(d['labels']) != len(labels) or not np.array_equal(d['labels'], labels):
            raise ValueError(f'Label mismatch in {p} — re-run all variants on the same dataset')
    if not parts:
        raise SystemExit('No per-variant probs files found. Run --variant {seq|evo_base|evo_full} first.')
    np.savez(out_path, labels=labels, **parts)
    log(f'Merged {len(parts)} variant(s) → {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Train aDNA read classifier(s).')
    parser.add_argument('--variant',
                        choices=['seq', 'evo_base', 'evo_full', 'all'],
                        default='all',
                        help='Train one variant in isolation (for parallel SLURM jobs) '
                             'or all three sequentially (default, legacy behaviour).')
    parser.add_argument('--merge', action='store_true',
                        help='Skip training; merge existing per-variant probs files '
                             'and produce final plots/summary.')
    args = parser.parse_args()

    (OUT_DIR / 'figures').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'results').mkdir(parents=True, exist_ok=True)

    # ── Merge mode: load probs from per-variant files; skip data + training ──
    if args.merge:
        log('Merging per-variant probs files...')
        merge_variant_files()
        d_unified = np.load(OUT_DIR / 'results' / 'classifier_probs.npz')
        labels    = d_unified['labels']
        probs_map = {k: d_unified[f'probs_{k}']
                     for k in VARIANT_CONFIG.keys() if f'probs_{k}' in d_unified}
        has_evo2  = any(k in probs_map for k in ('evo_base', 'evo_full'))
        # Falls through to plot/summary section below
    else:
        # ── Training mode: load data and train selected variant(s) ───────────
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log(f'Device: {device}  |  Variant: {args.variant}')

        has_evo2 = 'ref_evo2' in np.load(DATA_DIR / 'train.npz')

        log('Loading data...')
        tr_dmg, tr_cln, tr_len, tr_lbl, tr_ref = load_split('train', with_evo2=has_evo2)
        va_dmg, va_cln, va_len, va_lbl, va_ref = load_split('val',   with_evo2=has_evo2)
        te_dmg, te_cln, te_len, te_lbl, te_ref = load_split('test',  with_evo2=has_evo2)
        log(f'  Train: {len(tr_lbl):,}  ({tr_lbl.mean()*100:.1f}% ancient)')
        log(f'  Val  : {len(va_lbl):,}  ({va_lbl.mean()*100:.1f}% ancient)')
        log(f'  Test : {len(te_lbl):,}  ({te_lbl.mean()*100:.1f}% ancient)')

        if tr_ref is None:
            log('  No Evo2 reference — training sequence-only model only.')
            has_evo2 = False

        # Length-bucketed, class-balanced sampler for training. Each training
        # batch is drawn from a single length
        # quantile bucket and preserves the natural ancient/modern prevalence.
        # Val / test loaders stay sequential for deterministic evaluation.
        from length_bucketed_sampler import build_loader as build_bucketed_loader

        def make_loader(dmg, lns, lbl, ref, shuffle):
            ref_arr = ref if ref is not None else np.zeros((len(dmg), MAX_LEN, VOCAB_SIZE), np.float32)
            ds = ReadDataset(dmg, lns, lbl, ref_arr)
            if shuffle:
                return build_bucketed_loader(
                    ds,
                    lengths=np.asarray(lns),
                    labels=np.asarray(lbl).astype(np.int64),
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    n_buckets=10,
                    ancient_frac=None,   # match natural ~25 % prevalence
                    seed=42,
                    num_workers=2,
                    pin_memory=True,
                )
            return torch.utils.data.DataLoader(
                ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        tr_loader = make_loader(tr_dmg, tr_len, tr_lbl, tr_ref, shuffle=True)
        va_loader = make_loader(va_dmg, va_len, va_lbl, va_ref, shuffle=False)
        te_loader = make_loader(te_dmg, te_len, te_lbl, te_ref, shuffle=False)

        probs_map = {}   # key → averaged ensemble probabilities
        labels    = None

    def build_model(use_evo2, n_feat):
        return aDNAClassifier(use_evo2=use_evo2, n_dmg_features=n_feat).to(device)

    def train_ensemble(use_evo2, n_feat, key, name):
        """Train N_SEEDS models, return averaged test probabilities.

        Also saves per-seed weight checkpoints to outputs/models/ so the
        models can be re-evaluated on alternative test sets later
        (e.g. the UDG cross-protocol test in Code/13.1_evaluate_udg.py).
        """
        nonlocal labels
        models_dir = OUT_DIR / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        n_par = sum(p.numel() for p in build_model(use_evo2, n_feat).parameters())
        log(f'\n  {name}: {n_par:,} params  ×  {N_SEEDS} seed(s)')
        seed_probs = []
        for s_idx, seed in enumerate(SEEDS):
            torch.manual_seed(seed); np.random.seed(seed)
            log(f'\n  ── {name}  seed {seed}  ({s_idx+1}/{N_SEEDS}) ─────')
            mdl = build_model(use_evo2, n_feat)
            train_classifier(mdl, tr_loader, va_loader, device, f'{name} (seed={seed})')
            p, lab = evaluate(mdl, te_loader, device)
            if labels is None: labels = lab
            seed_probs.append(p)
            log(f'  seed-{seed} test ROC-AUC: {roc_auc_score(lab, p):.4f}')

            # Save weights so the model can be re-evaluated on UDG/other test sets
            ckpt_path = models_dir / f'classifier_{key}_seed{seed}.pt'
            torch.save(_unwrap(mdl).state_dict(), ckpt_path)
            log(f'  saved weights → {ckpt_path}')

            del mdl; torch.cuda.empty_cache() if device.type == 'cuda' else None
        avg = np.mean(np.stack(seed_probs, axis=0), axis=0)
        log(f'\n  {name} ENSEMBLE ({N_SEEDS}-seed) test ROC-AUC: '
            f'{roc_auc_score(labels, avg):.4f}')
        probs_map[key] = avg

    # ── Train the selected variant(s) ─────────────────────────────────────────
    if not args.merge:
        if args.variant == 'all':
            # Legacy behaviour: train everything sequentially in one job
            wanted = ['seq']
            if has_evo2:
                wanted += ['evo_base', 'evo_full']
        else:
            if args.variant in ('evo_base', 'evo_full') and not has_evo2:
                raise SystemExit(f'ERROR: --variant {args.variant} requires ref_evo2 in NPZ '
                                 f'(run Code/5.1_evo2_refs.py first).')
            wanted = [args.variant]

        for key in wanted:
            cfg = VARIANT_CONFIG[key]
            train_ensemble(use_evo2=cfg['use_evo2'], n_feat=cfg['n_feat'],
                           key=key, name=MODEL_NAMES[key])

        # Save per-variant probs file (or unified if 'all')
        if args.variant == 'all':
            save_dict = {'labels': labels}
            for k in wanted:
                save_dict[f'probs_{k}'] = probs_map[k]
            out_p = OUT_DIR / 'results' / 'classifier_probs.npz'
            np.savez(out_p, **save_dict)
            log(f'\nSaved → {out_p}')
        else:
            k = args.variant
            out_p = OUT_DIR / 'results' / f'classifier_probs_{k}.npz'
            np.savez(out_p, labels=labels, **{f'probs_{k}': probs_map[k]})
            log(f'\nSaved → {out_p}')
            log(f'Training of variant "{k}" complete. Run with --merge once all '
                f'variants are done to produce final plots/summary.')
            return  # Skip the unified plot/summary — they require all variants

    # ── Below: runs in --merge mode and in --variant=all mode ────────────────
    # (Single-variant mode returned early above and skipped this section.)
    # probs_map and labels are populated either from training (above) or from
    # the merge step at the top of main().

    # ── Compute stats for all trained models ──────────────────────────────────
    name_map = {'seq': MODEL_NAMES['seq'],
                'evo_base': MODEL_NAMES['evo_base'],
                'evo_full': MODEL_NAMES['evo_full']}
    order = [k for k in ['evo_full', 'evo_base', 'seq'] if k in probs_map]

    stats  = {}
    sweeps = {}
    for key in order:
        p       = probs_map[key]
        t_y     = youden_threshold(p, labels)
        t_f     = best_f1_threshold(p, labels)
        t_m     = best_mcc_threshold(p, labels)
        stats[key] = {
            'roc_auc':   roc_auc_score(labels, p),
            'pr_auc':    average_precision_score(labels, p),
            't_youden':  t_y, 't_f1': t_f, 't_mcc': t_m,
            'at_youden': all_threshold_stats(p, labels, t_y),
            'at_f1':     all_threshold_stats(p, labels, t_f),
            'at_mcc':    all_threshold_stats(p, labels, t_m),
            'at_50':     all_threshold_stats(p, labels, 0.50),
        }
        sweeps[key] = full_threshold_sweep(p, labels)

    # ── Figure 1: ROC / PR / metric bar chart ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('aDNA Read Classifier: Three Model Variants',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    for key in order:
        name = name_map[key]
        p    = probs_map[key]
        fpr_a, tpr_a, _ = roc_curve(labels, p)
        ax.plot(fpr_a, tpr_a, color=COLORS[name], lw=2,
                label=f'{name}\n(AUC = {stats[key]["roc_auc"]:.3f})')
        s = stats[key]['at_youden']
        ax.scatter([s['fpr']], [s['sens']], color=COLORS[name], s=70, zorder=5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.35, lw=1, label='Random (0.500)')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (● = Youden threshold)'); ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1]
    random_pr = labels.mean()
    for key in order:
        name = name_map[key]
        p    = probs_map[key]
        prec_a, rec_a, _ = precision_recall_curve(labels, p)
        ax.plot(rec_a, prec_a, color=COLORS[name], lw=2,
                label=f'{name}\n(AP = {stats[key]["pr_auc"]:.3f})')
    ax.axhline(random_pr, color='k', ls='--', alpha=0.35, lw=1,
               label=f'Random ({random_pr:.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve'); ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[2]
    m_keys = ['roc_auc', 'pr_auc', 'sens', 'spec', 'ppv', 'npv', 'f1', 'bal_acc', 'mcc', 'acc']
    m_lbls = ['ROC-AUC', 'PR-AUC', 'Sens', 'Spec', 'PPV', 'NPV', 'F1', 'BalAcc', 'MCC', 'Acc']
    x     = np.arange(len(m_keys))
    width = 0.8 / len(order)
    for i, key in enumerate(order):
        name = name_map[key]
        s    = stats[key]
        vals = [s['roc_auc'], s['pr_auc']] + [s['at_youden'][k] for k in m_keys[2:]]
        ax.bar(x + (i - len(order)/2 + 0.5)*width, vals, width,
               label=name, color=COLORS[name], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(m_lbls, fontsize=8, rotation=30, ha='right')
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
    ax.set_title('All metrics at Youden threshold'); ax.legend(fontsize=7.5)
    ax.axhline(0.5, color='grey', ls=':', alpha=0.4, lw=1)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'classifier.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 2: Full threshold sweep for best model ─────────────────────────
    best_key  = order[0]
    best_name = name_map[best_key]
    sw        = sweeps[best_key]
    thr       = sw['threshold']

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(f'Threshold Sweep — {best_name}', fontsize=13, fontweight='bold')

    def _ax(ax, y_keys, y_labels, title, hlines=None):
        line_styles = ['-', '--', '-.', ':']
        for j, (k, lbl) in enumerate(zip(y_keys, y_labels)):
            ax.plot(thr, sw[k], ls=line_styles[j % 4], lw=2, label=lbl)
        if hlines:
            for val, lbl in hlines:
                ax.axhline(val, color='grey', ls=':', lw=1, alpha=0.6, label=lbl)
        ax.axvline(stats[best_key]['t_youden'], color='black', ls='--', lw=1,
                   alpha=0.5, label=f'Youden ({stats[best_key]["t_youden"]:.2f})')
        ax.axvline(stats[best_key]['t_f1'], color='purple', ls=':', lw=1,
                   alpha=0.5, label=f'Best-F1 ({stats[best_key]["t_f1"]:.2f})')
        ax.set_xlabel('Threshold'); ax.set_title(title)
        ax.set_xlim(0.05, 0.95); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    _ax(axes[0, 0], ['ppv', 'sens', 'f1'],
        ['Precision (PPV)', 'Recall (Sensitivity)', 'F1'], 'Precision / Recall / F1')
    _ax(axes[0, 1], ['acc', 'bal_acc'], ['Accuracy', 'Balanced Accuracy'], 'Accuracy',
        hlines=[(labels.mean(), f'Prevalence ({labels.mean():.3f})')])
    _ax(axes[0, 2], ['mcc'], ['MCC'], 'Matthews Correlation Coefficient')
    _ax(axes[1, 0], ['spec', 'fpr', 'npv'], ['Specificity', 'FPR', 'NPV'],
        'Specificity / FPR / NPV')

    ax = axes[1, 1]
    n_total = len(labels)
    for k, lbl, ls in [('tp', 'TP (ancient correctly flagged)', '-'),
                        ('fp', 'FP (modern falsely flagged)',   '--'),
                        ('fn', 'FN (ancient missed)',           '-.'),
                        ('tn', 'TN (modern correctly passed)',  ':')]:
        ax.plot(thr, sw[k] / n_total, ls=ls, lw=2, label=lbl)
    ax.axvline(stats[best_key]['t_youden'], color='black', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Threshold'); ax.set_title('TP / FP / FN / TN (fraction of test set)')
    ax.set_xlim(0.05, 0.95); ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1, 2]
    t_y  = stats[best_key]['t_youden']
    s_y  = stats[best_key]['at_youden']
    cm_v = np.array([[s_y['tn'], s_y['fp']], [s_y['fn'], s_y['tp']]])
    im   = ax.imshow(cm_v, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Modern', 'Pred: Ancient'], fontsize=10)
    ax.set_yticklabels(['True: Modern', 'True: Ancient'], fontsize=10)
    ax.set_title(f'Confusion Matrix at Youden threshold ({t_y:.3f})')
    for (r, c_), val in np.ndenumerate(cm_v):
        ax.text(c_, r, f'{val:,}\n({val/n_total*100:.1f}%)',
                ha='center', va='center', fontsize=10,
                color='white' if cm_v[r, c_] > cm_v.max() * 0.6 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / 'classifier_threshold_sweep.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Text summary ──────────────────────────────────────────────────────────
    w1, w2 = 30, 8
    header = (f'{"Method":<{w1}}  {"ROC-AUC":>{w2}}  {"PR-AUC":>{w2}}  '
              f'{"Sens":>{w2}}  {"Spec":>{w2}}  {"PPV":>{w2}}  {"NPV":>{w2}}  '
              f'{"F1":>{w2}}  {"Acc":>{w2}}  {"BalAcc":>{w2}}  {"MCC":>{w2}}')
    sep = '-' * len(header)

    def fmt_row(key, stat_key='at_youden'):
        s  = stats[key]
        ay = s[stat_key]
        return (f'  {name_map[key]:<{w1-2}}  {s["roc_auc"]:>{w2}.4f}  {s["pr_auc"]:>{w2}.4f}  '
                f'{ay["sens"]:>{w2}.4f}  {ay["spec"]:>{w2}.4f}  {ay["ppv"]:>{w2}.4f}  '
                f'{ay["npv"]:>{w2}.4f}  {ay["f1"]:>{w2}.4f}  {ay["acc"]:>{w2}.4f}  '
                f'{ay["bal_acc"]:>{w2}.4f}  {ay["mcc"]:>{w2}.4f}')

    lines = [
        'aDNA Read Classifier — Full Evaluation',
        '=' * len(header),
        f'Test set: {len(labels):,} reads  ({labels.mean()*100:.1f}% ancient / '
        f'{(1-labels.mean())*100:.1f}% modern)',
        '',
        '── At Youden J threshold ──', header, sep,
    ] + [fmt_row(k, 'at_youden') for k in order]

    lines += ['', '── At best-MCC threshold (recommended) ──', header, sep]
    lines += [fmt_row(k, 'at_mcc') for k in order]

    lines += ['', '── At best-F1 threshold ──', header, sep]
    lines += [fmt_row(k, 'at_f1') for k in order]

    lines += ['', '── At fixed threshold = 0.50 ──', header, sep]
    lines += [fmt_row(k, 'at_50') for k in order]

    lines += ['', '── Optimal thresholds per model ──',
              f'  {"Method":<{w1-2}}  {"Youden":>10}  {"Best-MCC":>10}  {"Best-F1":>10}',
              '  ' + '-' * 36]
    for k in order:
        lines.append(f'  {name_map[k]:<{w1-2}}  {stats[k]["t_youden"]:>10.4f}  '
                     f'{stats[k]["t_mcc"]:>10.4f}  {stats[k]["t_f1"]:>10.4f}')

    # Full sweep for best model
    lines += ['', f'── Full threshold sweep — {best_name} ──',
              f'  {"Thr":>6}  {"Sens":>7}  {"Spec":>7}  {"PPV":>7}  '
              f'{"NPV":>7}  {"F1":>7}  {"Acc":>7}  {"BalAcc":>7}  '
              f'{"MCC":>7}  {"TP":>8}  {"FP":>8}  {"FN":>8}  {"TN":>8}',
              '  ' + '-' * 108]
    for i, t in enumerate(sw['threshold']):
        lines.append(
            f'  {t:>6.2f}  {sw["sens"][i]:>7.4f}  {sw["spec"][i]:>7.4f}  '
            f'{sw["ppv"][i]:>7.4f}  {sw["npv"][i]:>7.4f}  {sw["f1"][i]:>7.4f}  '
            f'{sw["acc"][i]:>7.4f}  {sw["bal_acc"][i]:>7.4f}  {sw["mcc"][i]:>7.4f}  '
            f'{sw["tp"][i]:>8,}  {sw["fp"][i]:>8,}  {sw["fn"][i]:>8,}  {sw["tn"][i]:>8,}'
        )

    lines += ['',
              'Metric definitions:',
              '  Sens   = Sensitivity / Recall / TPR = TP / (TP + FN)',
              '  Spec   = Specificity / TNR           = TN / (TN + FP)',
              '  PPV    = Precision                   = TP / (TP + FP)',
              '  NPV    = Negative Predictive Value   = TN / (TN + FN)',
              '  F1     = 2 * PPV * Sens / (PPV + Sens)',
              '  MCC    = Matthews Correlation Coefficient ∈ [−1, 1]',
              '  BalAcc = (Sensitivity + Specificity) / 2',
              f'  Random PR baseline = {labels.mean():.4f} (class prevalence)']

    summary = '\n'.join(lines)
    (OUT_DIR / 'results' / 'classifier_summary.txt').write_text(summary)
    log(summary)
    log(f'\nFigure 1 → outputs/figures/classifier.png')
    log(f'Figure 2 → outputs/figures/classifier_threshold_sweep.png')
    log(f'Summary  → outputs/results/classifier_summary.txt')


if __name__ == '__main__':
    main()
