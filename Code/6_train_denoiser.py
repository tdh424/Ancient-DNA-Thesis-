"""6_train_denoiser.py — Train the DNA denoiser (seq_only / evo2 / bwa / udg variants)."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

DATA_DIR      = Path('data')
OUT_DIR       = Path('outputs')

MAX_LEN       = 100
BATCH_SIZE    = 4096
LEARNING_RATE = 5e-5
EPOCHS        = 80
PATIENCE      = 15
WARMUP_EPOCHS = 5

# Two model size presets — selectable via --size.
MODEL_SIZES = {
    'small': dict(d_model=128, n_heads=8, n_layers=4, dim_ff=256),
    'large': dict(d_model=256, n_heads=8, n_layers=6, dim_ff=512),
}

DROPOUT       = 0.1
DAMAGE_WEIGHT = 200
FOCAL_GAMMA   = 2.0
BRIGGS_WIN    = 8

USE_BF16      = True   # autocast to bfloat16 on CUDA
BALANCE_CT_GA = True   # per-batch class-balance the C->T / G->A loss weights

VOCAB_SIZE = 6   # PAD=0 A=1 C=2 G=3 T=4 N=5
N_CLASSES  = 6


def log(msg=''):
    print(msg, flush=True)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_split(name, variant):
    """Load NPZ and return tensors appropriate for the given variant.

    The udg variant reads from data/{name}_udg.npz instead of data/{name}.npz
    but uses the seq-only architecture (no external reference signal). It
    isolates the effect of training on UDG-treated reads.
    """
    suffix = '_udg' if variant == 'udg' else ''
    d = np.load(DATA_DIR / f'{name}{suffix}.npz')
    damaged = torch.from_numpy(d['damaged'].astype(np.int64))
    clean   = torch.from_numpy(d['clean'].astype(np.int64))
    lengths = torch.from_numpy(d['lengths'].astype(np.int64))

    if variant == 'evo2':
        if 'ref_evo2' not in d:
            raise KeyError(f'{name}.npz missing ref_evo2 — run 5.1_evo2_refs.py first.')
        ref = torch.from_numpy(d['ref_evo2'].astype(np.float32))
    elif variant == 'bwa':
        if 'ref_bwa' not in d:
            raise KeyError(f'{name}.npz missing ref_bwa — run 5.2_bwa_refs.py first.')
        bwa_int = torch.from_numpy(d['ref_bwa'].astype(np.int64))
        ref = F.one_hot(bwa_int.clamp(0, VOCAB_SIZE - 1), num_classes=VOCAB_SIZE).float()
    else:
        ref = None

    return damaged, clean, ref, lengths


class NPZDataset(torch.utils.data.Dataset):
    def __init__(self, damaged, clean, ref, lengths):
        self.x, self.y, self.l = damaged, clean, lengths
        self.r = ref  # None for seq_only

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        r = self.r[i] if self.r is not None else torch.zeros(MAX_LEN, VOCAB_SIZE)
        return self.x[i], self.y[i], r, self.l[i]


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


class DNADenoiser(nn.Module):
    def __init__(self, use_ref=True, d_model=128, n_heads=8, n_layers=4, dim_ff=256):
        super().__init__()
        self.use_ref   = use_ref
        self.d_model   = d_model
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=0)
        if use_ref:
            self.ref_proj = nn.Linear(VOCAB_SIZE + 1, d_model, bias=False)
        self.pos_enc   = SinusoidalPE(d_model)
        self.rev_pos   = SinusoidalPE(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head    = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, N_CLASSES))

    def forward(self, x, ref=None, lengths=None):
        _, L = x.shape
        emb  = self.embedding(x)

        if self.use_ref and ref is not None:
            entropy = -(ref * torch.log(ref.clamp(min=1e-8))).sum(-1, keepdim=True)
            emb     = emb + self.ref_proj(torch.cat([ref, entropy], dim=-1))

        emb = self.pos_enc(emb)

        pad_mask = None
        if lengths is not None:
            pos      = torch.arange(L, device=x.device).unsqueeze(0)
            dist_3p  = (lengths.unsqueeze(1) - 1 - pos).clamp(0, L - 1)
            emb      = emb + self.rev_pos.pe[0][dist_3p]
            pad_mask = pos >= lengths.unsqueeze(1)

        return self.head(self.encoder(emb, src_key_padding_mask=pad_mask))


# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_fn(logits, targets, inputs, lengths):
    """Focal cross-entropy with 200× damage upweight in Briggs window."""
    B, L, C = logits.shape
    log_p   = torch.log_softmax(logits, dim=-1)
    p_t     = log_p.exp().gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    focal_w = (1 - p_t).pow(FOCAL_GAMMA).detach()

    ce_flat = -log_p.reshape(-1, C)[
        torch.arange(B * L, device=logits.device),
        targets.reshape(-1)
    ].reshape(B, L)

    weights = torch.ones(B, L, device=logits.device)
    pos     = torch.arange(L, device=logits.device).unsqueeze(0)
    dist_3p = (lengths.unsqueeze(1) - 1 - pos).clamp(min=0)

    ct_mask = (inputs == 4) & (targets == 2) & (pos < BRIGGS_WIN)   # C→T at 5'
    ga_mask = (inputs == 1) & (targets == 3) & (dist_3p < BRIGGS_WIN)  # G→A at 3'

    if BALANCE_CT_GA:
        # Per-batch class balancing — give C→T and G→A equal total weight mass
        # regardless of count imbalance (typically 3-4× more C→T events).
        n_ct = ct_mask.sum().clamp(min=1).float()
        n_ga = ga_mask.sum().clamp(min=1).float()
        ct_w = DAMAGE_WEIGHT * (1.0 / n_ct).clamp(max=1.0)
        ga_w = DAMAGE_WEIGHT * (1.0 / n_ga).clamp(max=1.0)
        # Renormalize so the sum of damage weights equals DAMAGE_WEIGHT × (n_ct + n_ga)
        scale = (n_ct + n_ga) / (ct_w * n_ct + ga_w * n_ga + 1e-8)
        weights[ct_mask] = (ct_w * scale).item()
        weights[ga_mask] = (ga_w * scale).item()
    else:
        weights[ct_mask] = DAMAGE_WEIGHT
        weights[ga_mask] = DAMAGE_WEIGHT

    valid = (pos < lengths.unsqueeze(1)).float()
    return (ce_flat * focal_w * weights * valid).sum() / valid.sum().clamp(min=1)


# ── Training loop ─────────────────────────────────────────────────────────────

def _autocast_ctx(device):
    if USE_BF16 and device.type == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    from contextlib import nullcontext
    return nullcontext()


def _unwrap(model):
    """Return the underlying nn.Module — torch.compile wraps it."""
    return getattr(model, '_orig_mod', model)


def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total_loss, total, correct = 0.0, 0, 0
    use_ref = _unwrap(model).use_ref
    with torch.set_grad_enabled(train):
        for xb, yb, rb, lb in loader:
            xb, yb, rb, lb = xb.to(device), yb.to(device), rb.to(device), lb.to(device)
            ref_in = rb if use_ref else None
            with _autocast_ctx(device):
                logits = model(xb, ref=ref_in, lengths=lb)
                loss   = loss_fn(logits, yb, xb, lb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(xb)
            preds = logits.argmax(-1)
            pos   = torch.arange(logits.shape[1], device=xb.device).unsqueeze(0)
            valid = pos < lb.unsqueeze(1)
            total   += valid.sum().item()
            correct += ((preds == yb) & valid).sum().item()
    return total_loss / len(loader.dataset), correct / max(total, 1)


def val_prauc(model, loader, device):
    """PR-AUC on C→T and G→A correction as validation metric."""
    model.eval()
    use_ref = _unwrap(model).use_ref
    pc_list, ct_lbl = [], []
    pg_list, ga_lbl = [], []
    with torch.no_grad():
        for xb, yb, rb, lb in loader:
            xb, yb, rb, lb = xb.to(device), yb.to(device), rb.to(device), lb.to(device)
            ref_in = rb if use_ref else None
            with _autocast_ctx(device):
                logits = model(xb, ref=ref_in, lengths=lb)
            # softmax in fp32 — numpy cannot consume bfloat16 tensors
            probs  = torch.softmax(logits.float(), -1)
            pos    = torch.arange(logits.shape[1], device=xb.device).unsqueeze(0)
            valid  = pos < lb.unsqueeze(1)
            t_pos  = (xb == 4) & valid
            a_pos  = (xb == 1) & valid
            pc_list.append(probs[:, :, 2].reshape(-1)[t_pos.reshape(-1)].cpu().numpy())
            ct_lbl.append((yb.reshape(-1)[t_pos.reshape(-1)] == 2).cpu().numpy().astype(np.int32))
            pg_list.append(probs[:, :, 3].reshape(-1)[a_pos.reshape(-1)].cpu().numpy())
            ga_lbl.append((yb.reshape(-1)[a_pos.reshape(-1)] == 3).cpu().numpy().astype(np.int32))

    pc = np.concatenate(pc_list); lc = np.concatenate(ct_lbl)
    pg = np.concatenate(pg_list); lg = np.concatenate(ga_lbl)
    pa_ct = average_precision_score(lc, pc) if lc.sum() > 0 else float('nan')
    pa_ga = average_precision_score(lg, pg) if lg.sum() > 0 else float('nan')
    return 0.5 * (pa_ct + pa_ga)


def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1 + np.cos(np.pi * progress))


# ── Main ──────────────────────────────────────────────────────────────────────

def train_one(variant, size, seed):
    """Train a single denoiser instance. Returns (best_prauc, history)."""
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_ref = variant in ('evo2', 'bwa')
    cfg     = MODEL_SIZES[size]

    torch.manual_seed(seed)
    np.random.seed(seed)

    log(f'\n── {variant} | size={size} | seed={seed} ────────────────')
    log(f'Device  : {device}')

    log('Loading data...')
    tr_x, tr_y, tr_r, tr_l = load_split('train', variant)
    va_x, va_y, va_r, va_l = load_split('val',   variant)
    log(f'  Train: {len(tr_x):,}   Val: {len(va_x):,}')

    # Length-bucketed, class-balanced sampler.
    # Each training batch is drawn from a single length quantile bucket and
    # preserves the natural ancient/modern prevalence. The validation loader
    # is a plain sequential loader so reported metrics are deterministic.
    from length_bucketed_sampler import build_loader as build_bucketed_loader
    tr_labels = ((tr_x != tr_y).any(dim=1) & (tr_x.sum(dim=1) > 0)).cpu().numpy().astype(np.int64)
    tr_loader = build_bucketed_loader(
        NPZDataset(tr_x, tr_y, tr_r, tr_l),
        lengths=tr_l.cpu().numpy(),
        labels=tr_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        n_buckets=10,
        ancient_frac=None,   # match natural ~25 % ancient prevalence
        seed=42,
        num_workers=0,
        pin_memory=True,
    )
    va_loader = torch.utils.data.DataLoader(
        NPZDataset(va_x, va_y, va_r, va_l), batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model    = DNADenoiser(use_ref=use_ref, **cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'Model   : {n_params:,} parameters ({size})')
    if device.type == 'cuda':
        amp_dtype = 'bfloat16' if (USE_BF16 and torch.cuda.is_bf16_supported()) else 'float16'
        log(f'Autocast: {amp_dtype} ({torch.cuda.get_device_name(0)})')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {'train_loss': [], 'val_loss': [], 'val_prauc': []}
    best_prauc, best_epoch, patience_cnt = 0.0, 0, 0
    suffix  = f'_seed{seed}' if seed != 42 else ''
    ckpt    = OUT_DIR / 'models' / f'denoiser_{variant}{suffix}.pt'

    log(f'\n{"Epoch":>5}  {"TrainLoss":>10}  {"ValLoss":>9}  {"ValPRAUC":>9}  {"LR":>9}')
    log('-' * 55)
    for epoch in range(EPOCHS):
        if hasattr(tr_loader.batch_sampler, 'set_epoch'):
            tr_loader.batch_sampler.set_epoch(epoch)
        tr_loss, _ = run_epoch(model, tr_loader, optimizer, device, train=True)
        va_loss, _ = run_epoch(model, va_loader, optimizer, device, train=False)
        prauc      = val_prauc(model, va_loader, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['val_prauc'].append(prauc)

        lr_now = optimizer.param_groups[0]['lr']
        log(f'{epoch+1:>5}  {tr_loss:>10.4f}  {va_loss:>9.4f}  {prauc:>9.4f}  {lr_now:>9.2e}')

        if prauc > best_prauc:
            best_prauc, best_epoch, patience_cnt = prauc, epoch, 0
            torch.save(_unwrap(model).state_dict(), ckpt)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                log(f'Early stop at epoch {epoch+1} (best {best_epoch+1}, PR-AUC {best_prauc:.4f})')
                break

    log(f'\nBest epoch {best_epoch+1}, val PR-AUC {best_prauc:.4f}')
    log(f'Saved → {ckpt}')
    return best_prauc, history, best_epoch


def main(variant, size, seeds):
    (OUT_DIR / 'models').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'figures').mkdir(parents=True, exist_ok=True)

    log(f'Variant : {variant}')
    log(f'Size    : {size}')
    log(f'Seeds   : {seeds}')

    all_histories = []
    all_praucs    = []
    for seed in seeds:
        prauc, history, best_epoch = train_one(variant, size, seed)
        all_histories.append(history)
        all_praucs.append(prauc)

    cfg = MODEL_SIZES[size]
    config = {
        **cfg, 'dropout': DROPOUT, 'trunc': MAX_LEN,
        'vocab_size': VOCAB_SIZE, 'n_classes': N_CLASSES,
        'use_ref': variant in ('evo2', 'bwa'), 'variant': variant, 'size': size,
        'seeds': seeds, 'best_val_prauc_per_seed': all_praucs,
        'best_val_prauc_mean': float(np.mean(all_praucs)),
    }
    (OUT_DIR / 'models' / f'config_{variant}.json').write_text(json.dumps(config, indent=2))

    # Plot training curves from the first seed (representative)
    history    = all_histories[0]
    best_epoch = int(np.argmax(history['val_prauc']))
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig = plt.gcf()
    fig.suptitle(f'Training curves — {variant} ({size}, seed={seeds[0]})', fontweight='bold')
    axes[0].plot(history['train_loss'], label='Train loss')
    axes[0].plot(history['val_loss'],   label='Val loss')
    axes[0].axvline(best_epoch, color='red', ls='--', label=f'Best (ep {best_epoch+1})')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title='Training loss')
    axes[0].legend()
    axes[1].plot(history['val_prauc'], color='green')
    axes[1].axvline(best_epoch, color='red', ls='--', label=f'Best (ep {best_epoch+1})')
    axes[1].set(xlabel='Epoch', ylabel='PR-AUC', title='Validation PR-AUC')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'figures' / f'training_curves_{variant}.png', dpi=150, bbox_inches='tight')
    plt.close()

    if len(seeds) > 1:
        log(f'\nSeed ensemble ({len(seeds)} seeds): '
            f'mean val PR-AUC = {np.mean(all_praucs):.4f} '
            f'(± {np.std(all_praucs):.4f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the DNADenoiser.')
    parser.add_argument('--variant', choices=['seq_only', 'evo2', 'bwa', 'udg'],
                        default='evo2', help='Model variant')
    parser.add_argument('--size', choices=['small', 'large'], default='small',
                        help='Model size preset')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='One or more random seeds (e.g. --seeds 42 43 44)')
    args = parser.parse_args()
    main(args.variant, args.size, args.seeds)
