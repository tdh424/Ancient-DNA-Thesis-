"""7_denoise.py — Apply a trained denoiser to the test set."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path('data')
OUT_DIR    = Path('outputs')
THRESHOLD  = 0.5
BATCH      = 512
MAX_LEN    = 100
VOCAB_SIZE = 6
N_CLASSES  = 6
# ─────────────────────────────────────────────────────────────────────────────


def log(msg=''):
    print(msg, flush=True)


# ── Model (mirrors 6_train_denoiser.py — must stay in sync) ──────────────────

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
    def __init__(self, cfg):
        super().__init__()
        d            = cfg['d_model']
        self.use_ref = cfg.get('use_ref', True)
        self.embedding = nn.Embedding(cfg['vocab_size'], d, padding_idx=0)
        if self.use_ref:
            self.ref_proj = nn.Linear(cfg['vocab_size'] + 1, d, bias=False)
        self.pos_enc  = SinusoidalPE(d)
        self.rev_pos  = SinusoidalPE(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg['n_heads'], dim_feedforward=cfg['dim_ff'],
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg['n_layers'])
        self.head    = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg['n_classes']))

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


def load_model(variant, seed, device):
    cfg_path = OUT_DIR / 'models' / f'config_{variant}.json'
    suffix   = f'_seed{seed}' if seed != 42 else ''
    wts_path = OUT_DIR / 'models' / f'denoiser_{variant}{suffix}.pt'
    if not cfg_path.exists() or not wts_path.exists():
        raise FileNotFoundError(
            f'Missing checkpoint for {variant} (seed {seed}) — run '
            f'Code/6_train_denoiser.py --variant {variant} --seeds {seed} first.'
        )
    cfg   = json.loads(cfg_path.read_text())
    model = DNADenoiser(cfg).to(device)
    model.load_state_dict(torch.load(wts_path, map_location=device))
    model.eval()
    return model, cfg


def load_ref(d, variant):
    """Return the per-position reference tensor for the given variant."""
    if variant == 'evo2':
        return d['ref_evo2'].astype(np.float32)
    if variant == 'bwa':
        bwa_int = d['ref_bwa'].astype(np.int64)
        ref_t   = torch.from_numpy(bwa_int)
        return F.one_hot(ref_t.clamp(0, VOCAB_SIZE - 1), VOCAB_SIZE).numpy().astype(np.float32)
    return None


def infer_one_seed(model, damaged, lengths, ref, device):
    """Run inference with a single model, return per-position prob_c, prob_g."""
    N = len(damaged)
    prob_c = np.zeros((N, MAX_LEN), dtype=np.float32)
    prob_g = np.zeros((N, MAX_LEN), dtype=np.float32)
    for i in range(0, N, BATCH):
        xb = torch.from_numpy(damaged[i:i+BATCH]).to(device)
        lb = torch.from_numpy(lengths[i:i+BATCH].astype(np.int64)).to(device)
        rb = torch.from_numpy(ref[i:i+BATCH]).to(device) if ref is not None else None
        with torch.no_grad():
            logits = model(xb, ref=rb, lengths=lb)
            probs  = torch.softmax(logits, -1).cpu().numpy()
        prob_c[i:i+BATCH] = probs[:, :, 2]
        prob_g[i:i+BATCH] = probs[:, :, 3]
    return prob_c, prob_g


def main(variant, seeds):
    (OUT_DIR / 'results').mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'Variant: {variant}')
    log(f'Device : {device}')
    log(f'Seeds  : {seeds}')

    # Read the appropriate test NPZ (UDG variant uses test_udg.npz)
    test_path = DATA_DIR / ('test_udg.npz' if variant == 'udg' else 'test.npz')
    log(f'Loading test data from {test_path}...')
    d       = np.load(test_path)
    damaged = d['damaged'].astype(np.int64)
    clean   = d['clean'].astype(np.int64)
    lengths = d['lengths']
    ref     = load_ref(d, variant)
    N       = len(damaged)
    log(f'  {N:,} test reads')

    # Average prob_c, prob_g across all seed checkpoints.
    prob_c_sum = np.zeros((N, MAX_LEN), dtype=np.float32)
    prob_g_sum = np.zeros((N, MAX_LEN), dtype=np.float32)
    for seed in seeds:
        log(f'\n── Inference with seed {seed} ──')
        model, _ = load_model(variant, seed, device)
        pc, pg = infer_one_seed(model, damaged, lengths, ref, device)
        prob_c_sum += pc
        prob_g_sum += pg
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    prob_c = prob_c_sum / len(seeds)
    prob_g = prob_g_sum / len(seeds)

    denoised = damaged.copy()
    denoised[(damaged == 4) & (prob_c > THRESHOLD)] = 2   # T -> C
    denoised[(damaged == 1) & (prob_g > THRESHOLD)] = 3   # A -> G

    out = OUT_DIR / 'results' / f'test_denoised_{variant}.npz'
    np.savez(
        out,
        damaged=damaged.astype(np.uint8),
        clean=clean.astype(np.uint8),
        denoised=denoised.astype(np.uint8),
        prob_c=prob_c,
        prob_g=prob_g,
        lengths=lengths,
    )
    n_cor = int(((denoised != damaged) & (denoised == 2)).sum())
    n_dmg = int(((damaged == 4) & (clean == 2)).sum())
    log(f'\nCorrections (T->C): {n_cor:,}   True damage sites: {n_dmg:,}')
    log(f'Saved -> {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply trained denoiser to test set.')
    parser.add_argument('--variant', choices=['seq_only', 'evo2', 'bwa', 'udg'],
                        default='evo2', help='Model variant to run')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Seed checkpoint(s) to average over (default: [42])')
    args = parser.parse_args()
    main(args.variant, args.seeds)
