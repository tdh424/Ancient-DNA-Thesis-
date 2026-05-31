"""5.1_evo2_refs.py — Add Evo2 per-position probability references to the NPZ datasets."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = Path('data')
SPLITS          = ['train', 'val', 'test']
EVO2_MODEL      = 'evo2_7b'
DEFAULT_BATCH   = 512        # bf16 fits comfortably on a 32 GB GPU
DTYPE_MAP       = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
# ─────────────────────────────────────────────────────────────────────────────

BASE_TO_IDX = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
IDX_TO_BASE = {0: 'N', 1: 'A', 2: 'C', 3: 'G', 4: 'T', 5: 'N'}
COMPLEMENT  = str.maketrans('ACGTNacgtn', 'TGCANtgcan')


def decode(arr):
    return ''.join(IDX_TO_BASE.get(int(b), 'N') for b in arr)


def rev_comp(seq):
    return seq.translate(COMPLEMENT)[::-1]


def load_evo2(model_name, device):
    # PyTorch 2.4+ defaults torch.load to weights_only=True; force False to
    # load the Evo2 checkpoint's custom unpicklers.
    import torch as _torch
    _orig_load = _torch.load
    def _patched_load(*a, **kw):
        kw['weights_only'] = False     # FORCE override, not setdefault
        return _orig_load(*a, **kw)
    _torch.load = _patched_load

    from evo2 import Evo2
    print(f'Loading {model_name}...', flush=True)
    evo2_obj = Evo2(model_name)
    evo2_obj.model.eval()           # NOTE: .model.eval(), not .eval()

    # Restore default behaviour after loading
    _torch.load = _orig_load
    return evo2_obj


def get_vocab(evo2_obj):
    """Extract {base_char: token_index} from the Evo2 tokenizer."""
    tok = evo2_obj.tokenizer
    for attr in ('vocabulary', 'token_to_id'):
        try:
            v = getattr(tok, attr)
            if callable(v):
                return {b: v(b) for b in 'ACGTNacgtn' if v(b) is not None}
            if isinstance(v, dict):
                return v
        except (AttributeError, NotImplementedError):
            continue
    vocab = {}
    for base in 'ACGTacgtN':
        try:
            result = tok.tokenize(base)
            vocab[base] = int(result[0] if hasattr(result, '__getitem__') else result)
        except Exception:
            pass
    return vocab


def build_index_map(vocab):
    """Map Evo2 vocab indices → our 6-class encoding."""
    idx_map = {}
    for char, evo_i in vocab.items():
        our_i = BASE_TO_IDX.get(char.upper())
        if our_i is not None:
            idx_map[evo_i] = our_i
    return idx_map


def tokenize_batch(tok, sequences, device):
    """Tokenize and pad a list of sequences in one go."""
    ids_list = [torch.tensor(tok.tokenize(seq), dtype=torch.long) for seq in sequences]
    max_len = max(t.shape[0] for t in ids_list)
    padded  = torch.zeros(len(ids_list), max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(ids_list):
        padded[i, :ids.shape[0]] = ids.to(device, non_blocking=True)
    return padded


@torch.no_grad()
def forward_pass(evo2_obj, sequences, device, dtype):
    """Run Evo2 on a list of sequences → (N, L, vocab_size) logits in fp32."""
    padded = tokenize_batch(evo2_obj.tokenizer, sequences, device)
    with torch.autocast(device_type='cuda', dtype=dtype, enabled=(dtype != torch.float32)):
        out = evo2_obj.model(padded)
    logits = out[0] if isinstance(out, tuple) else out
    return logits.float()   # cast back to fp32 for stable softmax + downstream math


def logits_to_probs_gpu(logits, index_map_t, L):
    """
    Convert Evo2 logits → (N, L, 6) probabilities (still on GPU; one CPU copy at end).
    index_map_t: int64 tensor of shape (M,) where M ≤ V_evo, mapping the *first M*
                 Evo2 vocab indices to one of our 6 classes (-1 for unmapped).
                 Evo2 vocab indices beyond M (special tokens etc.) are ignored.
    """
    probs = torch.softmax(logits[:, :L], dim=-1)             # (N, L, V_evo_full)
    N     = probs.shape[0]
    M     = index_map_t.shape[0]                              # mapped-vocab size
    out   = torch.zeros(N, L, 6, device=probs.device, dtype=torch.float32)
    valid = index_map_t >= 0                                  # (M,)
    if valid.any():
        idx  = index_map_t.clamp(min=0)                       # (M,)
        cols = torch.where(valid, idx, torch.zeros_like(idx)) # (M,)
        # scatter-add only the first M Evo2 vocab probs; indices beyond M
        # don't correspond to any of our 6 nucleotide classes.
        probs_m = probs[..., :M]                              # (N, L, M)
        out.index_add_(2, cols, probs_m * valid.float().unsqueeze(0).unsqueeze(0))
    out = out / out.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return out


def compute_soft_refs(sequences, L, evo2_obj, index_map, device, batch_size, dtype, do_rc):
    """
    Forward (and optional RC) passes → geometric mean → (N, L, 6) float32.

    Combines forward and RC sequences into a single double-sized batch per step
    so each iteration is one tokenize + one model forward instead of two.
    """
    N = len(sequences)
    out = np.zeros((N, L, 6), dtype=np.float32)

    # Build mapping tensor once (Evo2 vocab idx → our 6-class idx, -1 if unmapped)
    max_evo_idx = max(index_map.keys()) + 1
    idx_map_t   = torch.full((max_evo_idx,), -1, dtype=torch.int64, device=device)
    for evo_i, our_i in index_map.items():
        idx_map_t[evo_i] = our_i

    for i in range(0, N, batch_size):
        batch  = sequences[i:i + batch_size]
        nb     = len(batch)

        if do_rc:
            # Stack forward + reverse-complement into one big batch (one fwd call)
            stacked = batch + [rev_comp(s) for s in batch]
            logits  = forward_pass(evo2_obj, stacked, device, dtype)
            probs_f = logits_to_probs_gpu(logits[:nb],         idx_map_t, L)
            probs_r = logits_to_probs_gpu(
                torch.flip(logits[nb:, :L], dims=[1]), idx_map_t, L)
            geo  = torch.sqrt(probs_f * probs_r + 1e-12)
            geo  = geo / geo.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        else:
            logits = forward_pass(evo2_obj, batch, device, dtype)
            geo    = logits_to_probs_gpu(logits, idx_map_t, L)

        out[i:i + nb] = geo.cpu().numpy()

        if (i // batch_size) % 10 == 0:
            print(f'  {i:>7,}/{N:,}', flush=True)

    return out.astype(np.float32)


def process_split(name, evo2_obj, index_map, device, batch_size, dtype, do_rc):
    path = DATA_DIR / f'{name}.npz'
    d = np.load(path)
    if 'ref_evo2' in d:
        print(f'  {name}: ref_evo2 already present — skipping.')
        return

    N = len(d['damaged'])
    L = d['damaged'].shape[1]
    print(f'  {name}: computing Evo2 refs for {N:,} reads (length {L})...')

    sequences = [decode(row) for row in d['damaged']]
    soft_refs = compute_soft_refs(sequences, L, evo2_obj, index_map, device,
                                   batch_size, dtype, do_rc)

    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **dict(d), ref_evo2=soft_refs)
    shutil.move(str(tmp), str(path))
    print(f'  {name}: saved. ref_evo2 shape = {soft_refs.shape}')


def main():
    parser = argparse.ArgumentParser(description='Pre-compute Evo2 soft references.')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH,
                        help=f'inference batch size (default {DEFAULT_BATCH})')
    parser.add_argument('--dtype', choices=list(DTYPE_MAP.keys()), default='bf16',
                        help='inference precision (bf16 default ≈ 2× faster than fp32)')
    parser.add_argument('--no-rc', action='store_true',
                        help='skip reverse-complement pass (~2× faster, slight quality hit)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device     : {device}', flush=True)
    print(f'Batch size : {args.batch_size}')
    print(f'Dtype      : {args.dtype}')
    print(f'RC pass    : {"NO (faster)" if args.no_rc else "yes"}')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark         = True

    evo2_obj  = load_evo2(EVO2_MODEL, device)
    vocab     = get_vocab(evo2_obj)
    index_map = build_index_map(vocab)
    print(f'Vocab size : {len(vocab)} tokens mapped to 6-class encoding.')

    dtype = DTYPE_MAP[args.dtype]
    for name in SPLITS:
        process_split(name, evo2_obj, index_map, device,
                      args.batch_size, dtype, do_rc=not args.no_rc)

    print('\nDone. Run next:')
    print('  python Code/6_train_denoiser.py --variant evo2')


if __name__ == '__main__':
    main()
