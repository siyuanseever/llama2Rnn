import torch
from typing import Tuple
import numpy as np

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, k = 1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device) / k  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def ntk_rope_freqs_cis(dim: int, end: int, theta: float = 10000.0, k = 1.0):
    theta = theta * k ** (dim / (dim - 2))
    return precompute_freqs_cis(dim, end, theta)

def get_default_real(dim: int, end: int):
    t = torch.arange(end).float()
    i = torch.arange(dim / 2)
    a = i * 0 + 1
    real = a[None] ** t[:, None]
    return real

def get_xpos_real(dim: int, end: int, lam: int = 32):
    print("xpos lam=", lam)
    t = torch.arange(end).float()
    i = torch.arange(dim / 2)
    a = (2 * i / dim + lam) / (1 + lam)
    real = a[None] ** t[:, None]
    return real

def get_xpos_param(text):
    import re
    match = re.search(r'xpos(\d+)', text)
    if match:
        lam = int(match.group(1))  # 提取匹配的数字部分
    else:
        lam = 32
    return lam

def precompute_xpos(dim: int, end: int, theta: float = 10000.0, lam = 32, k = 1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device) / k  # type: ignore
    
    i = torch.arange(dim / 2)
    a = (2 * i / dim + lam) / (1 + lam)
    real = a[None] ** t[:, None]
    
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin, real

def ntk_freqs_cis(dim: int, end: int, theta: float = 10000.0, k = 1.0, b = 0.75):
    a = np.log(k) / (dim / 2)**b
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    freqs *= (-a * torch.arange(1, dim // 2 + 1).float()**b).exp()
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    real: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)
    real = reshape_for_broadcast(real, xq_r)

    # apply rotation using real numbers
    xq_out_r = (xq_r * freqs_cos - xq_i * freqs_sin) * real
    xq_out_i = (xq_r * freqs_sin + xq_i * freqs_cos) * real
    xk_out_r = (xk_r * freqs_cos - xk_i * freqs_sin) / real 
    xk_out_i = (xk_r * freqs_sin + xk_i * freqs_cos) / real

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)


    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_sum_cis_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    real: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)
    real = reshape_for_broadcast(real, xq_r)

    # apply rotation using real numbers
    xq_out_r = (xq_r * freqs_cos + xq_i * freqs_sin) * real
    xq_out_i = (xq_r * freqs_sin + xq_i * freqs_cos) * real
    xk_out_r = (xk_r * freqs_cos + xk_i * freqs_sin) / real 
    xk_out_i = (xk_r * freqs_sin + xk_i * freqs_cos) / real

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)


    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_rotary_emb_window(
    xq: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    real: torch.Tensor,
    window: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = freqs_cos[window-1, :].view(1, 1, 1, -1)
    freqs_sin = freqs_sin[window-1, :].view(1, 1, 1, -1)
    real = real[window-1, :].view(1, 1, 1, -1)

    # apply rotation using real numbers
    xq_out_r = (xq_r * freqs_cos - xq_i * freqs_sin) * real
    xq_out_i = (xq_r * freqs_sin + xq_i * freqs_cos) * real

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq)

def apply_rotary_emb_group(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    real: torch.Tensor,
    window: int,
    group: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    extend_seq_len, dim = freqs_cos.shape
    use_seq_len = (extend_seq_len + group - 1) // group
    shift = window - (window + group - 1) // group

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    k_freqs_cos = freqs_cos[:use_seq_len, None, :].expand(
            use_seq_len, group, dim).reshape(-1, dim)[:extend_seq_len]
    k_freqs_sin = freqs_sin[:use_seq_len, None, :].expand(
            use_seq_len, group, dim).reshape(-1, dim)[:extend_seq_len]
    k_real = real[:use_seq_len, None, :].expand(
            use_seq_len, group, dim).reshape(-1, dim)[:extend_seq_len]
    q_freqs_cos = freqs_cos[shift:shift+use_seq_len, None, :].expand(
            use_seq_len, group, dim).reshape(-1, dim)[-extend_seq_len:]
    q_freqs_sin = freqs_sin[shift:shift+use_seq_len, None, :].expand(
            use_seq_len, group, dim).reshape(-1, dim)[-extend_seq_len:]
    q_real = real[shift:shift+use_seq_len, None, :].expand(
            use_seq_len, group, dim).reshape(-1, dim)[-extend_seq_len:]

    # reshape freqs_cos and freqs_sin for broadcasting
    k_freqs_cos = reshape_for_broadcast(k_freqs_cos, xk_r)
    k_freqs_sin = reshape_for_broadcast(k_freqs_sin, xk_r)
    k_real = reshape_for_broadcast(k_real, xk_r)
    q_freqs_cos = reshape_for_broadcast(q_freqs_cos, xq_r)
    q_freqs_sin = reshape_for_broadcast(q_freqs_sin, xq_r)
    q_real = reshape_for_broadcast(q_real, xq_r)

    # apply rotation using real numbers
    xq_out_r = (xq_r * q_freqs_cos - xq_i * q_freqs_sin) * q_real
    xq_out_i = (xq_r * q_freqs_sin + xq_i * q_freqs_cos) * q_real
    xk_out_r = (xk_r * k_freqs_cos - xk_i * k_freqs_sin) / k_real
    xk_out_i = (xk_r * k_freqs_sin + xk_i * k_freqs_cos) / k_real

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3) # bsz, slen, heads, ndim
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)


    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

def repeat_freqs(freqs: torch.Tensor, n_rep: int) -> torch.Tensor:
    slen, dim = freqs.size()
    repeated_freqs = freqs.unsqueeze(0).repeat((n_rep, 1, 1))
    return repeated_freqs.view(n_rep * slen, dim)

def repeat_freqs_clip(freqs: torch.Tensor, n_rep: int) -> torch.Tensor:
    slen, dim = freqs.size()
    rep_len = slen * (n_rep-1)
    repeated_freqs = (freqs[:1]).repeat((rep_len, 1))
    return torch.concat([repeated_freqs, freqs], axis=0)
