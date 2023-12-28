import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    extend_seq_len: int = 2048
    extend_method: str = "extrapolation"
    dropout: float = 0.0
    attention_type: str = "attention"
    memory_attention: bool = False
    memseqlen: int = 128
    do_wm: bool = False
    do_memory_ffn: bool = False
    memory_norm: bool = False
    train_orimem: bool = False
    reuse_kv: bool = False
    lora: bool = False
    update_memory: bool = False
    use_saved_mem: bool = False
    key_norm: bool = False


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def l2norm(x: torch.tensor, eps: float):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, k = 1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device) / k  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

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
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)


    return xq_out.type_as(xq), xk_out.type_as(xk)

def apply_logn(
    xq_out: torch.Tensor, 
    training_length: int = 256,
    eval_only : bool = True
) -> torch.Tensor:
    bs, slen, heads, dim = xq_out.shape
    position_ids = torch.arange(slen)[None, :].repeat(bs, 1).type_as(xq_out)
    scale = ((position_ids + 1)[:, :, None, None].log() / np.log(training_length))
    if eval_only:
        scale = scale.clip(1.0)
    xq_out = xq_out * scale.to(xq_out.device)

    return xq_out

def apply_rotary_emb_window(
    xq: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
    window: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = freqs_cos[window-1, :].view(1, 1, 1, -1)
    freqs_sin = freqs_sin[window-1, :].view(1, 1, 1, -1)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq)

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

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.args = args

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        mask0 = torch.ones(1, 1, args.extend_seq_len, args.extend_seq_len, dtype=torch.bool).tril(diagonal=0)
        if "BCA" in args.extend_method:
            mask = mask0 ^ torch.ones(1, 1, args.extend_seq_len, args.extend_seq_len, dtype=torch.bool).tril(diagonal=-args.max_seq_len)
            self.register_buffer("mask", mask)
        else:
            mask = torch.full((1, 1, args.extend_seq_len, args.extend_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask_inf", mask)
            self.register_buffer("mask", mask0)
        if "ReRoPE" in args.extend_method:
            self.window = args.max_seq_len // 2
            rectified_mask = mask0 ^ torch.ones(1, 1, args.extend_seq_len, args.extend_seq_len, dtype=torch.bool).tril(diagonal=-self.window)
            self.register_buffer("rectified_mask", rectified_mask)
        if args.key_norm:
            print("use key norm")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq_tmp = xq
        xk_tmp = xk

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        logn = True if 'logn' in self.args.extend_method else False
        eval_only = False if 'train' in self.args.extend_method else True
        xq = apply_logn(xq, self.args.max_seq_len, eval_only) if logn else xq
        xk = l2norm(xk, self.args.norm_eps) if self.args.key_norm else xk

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if 'ReRoPE' in self.args.extend_method:
            xq2 = apply_rotary_emb_window(
                xq_tmp, freqs_cos, freqs_sin,
                window=self.window
            )
            xq2 = apply_logn(xq2, self.args.max_seq_len, eval_only) if logn else xq2
            xk2 = l2norm(xk_tmp, self.args.norm_eps) if self.args.key_norm else xk_tmp
            xk2 = repeat_kv(xk2, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
            xq2 = xq2.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            xk2 = xk2.transpose(1, 2)

            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores2 = torch.matmul(xq2, xk2.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores = torch.where(self.rectified_mask, scores, scores2)
            scores = scores + self.mask_inf[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        else:
            # flash implementation
            if self.flash:
                output = torch.nn.functional.scaled_dot_product_attention(
                    xq, xk, xv, 
                    attn_mask=self.mask[:, :, :seqlen, :seqlen], 
                    dropout_p=self.dropout if self.training else 0.0
                )
            else:
                # manual implementation
                scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
                scores = scores + self.mask_inf[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                scores = self.attn_dropout(scores)
                output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class MemoryAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.args = args

        # memory
        self.memseqlen = args.memseqlen
        self.do_wm = args.do_wm
        if self.do_wm:
            self.wm = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.memory_norm = args.memory_norm
        if self.memory_norm:
            self.memory_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.do_memory_ffn = args.do_memory_ffn
        if self.do_memory_ffn:
            self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )
        if args.reuse_kv:
            self.wqm, self.wkm, self.wvm = self.wq, self.wk, self.wv
        else:
            self.wqm = nn.Linear(args.n_heads * self.head_dim, args.n_heads * self.head_dim, bias=False)
            self.wkm = nn.Linear(args.n_heads * self.head_dim, args.n_heads * self.head_dim, bias=False)
            self.wvm = nn.Linear(args.n_heads * self.head_dim, args.n_heads * self.head_dim, bias=False)
        self.dim = args.dim
        self.update_memory = args.update_memory
        self.use_saved_mem = args.use_saved_mem
        if args.train_orimem:
            self.origin_mem = nn.Parameter(torch.zeros([1, self.memseqlen, self.dim]))
        elif not args.use_saved_mem:
            self.register_buffer("origin_mem", torch.zeros([1, self.memseqlen, self.dim]))
        if self.use_saved_mem or self.update_memory:
            self.register_buffer("memory", torch.zeros([1, self.memseqlen, self.dim]))

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        if args.key_norm:
            print("use key norm")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        outputs = []
        if self.use_saved_mem:
            om = self.memory.expand(bsz, self.memseqlen, self.dim)
        else:
            om = self.origin_mem.expand(bsz, self.memseqlen, self.dim)
        for idx in range(0, seqlen, self.memseqlen):
            subx = x[:, idx:idx+self.memseqlen]
            _, subseqlen, _ = subx.shape

            # QKV
            xq, xk, xv = self.wq(subx), self.wk(subx), self.wv(subx)
            xq = xq.view(bsz, subseqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, subseqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, subseqlen, self.n_local_kv_heads, self.head_dim)

            # Memory KV
            if self.do_wm:
                om = self.wm(om)
            if self.do_memory_ffn:
                om = om + self.feed_forward.forward(self.ffn_norm(om))
            if self.memory_norm:
                om = self.memory_norm(om)
            mq, mk, mv = self.wqm(om), self.wkm(om), self.wvm(om)
            mq = mk.view(bsz, self.memseqlen, self.n_local_kv_heads, self.head_dim)
            mk = mk.view(bsz, self.memseqlen, self.n_local_kv_heads, self.head_dim)
            mv = mv.view(bsz, self.memseqlen, self.n_local_kv_heads, self.head_dim)
            xq = torch.concat([mq, xq], dim=1)
            xk = torch.concat([mk, xk], dim=1)
            xv = torch.concat([mv, xv], dim=1)

            # RoPE relative positional embeddings
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos[:self.memseqlen+subseqlen], freqs_sin[:self.memseqlen+subseqlen])
            logn = True if 'logn' in self.args.extend_method else False
            eval_only = False if 'train' in self.args.extend_method else True
            xq = apply_logn(xq, self.args.max_seq_len, eval_only) if logn else xq
            xk = l2norm(xk, self.args.norm_eps) if self.args.key_norm else xk

            # grouped multiquery attention: expand out keys and values
            xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
            xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

            # make heads into a batch dimension
            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)

            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)

            # restore time as batch dimension and concat heads
            om = output.transpose(1, 2).contiguous().view(bsz, self.memseqlen+subseqlen, self.dim)[:, self.memseqlen:]
            outputs.append(om)
            if self.update_memory:
                self.memory.data.copy_(om[:1])

        output = torch.concat(outputs, dim=1)
        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class ChunkLSTM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.memseqlen = args.memseqlen
        self.s_dim = args.dim // args.memseqlen
        self.dim = args.dim

        self.h = nn.Parameter(torch.zeros([1, args.memseqlen, args.dim]))
        self.c = nn.Parameter(torch.zeros([1, args.memseqlen, args.dim]))
        self.wii = nn.Linear(args.dim, self.dim, bias=False)
        self.whi = nn.Linear(args.dim, self.dim, bias=False)
        self.wif = nn.Linear(args.dim, self.dim, bias=False)
        self.whf = nn.Linear(args.dim, self.dim, bias=False)
        self.wig = nn.Linear(args.dim, self.dim, bias=False)
        self.whg = nn.Linear(args.dim, self.dim, bias=False)
        self.wio = nn.Linear(args.dim, self.dim, bias=False)
        self.who = nn.Linear(args.dim, self.dim, bias=False)
        self.wo  = nn.Linear(args.dim, args.dim, bias=False)

        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.reduce_type = "train"
        assert self.reduce_type in ["old", "mask", "train"]

        if self.reduce_type == "old":
            mask = torch.full((1, args.memseqlen, args.memseqlen), 1.0/args.memseqlen)
            mask = torch.tril(mask, diagonal=0)
            self.register_buffer("mask", mask)
        elif self.reduce_type == "fix":
            mask = torch.tril(torch.ones(1, args.memseqlen, args.memseqlen))
            mask = mask / torch.sum(mask, -1, keepdim=True)
            self.register_buffer("mask", mask)
        else:
            reduce = torch.tril(torch.ones(1, args.memseqlen, args.memseqlen))
            reduce = reduce / torch.sum(reduce, -1, keepdim=True)
            self.reduce = nn.Parameter(reduce)
            mask = torch.tril(torch.ones(1, args.memseqlen, args.memseqlen))
            self.register_buffer("mask", mask)
        
    def forward(
        self,
        seq_x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = seq_x.shape        
        outputs = []
        h = self.h.expand(bsz, self.memseqlen, self.dim)
        c = self.c.expand(bsz, self.memseqlen, self.dim)
        for idx in range(0, seqlen, self.memseqlen):
            x = seq_x[:, idx:idx+self.memseqlen]

            i = F.sigmoid(self.wii(x) + self.whi(h))
            f = F.sigmoid(self.wif(x) + self.whf(h))

            g = self.wig(x) + self.whg(h)
            if self.reduce_type == "train":
                g = torch.matmul(self.reduce * self.mask, g) # with local chunk fuse
            else:
                g = torch.matmul(self.mask, g) # with local chunk fuse
            g = F.tanh(g)

            o = F.sigmoid(self.wio(x) + self.who(h))
            c = f * c + i * g
            h = o * F.tanh(c)
    
            outputs.append(h)

        output = torch.concat(outputs, dim=1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        if args.attention_type == "memory_attention" or args.memory_attention:
            self.attention = MemoryAttention(args)
        elif args.attention_type == "LSTM":
            self.attention = ChunkLSTM(args)
        else:
            self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        k = self.params.extend_seq_len / self.params.max_seq_len
        # various sequence length extrapolation
        if "extrapolation" in self.params.extend_method:
            freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.extend_seq_len)
        elif "interpolation" in self.params.extend_method:
            freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.extend_seq_len, k = k)
        elif "radix" in self.params.extend_method:
            freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.extend_seq_len, theta = 10000.0 * k)
        elif "ntk" in self.params.extend_method:
            freqs_cos, freqs_sin = ntk_freqs_cis(self.params.dim // self.params.n_heads, self.params.extend_seq_len, k=k)
        elif self.params.extend_method == "rotate":
            freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
            if k > 1:
                freqs_cos = repeat_freqs(freqs_cos, int(k))
                freqs_sin = repeat_freqs(freqs_sin, int(k))
        elif "PEClip" in self.params.extend_method:
            freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
            if k > 1:
                freqs_cos = repeat_freqs_clip(freqs_cos, int(k))
                freqs_sin = repeat_freqs_clip(freqs_sin, int(k))
        else:
            freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.extend_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight') or pn.endswith('wm.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None
        self.last_acc = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _repeat_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        # 获取tokens的最后self.params.max_seq_len部分
        last_tokens = tokens[:, -self.params.max_seq_len:]
        # 重复last_tokens直到它和原始tokens的长度一样
        repeated_tokens = last_tokens.repeat(1, tokens.size(1) // last_tokens.size(1))
        return repeated_tokens

    def forward(self, 
        tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, 
        eval_last: bool = False, repeat_tokens: bool = False,
    ) -> torch.Tensor:
        if self.params.extend_method == "clip":
            tokens = tokens[:, -self.params.max_seq_len:]
            targets = targets[:, -self.params.max_seq_len:]
        if repeat_tokens:
            tokens = self._repeat_tokens(tokens)
            targets = self._repeat_tokens(targets)

        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h[:, -self.params.max_seq_len:, :])
            targets = targets[:, -self.params.max_seq_len:]
            if eval_last:
                logits = logits[:, [-1], :]
                targets = targets[:, [-1]]
            self.last_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1),
                ignore_index=-1
            )

            _, predicts = torch.max(logits, -1)
            ignore_mask = targets != -1
            total_samples = ignore_mask.sum()
            self.last_acc = ((predicts == targets) & ignore_mask).sum().float() / total_samples.float()
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
