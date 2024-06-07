import torch
from torch import nn
import torch.nn.functional as F
import math

from modelargs import ModelArgs
from layers import RMSNorm, FeedForward, l2norm
import attention_extend
from attention_extend import apply_logn
from position_embedding import apply_rotary_emb, apply_rotary_emb_window, apply_rotary_emb_group, repeat_kv


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
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
        self.layer_id = layer_id

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        print(f'extend_seq_len: {args.extend_seq_len}, max_seq_len: {args.max_seq_len}')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        mask0 = torch.ones(1, 1, args.extend_seq_len, args.extend_seq_len, dtype=torch.bool).tril(diagonal=0)
        if "SWA" in args.extend_method or "BCA" in args.extend_method:
            self.window = args.max_seq_len
            if "SWAlayer04" in args.extend_method and layer_id == 5:
                self.window = args.extend_seq_len
            elif "SWAlayer15" in args.extend_method and layer_id == 0:
                self.window = args.extend_seq_len
            elif "SWAlayer03" in args.extend_method and layer_id in [4, 5]:
                self.window = args.extend_seq_len
            elif "SWAlayer02" in args.extend_method and layer_id in [3, 4, 5]:
                self.window = args.extend_seq_len
            elif "SWAlayer01" in args.extend_method and layer_id in [2, 3, 4, 5]:
                self.window = args.extend_seq_len
            elif "SWAlayer00" in args.extend_method and layer_id in [1, 2, 3, 4, 5]:
                self.window = args.extend_seq_len
            elif "NAlayer5" in args.extend_method and layer_id == 5:
                self.window = 1
            elif "NAlayer4" in args.extend_method and layer_id == 4:
                self.window = 1
            elif "NAlayer3" in args.extend_method and layer_id == 3:
                self.window = 1
            elif "NAlayer2" in args.extend_method and layer_id == 2:
                self.window = 1
            elif "NAlayer1" in args.extend_method and layer_id == 1:
                self.window = 1
            elif "NAlayer0" in args.extend_method and layer_id == 0:
                self.window = 1
            mask = mask0 ^ torch.ones(1, 1, args.extend_seq_len, args.extend_seq_len, dtype=torch.bool).tril(diagonal=-self.window)
            print(f"window: {self.window}, layer_id: {layer_id}")
        else:
            mask = mask0
        self.register_buffer("mask", mask)
        self.register_buffer("mask_inf", attention_extend.trans_mask_to_mask_inf(mask))

        if "ReRoPE" in args.extend_method:
            self.window = args.max_seq_len
            rectified_mask = mask0 ^ torch.ones(1, 1, args.extend_seq_len, args.extend_seq_len, dtype=torch.bool).tril(diagonal=-self.window)
            self.register_buffer("rectified_mask", rectified_mask)
            print(f"window: {self.window}")
        elif "selfExtend" in args.extend_method:
            self.window = args.max_seq_len - 1
            if 'layer5group' in self.args.extend_method and self.layer_id == 5:
                self.window = (args.max_seq_len+1) // 2
            elif 'layer4group' in self.args.extend_method and self.layer_id == 4:
                self.window = (args.max_seq_len+1) // 2
            elif 'layer3group' in self.args.extend_method and self.layer_id == 3:
                self.window = (args.max_seq_len+1) // 2
            elif 'layer1group' in self.args.extend_method and self.layer_id == 1:
                self.window = (args.max_seq_len+1) // 2
            elif 'layer0group' in self.args.extend_method and self.layer_id == 0:
                self.window = (args.max_seq_len+1) // 2
            elif 'layer5allgroup' in self.args.extend_method and self.layer_id == 5:
                self.window = 0
            elif 'layer3allgroup' in self.args.extend_method and self.layer_id == 3:
                self.window = 0
            elif 'layer0allgroup' in self.args.extend_method and self.layer_id == 0:
                self.window = 0
            left_size = args.max_seq_len - self.window
            self.group = (args.extend_seq_len - self.window + left_size -1) // left_size
            rectified_mask = mask0 ^ torch.ones(1, 1, args.extend_seq_len, args.extend_seq_len, dtype=torch.bool).tril(diagonal=-self.window)
            self.register_buffer("rectified_mask", rectified_mask)
            print(f"window: {self.window}, group: {self.group}")
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

        if 'ReRoPE' in self.args.extend_method or 'selfExtend' in self.args.extend_method:
            xq_tmp = xq
            xk_tmp = xk

        # RoPE relative positional embeddings
        if 'nope' in self.args.extend_method:
            pass
        elif 'layeradapt' in self.args.extend_method and self.layer_id == 5:
            pass
        elif 'layer4adapt' in self.args.extend_method and self.layer_id == 4:
            pass
        elif 'layer3adapt' in self.args.extend_method and self.layer_id == 3:
            pass
        elif 'layer2adapt' in self.args.extend_method and self.layer_id == 2:
            pass
        elif 'layer1adapt' in self.args.extend_method and self.layer_id == 1:
            pass
        elif 'layer0adapt' in self.args.extend_method and self.layer_id == 0:
            pass
        else:
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
        elif 'selfExtend' in self.args.extend_method:
            xq2, xk2 = apply_rotary_emb_group(
                xq_tmp, xk_tmp, freqs_cos, freqs_sin,
                window=self.window, group=self.group
            )
            xq2 = apply_logn(xq2, self.args.max_seq_len, eval_only) if logn else xq2
            xk2 = l2norm(xk2, self.args.norm_eps) if self.args.key_norm else xk2
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
            # if self.flash:
            if False:
                output = torch.nn.functional.scaled_dot_product_attention(
                    xq, xk, xv, 
                    attn_mask=self.mask[:, :, :seqlen, :seqlen], 
                    dropout_p=self.dropout if self.training else 0.0
                )
                scores = None
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
        else:
            self.register_buffer("origin_mem", torch.zeros([1, self.memseqlen, self.dim]))
        if self.use_saved_mem or self.update_memory:
            self.register_buffer("memory", torch.zeros([32, self.memseqlen, self.dim]))

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            assert False
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
        if self.use_saved_mem or self.update_memory:
            mbsz = self.memory.shape[0]
            if mbsz == 1:
                om = self.memory.expand(bsz, self.memseqlen, self.dim)
            else:
                om = self.memory.clone().detach()
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
            # self.memory.data.copy_(om.clone().detach())
            if 'emamem' in self.args.extend_method:
                self.memory = self.memory * 0.6 + om.clone().detach() * 0.4
            else:
                self.memory = om.clone().detach()

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
