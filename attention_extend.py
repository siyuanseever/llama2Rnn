import torch
import numpy as np

def trans_mask_to_mask_inf(mask):
    mask_inf = torch.zeros_like(mask, dtype=torch.float32)
    mask_inf[~mask] = float("-inf")
    return mask_inf

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
