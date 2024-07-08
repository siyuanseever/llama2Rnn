from typing import Optional
from dataclasses import dataclass

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
    theta : int = 10000
