# train repo with token4096 on 8 GPUs.
# torchrun --standalone --nproc_per_node=8 train.py config/train_tinystories_token4096.py

from datetime import datetime

# I/O
init_from = "finetune"
out_dir = "out/custom_4096"
# data
batch_size = 32 // 2  # if gradient_accumulation_steps > 1, this is the micro-batch size
vocab_source = "custom" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 4096 # the Llama 2 tokenizer has 32K tokens

max_seq_len = 1024

attention_type = "memory_attention"
memseqlen = 256 // 2
do_wm = False
do_memory_ffn = True
memory_norm = True
# adamw optimizer
gradient_accumulation_steps = 4 * 4 * 2  # used to simulate larger batch sizes
# system
dtype = "float32"  # float32|bfloat16|float16 2080Ti does not support bfloat16
# I/O
exp_name = f"{vocab_source}{vocab_size}_len{max_seq_len}"
if attention_type == "memory_attention":
    exp_name += f'_memory{memseqlen}'
    if do_wm:
        exp_name += '_wm'
    if do_memory_ffn:
        exp_name += '_ffn'
    if memory_norm:
        exp_name += '_norm'

exp_name += f"_{init_from}"

out_dir = f"out/{exp_name}"
# wandb logging
wandb_log = True  # disabled by default
wandb_project = "llamac"
wandb_run_name = exp_name + ' ' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
