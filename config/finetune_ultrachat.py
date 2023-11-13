# train repo with token4096 on 8 GPUs.
# torchrun --standalone --nproc_per_node=8 train.py config/train_tinystories_token4096.py

from datetime import datetime

# data
task_name = "ultrachat"
batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
vocab_source = "llama2" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 32000 # the Llama 2 tokenizer has 32K tokens

max_seq_len = 1024
# model
init_from = "finetune"
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0
# memory
attention_type = "memory_attention"
memseqlen = 256 // 2
do_wm = False
do_memory_ffn = True
memory_norm = True
reuse_kv = True
train_orimem = True
# adamw optimizer
gradient_accumulation_steps = 4 * 4  # used to simulate larger batch sizes
# system
dtype = "float32"  # float32|bfloat16|float16 2080Ti does not support bfloat16
test_model = False
# I/O
exp_name = f"{task_name}_{vocab_source}{vocab_size}_len{max_seq_len}"
if attention_type == "memory_attention":
    exp_name += f'_memory{memseqlen}'
    if do_wm:
        exp_name += '_wm'
    if do_memory_ffn:
        exp_name += '_ffn'
    if memory_norm:
        exp_name += '_norm'
    if reuse_kv:
        exp_name += '_reusekv'
    if train_orimem:
        exp_name += '_trainmem'
exp_name += '_finetunewiki-en'

out_dir = f"out/{exp_name}"
# wandb logging
wandb_log = True  # disabled by default
wandb_project = f"llamac_{task_name}"
wandb_run_name = exp_name + ' ' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# save config
import os
import shutil
import inspect
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    current_file = inspect.getmodule(lambda: None).__file__
    file_name = os.path.basename(current_file)
    destination_path = os.path.join(out_dir, file_name)
    shutil.copy2(current_file, destination_path)
