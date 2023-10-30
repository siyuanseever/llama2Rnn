# train repo with token4096 on 8 GPUs.
# torchrun --standalone --nproc_per_node=8 train.py config/train_tinystories_token4096.py

from datetime import datetime

# I/O
out_dir = "out/custom_4096"
# data
batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
vocab_source = "custom" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 4096 # the Llama 2 tokenizer has 32K tokens

max_seq_len = 256

# init_from = "resume"
attention_type = "LSTM"
memseqlen = 32
# adamw optimizer
gradient_accumulation_steps = 4 * 4 # used to simulate larger batch sizes
# system
dtype = "float32"  # float32|bfloat16|float16 2080Ti does not support bfloat16
# I/O
exp_name = f"{vocab_source}{vocab_size}_len{max_seq_len}"
exp_name += f'_{attention_type}{memseqlen}_trainreduce'

out_dir = f"out/{exp_name}"


# wandb logging
wandb_log = True  # disabled by default
wandb_project = "llamac"
wandb_run_name = exp_name + ' ' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
