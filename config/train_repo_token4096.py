# train repo with token4096 on 8 GPUs.
# torchrun --standalone --nproc_per_node=8 train_repo.py config/train_repo_token4096.py

# I/O
out_dir = "out/repo/custom_4096"
# wandb logging
wandb_log = True  # disabled by default
wandb_project = "llamac-repo"
# data
batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
vocab_source = "custom" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 4096 # the Llama 2 tokenizer has 32K tokens
# adamw optimizer
gradient_accumulation_steps = 4 * 4  # used to simulate larger batch sizes
# system
dtype = "float32"  # float32|bfloat16|float16 2080Ti does not support bfloat16
