# train repo with token4096 on 8 GPUs.
# torchrun --standalone --nproc_per_node=8 train_repo.py config/train_repo_token4096.py

# wandb logging
wandb_log = True  # disabled by default
wandb_project = "llamac-repo-v1"
# data
batch_size = 16  # if gradient_accumulation_steps > 1, this is the micro-batch size
vocab_source = "llama2" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 32000 # the Llama 2 tokenizer has 32K tokens
# model
dim = 768
n_layers = 12
n_heads = 12
n_kv_heads = n_heads
multiple_of = 32
dropout = 0.2
# adamw optimizer
gradient_accumulation_steps = 4 * 8  # used to simulate larger batch sizes
learning_rate = 3e-4  # max learning rate
# system
dtype = "float32"  # float32|bfloat16|float16 2080Ti does not support bfloat16
# I/O
out_dir = f"out/repo-v1/layer{n_layers}_tok{vocab_source}{vocab_size}_dropout{dropout}"

