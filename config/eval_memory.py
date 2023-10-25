# train repo with token4096 on 8 GPUs.
# torchrun --standalone --nproc_per_node=8 train.py config/train_tinystories_token4096.py


# data
batch_size = 1  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 65536
vocab_source = "custom" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 4096 # the Llama 2 tokenizer has 32K tokens
memory_attention = True
# adamw optimizer
gradient_accumulation_steps = 4 * 4  # used to simulate larger batch sizes
# system
dtype = "float32"  # float32|bfloat16|float16 2080Ti does not support bfloat16
# I/O
eval_only = True
init_from = "resume"
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
out_dir = "out/custom4096_len512_memory"
# out_dir = "out/custom4096_len1024_memory"
# out_dir = "out/custom_4096_memory_len256"
