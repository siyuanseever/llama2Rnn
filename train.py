"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from tqdm import tqdm

import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
eval_last = False
repeat_tokens = False
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "llamac"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data
tasks = []
task_name = "tinystories"
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 256
extend_method = "extrapolation"
vocab_source = "llama2" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 32000 # the Llama 2 tokenizer has 32K tokens
# model
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0
# extrapolation
key_norm = False
# memory
attention_type = "attention"
memseqlen = 128
do_wm = False
do_memory_ffn = False
memory_norm = False
train_orimem = False
reuse_kv = False
save_memory = ""
update_memory = False
use_saved_mem = ""
# adamw optimizer
gradient_accumulation_steps = 4  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for
# system
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

dtype = "float32"  # float32|bfloat16|float16
compile = True  # use PyTorch 2.0 to compile the model to be faster
# test_model
test_model = False
# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------
# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    extend_seq_len=max_seq_len,
    extend_method=extend_method,
    dropout=dropout,
    attention_type=attention_type,
    memseqlen=memseqlen,
    do_wm=do_wm,
    do_memory_ffn=do_memory_ffn,
    memory_norm=memory_norm,
    train_orimem=train_orimem,
    reuse_kv=reuse_kv,
    update_memory=update_memory,
    use_saved_mem=bool(use_saved_mem),
    key_norm=key_norm,
)  # start with model_args from command line

# validating checks
assert vocab_source in ["llama2", "custom"]
assert vocab_source == "custom" or vocab_size == 32000, "The vocab from Meta has 32K tokens"

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# task-specific setup
num_workers = os.cpu_count() // ddp_world_size - 1
num_workers = 0
print(f'task num workers = {num_workers}')
task_args = dict(
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    vocab_size=vocab_size,
    vocab_source=vocab_source,
    device=device,
    num_workers = num_workers,
)
if task_name == 'tinystories':
    from tinystories import Task
elif task_name == 'tinystories_infinity':
    from tinystories_infinity import Task
elif task_name == 'tinystories_order':
    from tinystories_order import Task
elif task_name == 'tinystories_repeat':
    from tinystories_repeat import Task
elif task_name == 'tinystories_reverse':
    from tinystories_reverse import Task
elif task_name == 'tinystories_reverse_infinity':
    from tinystories_reverse_infinity import Task
elif task_name == 'ultrachat':
    from ultrachat import Task
elif task_name == 'wikipedia_en':
    from wikipedia_en import Task
elif task_name == 'wiki_zh':
    from wiki_zh import Task
elif task_name == 'wiki':
    from wiki import Task
elif task_name == 'zhihu':
    from zhihu import Task
elif task_name == 'jiiov':
    from jiiov import Task
elif task_name.startswith('all'):
    from datatask import Task
    task_args["tasks"] = tasks
elif task_name.startswith('ds_'):
    from dataset import Task
    tasks = task_name[len('ds_'):].split('_')
    task_args["tasks"] = tasks
elif task_name.startswith('dg_'):
    from data_generator import Task
    tasks = task_name[len('dg_'):].split('_')
    task_args["tasks"] = tasks
iter_batches = partial(Task.iter_batches, **task_args)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
else:
    print(f"{init_from}ing training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, f"{use_saved_mem}.pt") if bool(use_saved_mem) else \
        os.path.join(out_dir, "ckpt.pt")
    print(f'load model from {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    state_dict = {k: v for k, v in state_dict.items() if 'attention.mask' not in k}
    # if memory not exits, use origin_mem to initialize
    if bool(use_saved_mem):
        for name, buffer in model.named_buffers():
            if "attention.memory" not in name:
                continue
            if name not in state_dict.keys():
                print(name, buffer.size(), 'not exits, use origin_mem to initialize')
                origin_mem = state_dict[name.replace('memory', 'origin_mem')] 
                state_dict[name] = origin_mem.expand(1, memseqlen, dim)
            else:
                print(name, buffer.size(), 'exits, use mmeory batch_idx=1 to initialize')
                state_dict[name] = (state_dict[name][:1]).expand(1, memseqlen, dim)
    if init_from == "resume":
        model.load_state_dict(state_dict, strict=False)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    elif init_from == "finetune":
        model.load_state_dict(state_dict, strict=False)
    elif "freeze" in init_from:
        model.load_state_dict(state_dict, strict=False)
        ft_params = [
            "attention.wm",
            "attention.ffn_norm", "attention.feed_forward",
            "attention.memory_norm", 
            "attention.wqm", "attention.wkm", "attention.wvm", 
        ]
        if "attention" in init_from:
            ft_params.extend(["attention"]) 
        for name, param in model.named_parameters():
            param.requires_grad = False
            for np in ft_params:
                if np in name:
                    param.requires_grad = True
        for name, param in model.named_parameters():
            print(name, param.size(), param.requires_grad)
    else:
        assert False, init_from
if test_model:
    exit(0)
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
    # construction time since NCCL does not support `ComplexFloat`
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {"train_loss": 0.0, "val_loss": 0.0, "train_acc": 0.0, "val_acc": 0.0}
    model.eval()
    splits = ["test"] if eval_only else ["train", "val"]
    for split in splits:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        acces = torch.zeros(eval_iters)  # keep on CPU
        for k in tqdm(range(eval_iters)):
            X, Y = next(batch_iter)
            with ctx:
                _ = model(X, Y, eval_last=eval_last, repeat_tokens=repeat_tokens)
                loss = raw_model.last_loss
                acc = raw_model.last_acc

                if bool(save_memory):
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving memory checkpoint to {out_dir}, loss {loss}, acc {acc}")
                    torch.save(checkpoint, os.path.join(out_dir, f"{save_memory}.pt"))
                    return out
            losses[k] = loss.item()
            acces[k] = acc.item()
        out[split+"_loss"] = losses.mean()
        out[split+"_acc"] = acces.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
if not eval_only:
    train_batch_iter = iter_batches(split="train")
    X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f" task {task_name} extend_method {extend_method}"
            f" step {iter_num} eval_last {eval_last} max_seq_len {max_seq_len}"
            f" repeat_tokens {repeat_tokens}"
            # f"use_saved_mem {use_saved_mem}: update_memory {update_memory} "
            f" train loss {losses['train_loss']:.4f}, val loss {losses['val_loss']:.4f}"
            f" train acc {losses['train_acc']:.4f}, val acc {losses['val_acc']:.4f}"
        )
        if eval_only:
            print(
                f" test acc {losses['test_acc']:.4f}, test loss {losses['test_loss']:.4f}"              )
        if wandb_log:
            try:
                wandb.log(
                    {
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "loss/train": losses["train_loss"],
                        "loss/val": losses["val_loss"],
                        "acc/train": losses["train_acc"],
                        "acc/val": losses["val_acc"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    }, step = iter_num
                )
            except Exception as e:
                print(f"logging to wandb failed: {e}")
        if (losses["val_loss"] < best_val_loss and not eval_only) or always_save_checkpoint:
            best_val_loss = losses["val_loss"]
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    dt_data = 0
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            logits = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        td = time.time()
        X, Y = next(train_batch_iter)
        dt_data += time.time() - td
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | data {dt_data*1000:.2f}ms"
            f" | mfu {running_mfu*100:.2f}% | mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
