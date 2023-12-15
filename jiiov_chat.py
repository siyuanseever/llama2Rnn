"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"
PRETOKEN_DIR = "jiiov_chat"
DATA_NAME_DIRS = [
    "ultrachat",
    "zhihu",
    "jiiov"
]
TRAIN_DATA_DIRS = ["wikipedia_en", "wiki_zh"]
TOKENIZER_DIR = "wiki"

def process_shard(args, vocab_size):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        for talk in example["conversation"]:
            query = talk["human"]
            answer = talk["assistant"]
            text = f"[INST] {query} [/INST] {answer}";
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=False, eos=True)  # encode the text, use BOS
            all_tokens.extend(tokens)
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename
    if vocab_size == 0:
        # if we're using Llama 2, just save the tokenized file in the same dir
        tokenized_filename = shard.replace(".json", ".bin")
    else:
        # save .bin files into a new tok{N} directory
        bin_dir = os.path.join(DATA_CACHE_DIR, PRETOKEN_DIR, f"tok{vocab_size}")
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(bin_dir, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by EOS=2)
    avg_seq_len = all_tokens.size / ((all_tokens == 2).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize(vocab_size):
    # iterate the shards and tokenize all of them one by one
    data_dir = os.path.join(DATA_CACHE_DIR, PRETOKEN_DIR)
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if vocab_size > 0:
        # .bin files will be saved into tok{N} directory, create it once here
        bin_dir = os.path.join(DATA_CACHE_DIR, PRETOKEN_DIR, f"tok{vocab_size}")
        os.makedirs(bin_dir, exist_ok=True)

    # process all the shards in a process pool
    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        shard_filenames = []
        for data_name in DATA_NAME_DIRS:
            if self.vocab_source == "llama2":
                # the .bin files are right along the .json files
                bin_dir = os.path.join(DATA_CACHE_DIR, data_name)
                sub_shards = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
            elif self.vocab_source == "custom":
                # the .bin files are in tok{N} directory
                bin_dir = os.path.join(DATA_CACHE_DIR, data_name, f"tok{self.vocab_size}")
                sub_shards = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
            sub_shards = sub_shards[1:] if self.split == "train" else sub_shards[:1]
            shard_filenames.extend(sub_shards)
        # train/test split. let's use only shard 0 for test split, rest train
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        while True:
            shard = rng.choice(shard_filenames)
            # open the dataset for reading but keep it on disk with memmap
            m = np.memmap(shard, dtype=np.uint16, mode="r")
            num_batches = len(m) // self.max_seq_len
            num_batches -= 1  # drop the last partial batch
            assert num_batches > 0, "this shard is way too small? investigate."
            ixs = list(range(num_batches))
            ix = rng.choice(ixs)
            start = ix * self.max_seq_len
            end = start + self.max_seq_len + 1
            # calling .astype will copy the data into a new numpy array, now in RAM
            chunk = torch.from_numpy((m[start:end]).astype(np.int64))
            x = chunk[:-1]
            y = chunk[1:]
            yield x, y

# -----------------------------------------------------------------------------
# public interface functions

def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_CACHE_DIR, TOKENIZER_DIR, f"tok{vocab_size}.model")

class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab"])
    parser.add_argument("--vocab_size", type=int, default=0, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    args = parser.parse_args()

    # depending on the stage call the appropriate function
    if args.stage == "pretokenize":
        pretokenize(vocab_size=args.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
