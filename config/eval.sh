#!/bin/bash

# data
batch_size=1  # if gradient_accumulation_steps > 1, this is the micro-batch size
vocab_source="custom" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size=4096 # the Llama 2 tokenizer has 32K tokens
memory_attention=True
memseqlen=128
do_wm=False
do_memory_ffn=False
memory_norm=False
# I/O
out_dir=out/custom4096_len256_memory128


for ((i=8; i<=15; i++))
do
    max_seq_len=$((2 ** i))
    echo "eval $max_seq_len"
    date
    python3 train.py \
        --batch_size=${batch_size} --max_seq_len=${max_seq_len} \
        --vocab_source=${vocab_source} --vocab_size=${vocab_size} \
        --memory_attention=${memory_attention} --memseqlen=${memseqlen} \
        --memory_norm=${memory_norm} --do_memory_ffn=${do_memory_ffn} --do_wm=${do_wm} \
        --dtype="float32" \
        --eval_only=True --init_from="resume" --always_save_checkpoint=False \
        --out_dir=${out_dir} \
        | tee -a ${out_dir}/log.txt
done
