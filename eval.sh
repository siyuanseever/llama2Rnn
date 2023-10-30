#!/bin/bash

# data
vocab_source="custom" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size=4096 # the Llama 2 tokenizer has 32K tokens
# others
dtype="float32"
eval_only=True
init_from="resume"
always_save_checkpoint=False

# batch_size=1  # if gradient_accumulation_steps > 1, this is the micro-batch size
# eval_iters=100
# eval_last=False
batch_size=32  # if gradient_accumulation_steps > 1, this is the micro-batch size
eval_iters=1000
eval_last=True

# model
# attention_type="attention"
attention_type="memory_attention"
memseqlen=32
do_wm=False
do_memory_ffn=True
memory_norm=True
# I/O
out_dir=./out/custom4096_len256_memory32_ffn_norm_finetune

mkdir -p ${out_dir}
cp $0 ${out_dir}/eval.sh


for ((i=8; i<=16; i++))
do
    max_seq_len=$((2 ** i))
    echo "eval $max_seq_len"
    date
    python3 train.py \
        --batch_size=${batch_size} --max_seq_len=${max_seq_len} \
        --vocab_source=${vocab_source} --vocab_size=${vocab_size} \
        --attention_type=${attention_type} --memseqlen=${memseqlen} \
        --memory_norm=${memory_norm} --do_memory_ffn=${do_memory_ffn} --do_wm=${do_wm} \
        --dtype="float32" \
        --device="cuda" --compile=False \
        --eval_only=True --init_from="resume" --always_save_checkpoint=False \
        --eval_last=${eval_last} --eval_iters=${eval_iters}\
        --out_dir=${out_dir} \
        | tee -a ${out_dir}/log.txt
done
