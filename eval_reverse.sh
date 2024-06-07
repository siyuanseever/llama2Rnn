#!/bin/bash

# data
task_name="tinystories_reverse"
vocab_source="custom" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size=4096 # the Llama 2 tokenizer has 32K tokens

# eval
batch_size=32  # if gradient_accumulation_steps > 1, this is the micro-batch size
eval_iters=1000
eval_last=False
repeat_tokens=False

# model
attention_type="attention"
extend_method="selfExtend"
key_norm=False


# I/O
out_dir=./out/retry_reverse_custom4096_len256
# out_dir=./out/retry_reverse_custom4096_len1024
# out_dir=./out/retry_reverse_custom4096_len256_nope

mkdir -p ${out_dir}
cp $0 ${out_dir}/eval.sh


for ((i=8; i<=11; i++))
do
    if [ $i -ge 14 ]; then
        batch_size=8
    fi
    if [ $i -ge 17 ]; then
        batch_size=1
        eval_iters=100
    fi
    max_seq_len=$((2 ** i))
    # max_seq_len=$((32 * i))
    echo "eval $max_seq_len"
    date
    python3 train.py \
        --task_name=${task_name} \
        --batch_size=${batch_size} --max_seq_len=${max_seq_len} \
        --extend_method=${extend_method} \
        --key_norm=${key_norm} \
        --vocab_source=${vocab_source} --vocab_size=${vocab_size} \
        --attention_type=${attention_type} \
        --dtype="float32" \
        --device="cuda" --compile=False \
        --eval_only=True --init_from="resume" --always_save_checkpoint=False \
        --eval_last=${eval_last} --eval_iters=${eval_iters} \
        --repeat_tokens=${repeat_tokens} \
        --out_dir=${out_dir} \
        | tee -a ${out_dir}/log_${extend_method}_${key_norm}.txt
done
