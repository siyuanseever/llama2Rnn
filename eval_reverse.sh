#!/bin/bash

# data
task_name="tinystories_reverse"
vocab_source="custom" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size=4096 # the Llama 2 tokenizer has 32K tokens

# eval
batch_size=4  # if gradient_accumulation_steps > 1, this is the micro-batch size
eval_iters=1000
eval_last=False
repeat_tokens=False

# model
attention_type="attention"
# attention_type="memory_attention"
# extend_methods=("ReRoPE" "selfExtendHalf" "interpolation")
# extend_methods=("AMask")
# extend_methods=("ADoubleWindow")
# extend_methods=""
extend_methods=("NTK-RoPE")
# extend_methods=("SWAHalf_selfExtendHalf" "SWAQuater_selfExtendHalf")
# extend_methods=("SWAHalf_selfExtendQuater" "SWAQuater_selfExtendQuater")
# extend_methods=("extrapolation" "interpolation" "SWA" "selfExtendHalf" "ReRoPE")
# extend_methods=("extrapolation" "interpolation" "SWA")
# extend_methods=("ntk" "radix" "PEClip" "clip")
# extend_methods=("extrapolation" "SWA")
theta=10000

key_norm=False
memseqlen=32
do_wm=False
do_memory_ffn=True
memory_norm=True
reuse_kv=True
train_orimem=False

# memory save or use
use_saved_mem=""
update_memory=True
save_memory=""
if [ "$save_memory" ]; then
    update_memory=True
fi

# I/O
# out_dir=./out/retry_reverse_custom4096_len256
# out_dir=./out/retry_reverse_custom4096_len256_theta
# out_dir=./out/retry_reverse_custom4096_len256_theta1000000
# out_dir=./out/retry_reverse_custom4096_len256_xpos1024
# out_dir=./out/retry_reverse_custom4096_len256_nope
# out_dir=./out/retry_reverse_custom4096_len256_freqsAbs
# out_dir=./out/retry_reverse_custom4096_len256_sumCis_freqsAbs
#
# out_dir=./out/retry_reverse_custom4096_len1024
# out_dir=./out/retry5_reverse_custom4096_len1024
out_dir=./out/retry5_reverse_custom4096_len1024_ConcatPE
# out_dir=./out/retry5_reverse_custom4096_len1024_xpos32
#
# out_dir=./out/retry_reverse_custom4096_len256_memory32_ffn_norm_reusekv_updatemem

mkdir -p ${out_dir}
cp $0 ${out_dir}/eval.sh

for extend_method in "${extend_methods[@]}"; do
    extend_method="ConcatPE_${extend_method}"
    log_file="${out_dir}/log2_${use_saved_mem}_update${update_memory}_${extend_method}_${key_norm}.txt"
    for ((i=8; i<=13; i++))
    do
        # if [ $i -ge 17 ]; then
        #     batch_size=1
        #     eval_iters=100
        # fi
        max_seq_len=$((2 ** i))
        echo "================== eval $max_seq_len at $log_file ================"
        date
        python3 train.py \
            --task_name=${task_name} \
            --batch_size=${batch_size} --max_seq_len=${max_seq_len} \
            --extend_method=${extend_method} \
            --key_norm=${key_norm} \
            --theta=${theta} \
            --vocab_source=${vocab_source} --vocab_size=${vocab_size} \
            --attention_type=${attention_type} --memseqlen=${memseqlen} \
            --memory_norm=${memory_norm} --do_memory_ffn=${do_memory_ffn} --do_wm=${do_wm} \
            --reuse_kv=${reuse_kv} --train_orimem=${train_orimem} \
            --dtype="float32" \
            --device="cuda" --compile=False \
            --eval_only=True --init_from="resume" --always_save_checkpoint=False \
            --eval_last=${eval_last} --eval_iters=${eval_iters} \
            --repeat_tokens=${repeat_tokens} \
            --save_memory=${save_memory} --use_saved_mem=${use_saved_mem} --update_memory=${update_memory}\
            --out_dir=${out_dir} \
            | tee -a ${log_file}
    done
done
