import re
import glob

def parse(file):
    print(file)
    with open(file, 'r') as file:
        log_content = file.read()

    # 使用正则表达式匹配 max_seq_len 和 test loss
    max_seq_len_pattern = re.compile(r'max_seq_len (\d+)')
    test_loss_pattern = re.compile(r'test loss ([\d.]+)')

    # 找到所有匹配的 max_seq_len 和 test loss
    max_seq_lens = max_seq_len_pattern.findall(log_content)
    test_losses = test_loss_pattern.findall(log_content)

    # 打印结果
    for max_seq_len, test_loss in zip(max_seq_lens, test_losses):
        print(f"{max_seq_len:5}: {test_loss}")
        # print(f"max_seq_len: {max_seq_len}, test loss: {test_loss}")



for name in [
    # "retry2_repeat_custom4096_len256",
    # "retry_repeat_custom4096_len256_nope",
    # "retry2_repeat_custom4096_len256_memory32_ffn_norm_reusekv_trainmem",

    "retry_reverse_custom4096_len256",
    # "retry_reverse_custom4096_len256_nope",
    # "retry_reverse_custom4096_len256_memory32_ffn_norm_reusekv_trainmem",
    # "retry_reverse_custom4096_len1024",

    # "infinity_repeat_custom4096_len256_memory32_ffn_norm_reusekv_updatemem*"
]:
    # pattern = re.compile(r".*(ReRoPE).*\.txt")
    pattern = re.compile(r".*(log_selfExtend).*\.txt")
    # pattern = re.compile(r".*(SWA).*\.txt")
    # pattern = re.compile(r".*(selfExtend|ReRoPE).*\.txt")
    files = glob.glob(f"./out/{name}/*.txt")
    matching_files = [file for file in files if pattern.search(file)]


    print(name)
    for file in matching_files:
        parse(file)
