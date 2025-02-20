import re
import glob

def parse(file):
    print(file)
    with open(file, 'r') as file:
        log_content = file.read()

    # 使用正则表达式匹配 max_seq_len 和 test loss
    max_seq_len_pattern = re.compile(r'max_seq_len (\d+)')
    test_loss_pattern = re.compile(r'test loss ([\d.]+)')
    test_acc_pattern = re.compile(r'test acc ([\d.]+)')

    # 找到所有匹配的 max_seq_len 和 test loss
    max_seq_lens = max_seq_len_pattern.findall(log_content)
    test_losses = test_loss_pattern.findall(log_content)
    test_accs = test_acc_pattern.findall(log_content)

    # 打印结果
    for max_seq_len, test_loss, test_acc in zip(max_seq_lens, test_losses, test_accs):
        # print(f"{max_seq_len:5}: {test_acc}")
        # print(f"{max_seq_len:5}: {1 - float(test_acc):.4}")
        print(f"max_seq_len: {max_seq_len} loss: {test_loss} acc: {test_acc:.4}")



for name in [
    # "retry2_repeat_custom4096_len256",
    # "retry_repeat_custom4096_len256_nope",
    # "retry2_repeat_custom4096_len256_memory32_ffn_norm_reusekv_trainmem",

    # "retry_reverse_custom4096_len256_theta",
    # "retry_reverse_custom4096_len256_theta100000",
    # "retry_reverse_custom4096_len256_theta1000000",
    # "retry_reverse_custom4096_len256_xpos",
    # "retry_reverse_custom4096_len256_xpos128",
    # "retry_reverse_custom4096_len256_xpos1024",

    # "retry_reverse_custom4096_len1024",
    # "retry5_reverse_custom4096_len1024",
    "retry5_reverse_custom4096_len1024_ConcatPE",
    # "retry5_reverse_custom4096_len1024_xpos1024",
    # "retry5_reverse_custom4096_len1024_xpos128",
    # "retry5_reverse_custom4096_len1024_xpos32",

    # "retry_reverse_custom4096_len256",
    # "retry_reverse_custom4096_len256_nope",
    # "retry_reverse_custom4096_len256_memory32_ffn_norm_reusekv_trainmem",

    # "retry_reverse_custom4096_len256_freqsAbs",
    # "retry_reverse_custom4096_len256_sumCis",
    # "retry_reverse_custom4096_len256_sumCis_freqsAbs",

    # "infinity_repeat_custom4096_len256_memory32_ffn_norm_reusekv",
    # "infinity_repeat_custom4096_len256_memory32_ffn_norm_reusekv_updatemem",
    # "retry_reverse_custom4096_len256_memory32_ffn_norm_reusekv_updatemem",

    # "infinity_repeat_custom4096_len1024_memory64_ffn_norm_reusekv_updatemem*",
    # "infinity_reverse_custom4096_len1024_memory64_ffn_norm_reusekv_updatemem*",
]:
    pattern = re.compile(r".*\.txt")
    #pattern = re.compile(r".*(ReRoPE).*\.txt")
    # pattern = re.compile(r".*(log_selfExtend).*\.txt")
    # pattern = re.compile(r".*(SWA|AMask|ADouble).*\.txt")
    # pattern = re.compile(r".*(selfExtend|SWA).*\.txt")
    # pattern = re.compile(r".*(SWA|ReRoPE).*\.txt")
    files = glob.glob(f"./out/{name}/log2*.txt")
    matching_files = [file for file in files if pattern.search(file)]


    print("="*30, name)
    for file in matching_files:
        parse(file)
