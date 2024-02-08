# Llama2RNN.c：一个用C语言实现的终身 RNN 模型

[![zh](https://img.shields.io/badge/zh-简体中文-red.svg)](README.md)
[![en](https://img.shields.io/badge/en-English-green.svg)](README.en.md)

这是一个使用 Llama2 权重设计的循环神经网络（RNN）模型，旨在无限期运行（终身）。

- **llama2**: 可以使用 llama2 各种版本模型的权重
- **rnn**: 每个token的 attention sequence 长度固定，计算和内存开销不会增加，理论上支持无限长序列，可以从硬盘读取和保存记忆
- **.c**: 可以在本地设备上运行，甚至是移动平台

## 如何训练

### 数据处理

参考[README_llama2.c.md](./README_llama2.c.md)处理好数据

```bash
python3 tinystories.py download
python3 tinystories.py train_vocab --vocab_size=4096
python3 tinystories.py pretokenize --vocab_size=4096
```

### 训练

```bash
python3 train.py config/train_tinystories_token4096_memorynorm.py
```

### 保存

```bash
python3 tokenizer.py --tokenizer-model ./data/tok4096.model
export.py out_path/model_q80.bin --version 2 --mem --checkpoint out_path/ckpt.pt
```

## 其它

更多细节说明见[llama2Rnn.c/README.md at main · siyuanseever/llama2Rnn.c (github.com)](https://github.com/siyuanseever/llama2Rnn.c/blob/main/README.md)

## 许可证

MIT
