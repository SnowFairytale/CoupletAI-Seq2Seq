# 自动对联 Seq2Seq 最优模型参数记录

本文件对应课程项目“基于 Seq2Seq（GRU + Attention）的自动对联生成”的**当前最优模型**。权重文件为 `gru_attn_seq2seq_best.pt`，按验证集损失选择，对应第 **7** 轮。

## 1. 提交文件

| 文件 | 说明 |
| --- | --- |
| `gru_attn_seq2seq_best.pt` | 最优模型权重，标准 FP32 版（PyTorch checkpoint，58.42 MB，推荐） |
| `gru_attn_seq2seq_best_fp16.pt` | 同一权重的 FP16 压缩版（29.22 MB，适合有 50 MB 限制的提交系统；加载时自动转回 FP32） |
| `vocab.json` | 字符级词表，推理时必需 |
| `model_config.json` | 机器可读的完整参数记录（结构、超参数、指标、SHA256） |
| `best_params.md` | 本文件，人类可读的参数说明 |
| `test_metrics_greedy.json` | 测试集贪心解码指标（4,000 条） |
| `test_metrics_fixed_len.json` | 测试集固定长度解码指标（4,000 条） |
| `test_metrics_beam5_1000.json` | 测试集 beam=5 指标（1,000 条子集） |
| `model_architecture.png` | 模型结构图 |
| `loss_curves.png` | 训练/验证损失曲线 |
| `inspect_checkpoint.py` | 不依赖项目代码的权重检查脚本 |

两个权重文件内容等价、任选其一：标准 FP32 版精度最高，FP16 版体积约一半、加载时自动转为 FP32，测试指标几乎相同。提交系统若限制 50 MB，使用 FP16 版。

## 2. 模型结构

| 组件 | 设置 | 参数量 |
| --- | --- | ---: |
| 编码器字符嵌入 | 9,132 × 256 | 2,337,792 |
| 编码器 | 2 层双向 GRU，hidden 256，含初始状态 bridge | 2,234,880 |
| 注意力 | Bahdanau 加性注意力，256 维 | 196,864 |
| 解码器字符嵌入 | 9,132 × 256 | 2,337,792 |
| 解码器 | 2 层 GRU，输入 768（256 嵌入 + 512 上下文），hidden 256 | 1,182,720 |
| 输出层 | 768 → 9,132 | 7,022,508 |
| **合计** |  | **15,312,556** |

- 词表大小：9,132
- 特殊符号：`[PAD]=0`、`[UNK]=1`、`[BOS]=2`、`[EOS]=3`
- 最大序列长度：32
- 解码：默认 beam=5、长度惩罚 0.6、固定长度（利用上下联等长先验）；也支持贪心与自由 EOS

## 3. 训练超参数

| 超参数 | 值 |
| --- | --- |
| 训练轮数 | 7 |
| batch size | 512 |
| 优化器 | Adam |
| 学习率 | 1×10⁻³ |
| weight decay | 0 |
| 梯度裁剪 | 3.0 |
| teacher forcing | 1.0 |
| label smoothing | 0 |
| 混合精度 | FP16（`torch.amp`） |
| 学习率调度 | ReduceLROnPlateau（factor 0.5，patience 1，min_lr 1e-5） |
| 早停 | 验证损失连续 2 轮不降则停止（本次未触发） |
| 随机种子 | 42 |
| 最佳轮次 | 第 7 轮（验证损失 3.596798） |
| 训练总耗时 | 约 4,784 秒（约 80 分钟） |
| 硬件 | NVIDIA RTX 4060 Laptop GPU（8 GB） |
| 框架 | PyTorch 2.10.0+cu126，Python 3.13 |

## 4. 数据划分

| 划分 | 对数 | 说明 |
| --- | ---: | --- |
| 训练 | 750,491 | 原始 770,491 对中划出 20,000 对验证 |
| 验证 | 20,000 | 从训练集随机抽取（seed=42），用于选模型 |
| 测试 | 4,000 | 官方测试集，仅最终评估使用 |

词表 9,132，OOV 率 0，最大长度 32，平均长度约 9.35。

## 5. 最优模型指标

### 5.1 验证集（第 7 轮）

| 指标 | 值 |
| --- | ---: |
| 验证损失 | 3.5968 |
| 验证困惑度 | 36.48 |
| BLEU-4（1,000 条验证子集，贪心） | 0.0366 |
| ROUGE-L（同上） | 0.1543 |
| 长度一致率（同上） | 0.994 |
| 字级准确率（同上） | 0.1438 |

### 5.2 测试集（4,000 条，贪心解码）

| 指标 | 值 |
| --- | ---: |
| BLEU-1 | 0.1523 |
| BLEU-2 | 0.0703 |
| BLEU-3 | 0.0435 |
| BLEU-4 | 0.0326 |
| ROUGE-L | 0.1459 |
| 完全匹配率 | 0.00125 |
| 长度一致率 | 0.9963 |
| 字级准确率 | 0.1353 |
| distinct-1 / distinct-2 | 0.0541 / 0.4554 |
| teacher forcing 损失 | 3.5961 |
| 困惑度 | 36.46 |

### 5.3 测试集（1,000 条子集，beam=5）

| 指标 | 值 |
| --- | ---: |
| BLEU-1 / 2 / 3 / 4 | 0.1569 / 0.0764 / 0.0483 / 0.0366 |
| ROUGE-L | 0.1505 |
| 完全匹配率 | 0.0030 |
| 长度一致率 | 1.0000 |
| 字级准确率 | 0.1402 |

说明：由于同一上联存在多个合法下联，单参考 BLEU 非常严苛；同一上联的两个不同人工下联之间 BLEU-4 平均仅 0.092（中位数 0.020）。详细分析见项目 `report.md`。

## 6. 如何加载

### 6.1 在完整项目中直接预测

```bash
conda activate dip
cd MySolution
python predict.py \
    --checkpoint submission/gru_attn_seq2seq_best.pt \
    --data submission \
    --text "一曲笙歌春似海" \
    --beam_size 5 --fixed_len
```

### 6.2 仅检查权重文件

```bash
python submission/inspect_checkpoint.py
```

### 6.3 Python 中读取

```python
import torch

ckpt = torch.load("submission/gru_attn_seq2seq_best.pt",
                  map_location="cpu", weights_only=False)
print(ckpt["model_name"])    # gru_attn
print(ckpt["model_kwargs"])  # embed/hidden/layers/dropout
print(ckpt["epoch"], ckpt["val_loss"])
state_dict = ckpt["model_state"]
```

`model_kwargs` 为：

```json
{"embed_dim": 256, "hidden_dim": 256, "enc_layers": 2, "dec_layers": 2, "dropout": 0.2}
```

## 7. 示例输出

| 上联 | 模型生成（beam=5） |
| --- | --- |
| 一曲笙歌春似海 | 万家灯火夜如潮 |
| 我有诗情堪纵酒 | 谁无画意可吟诗 |
| 海内存知己 | 天涯有故人 |
| 明月松间照 | 清风袖里香 |
| 晓风亲绿水 | 旭日映红霞 |
| 水流知入海 | 山峻欲凌云 |

## 8. 文件校验

### 8.1 标准 FP32 版

| 属性 | 值 |
| --- | --- |
| 文件名 | `gru_attn_seq2seq_best.pt` |
| 大小 | 58.42 MB（61258856 字节） |
| SHA256 | `3778535938d78f3e9c19ca41536d145ba1c5f73ef69691a5e860fe25a6420edb` |
| checkpoint keys | `model_state`, `model_name`, `model_kwargs`, `epoch`, `val_loss`, `metrics`, `args` |
| state_dict 张量数 | 33 |
| 可训练参数量 | 15,312,556 |

### 8.2 FP16 压缩版

| 属性 | 值 |
| --- | --- |
| 文件名 | `gru_attn_seq2seq_best_fp16.pt` |
| 大小 | 29.22 MB（30636352 字节） |
| SHA256 | `d402413054384ce1804a4b5516eaf6cc3173abcca7ff6316cd461f84f5fe8878` |
| 精度 | 权重以 FP16 存储；`load_state_dict` 载入 FP32 模型时自动转换 |
| 测试集 BLEU-4 | 0.0325544（标准版 0.0325548） |
| 测试集 ROUGE-L | 0.14586（标准版 0.14585） |
| 测试集困惑度 | 36.4565（与标准版相同） |

两个版本的预测结果几乎完全一致。若提交系统限制 50 MB，使用 `gru_attn_seq2seq_best_fp16.pt` 即可；否则推荐使用标准 FP32 版。
