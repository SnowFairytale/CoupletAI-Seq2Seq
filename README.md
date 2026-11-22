# 自动对联生成：Seq2Seq（GRU + Attention）

机器学习概论课程项目。给定上联，模型自回归地逐字生成下联。

样例代码 `CoupletAI` 利用“上下联字数相同”这一特点，把对联生成转化为**序列标注**问题：用 BiLSTM 对每个上联字打一个下联字标签，一次前向得到整句下联。本项目按照课程要求把它改写为**编码器—解码器（Seq2Seq）**框架，并使用与样例不同的 RNN 结构：**双向 GRU 编码器 + 带 Bahdanau 注意力的 GRU 解码器**。

## 1. 与样例的主要区别

| 方面 | 样例 CoupletAI | 本项目 MySolution |
| --- | --- | --- |
| 任务形式 | 序列标注（逐位置分类） | Seq2Seq（编码 + 自回归解码） |
| RNN 结构 | 单层 BiLSTM（默认） | 2 层双向 GRU 编码器 + 2 层 GRU 解码器 |
| 注意力 | 无（BiLSTM 版本） | Bahdanau 加性注意力 |
| 解码 | 一次前向，argmax | 逐字解码：贪心 / 束搜索（beam search） |
| 句子边界 | 无 BOS/EOS | 显式 `[BOS]` / `[EOS]`，teacher forcing 训练 |
| 数据划分 | 官方 train / test | 从 train 中再划出 20,000 条验证集，test 只用于最终评估 |
| 评估 | BLEU-4、ROUGE-L | BLEU-1~4、ROUGE-L、完全匹配、长度一致率、字级准确率、distinct-n |
| 对照实验 | — | 同数据同轮数的序列标注基线；无注意力消融 |

## 2. 项目结构

```
MySolution/
├── module/
│   ├── tokenizer.py      # 字符级词表，含 PAD/UNK/BOS/EOS
│   ├── dataset.py        # 变长 batch 的 Dataset / collate
│   ├── model.py          # Seq2Seq(GRU+Attention)、无注意力消融、BiLSTM 标注基线
│   ├── inference.py      # 贪心 / 束搜索解码封装
│   ├── metrics.py        # BLEU、ROUGE-L、长度一致率、distinct-n 等
│   └── utils.py
├── preprocess.py         # 原始数据 -> 词表 + 训练/验证/测试数组
├── train.py              # 训练、每轮验证、早停、日志、checkpoint
├── evaluate.py           # 在 val/test 上计算指标并保存逐句结果
├── predict.py            # 命令行交互式对对联
├── plot.py               # 损失曲线 / 指标曲线 / 模型对比图
├── requirements.txt
├── report.md             # 实验报告
└── results/              # 日志、指标、样例、图片
```

## 3. 环境与数据

```bash
conda activate dip
pip install -r requirements.txt
```

原始数据沿用样例目录 `../CoupletAI/couplet`（70 余万条对联，上/下联以空格分隔的单字）。预处理：

```bash
python preprocess.py --input ../CoupletAI/couplet --output data --val_size 20000
```

会得到 `data/vocab.json`、`data/train.npz`、`data/val.npz`、`data/test.npz` 和 `data/meta.json`。数据规模：训练 750,491 对，验证 20,000 对，测试 4,000 对；词表 9,132（含 4 个特殊符号），OOV 率为 0。

## 4. 训练

主模型（2 层 BiGRU 编码器 + 2 层 GRU 解码器 + 加性注意力，batch 512，Adam，学习率 1e-3）：

```bash
python train.py --model gru_attn --run_name gru_attn \
    --epochs 8 --batch_size 512 --lr 1e-3 --eval_samples 1000
```

序列标注基线（复现样例的 BiLSTM 标注结构，用于对照）：

```bash
python train.py --model bilstm_tagger --run_name bilstm_tagger \
    --epochs 3 --batch_size 512 --embed_dim 256 --hidden_dim 256
```

无注意力消融：

```bash
python train.py --model gru_noattn --run_name gru_noattn \
    --epochs 6 --batch_size 512
```

训练日志写入 `results/logs/`，最优/最后 checkpoint 写入 `results/checkpoints/`（已在 `.gitignore` 中排除）。

## 5. 评估与推理

```bash
python evaluate.py --checkpoint results/checkpoints/gru_attn_best.pt --split test --name gru_attn
python predict.py --checkpoint results/checkpoints/gru_attn_best.pt --text "春暮偶登楼，上下鱼龙，应惜满湖绿水" --beam_size 5
python predict.py --checkpoint results/checkpoints/gru_attn_best.pt --demo
```

评估结果保存为 `results/metrics/<name>_<split>.json`，逐句预测保存为 `results/samples/<name>_<split>.csv`。

## 6. 实验结果

（见 [report.md](report.md)；训练曲线在 `results/figures/`。）

## 7. 说明

样例代码与数据来自 [wb14123/CoupletAI](https://github.com/wb14123/CoupletAI)，仅用于课程实验对照。
