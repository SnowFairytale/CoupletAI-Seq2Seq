# 模型参数提交包

本目录是“基于 Seq2Seq（GRU + Attention）的自动对联生成”项目的**当前最优模型参数包**。

> 权重提供两个版本，二者任选其一即可：标准 FP32 版 `gru_attn_seq2seq_best.pt`（58.42 MB，推荐），以及 FP16 压缩版 `gru_attn_seq2seq_best_fp16.pt`（29.22 MB，适合 50 MB 限制的提交系统，预测结果几乎一致）。根据体积限制，压缩包中只放其中一个版本。

## 快速使用

在完整的项目目录 `MySolution/` 下运行：

```bash
conda activate dip
cd MySolution

python predict.py \
    --checkpoint submission/gru_attn_seq2seq_best.pt \
    --data submission \
    --text "一曲笙歌春似海" \
    --beam_size 5 --fixed_len
```

输出示例：

```
上联：一曲笙歌春似海
下联：万家灯火夜如潮
```

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `gru_attn_seq2seq_best.pt` | 标准 FP32 最优权重（第 7 轮，验证损失 3.5968，15,312,556 参数，58.42 MB） |
| `gru_attn_seq2seq_best_fp16.pt` | FP16 压缩版权重（29.22 MB，适合 50 MB 限制；预测几乎一致） |
| `vocab.json` | 字符级词表（9,132 个 token，含 PAD/UNK/BOS/EOS） |
| `model_config.json` | 完整参数记录：结构、训练超参数、数据划分、测试指标、SHA256 |
| `best_params.md` | 人类可读的参数说明与结果表 |
| `test_metrics_*.json` | 测试集指标（贪心 4,000 条 / 固定长度 4,000 条 / beam=5 1,000 条） |
| `model_architecture.png` | 模型结构图 |
| `loss_curves.png` | 训练与验证损失曲线 |
| `inspect_checkpoint.py` | 仅依赖 PyTorch 的权重检查脚本 |

## 仅检查权重

```bash
python submission/inspect_checkpoint.py
```

输出包括模型名、结构参数、最佳轮次、验证损失、参数量和 SHA256。

## 体积压缩版

如果提交系统限制 50 MB，使用同目录的 FP16 版：

```bash
python predict.py \
    --checkpoint submission/gru_attn_seq2seq_best_fp16.pt \
    --data submission \
    --text "一曲笙歌春似海" \
    --beam_size 5 --fixed_len
```

FP16 版权重以半精度存储，但加载到 FP32 模型时会自动转换；在 4,000 条测试集上
BLEU-4 为 0.0325544（标准版 0.0325548），ROUGE-L 为 0.14586（标准版 0.14585），
困惑度相同。若没有体积限制，优先提交标准 FP32 版。

## Python 加载

```python
import torch

ckpt = torch.load("submission/gru_attn_seq2seq_best.pt",
                  map_location="cpu", weights_only=False)
print(ckpt["model_name"])       # gru_attn
print(ckpt["model_kwargs"])     # 结构参数
print(ckpt["epoch"], ckpt["val_loss"])
state_dict = ckpt["model_state"]
```

## 完整性校验

```text
SHA256: 3778535938d78f3e9c19ca41536d145ba1c5f73ef69691a5e860fe25a6420edb
```

## 注意

- `predict.py` 和 `module/` 位于项目根目录；本目录只放模型参数与说明，不重复代码。
- 数据集（约 200 MB）和训练日志不在本包中；数据集下载和完整复现命令见项目根目录 `README.md`。
- 更详细的实验分析、错误分析和定性样例见项目根目录 `report.md`。
