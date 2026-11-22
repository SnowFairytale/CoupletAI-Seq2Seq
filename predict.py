"""Interactive / command-line inference for a trained couplet model.

Examples::

    python predict.py --checkpoint results/checkpoints/gru_attn_best.pt \
        --text "春暮偶登楼，上下鱼龙，应惜满湖绿水" --beam_size 5

    python predict.py --checkpoint results/checkpoints/gru_attn_best.pt
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import torch

from module import Tokenizer, build_model
from module.inference import predict_texts

DEMOS = [
    "春暮偶登楼，上下鱼龙，应惜满湖绿水",
    "一曲笙歌春似海",
    "马齿草焉无马齿",
    "我有诗情堪纵酒",
    "海内存知己",
    "天增岁月人增寿",
    "日出江花红胜火",
    "明月松间照",
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data", default="data", type=str)
    parser.add_argument("--text", nargs="*", default=None, type=str)
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--demo", action="store_true",
                        help="run the built-in examples instead of the interactive shell")
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = Tokenizer.load(Path(args.data) / "vocab.json")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    model = build_model(model_name, tokenizer.vocab_size,
                        pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id,
                        eos_id=tokenizer.eos_id,
                        **checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"loaded {args.checkpoint} ({model_name}, epoch {checkpoint.get('epoch')})")

    if args.text:
        predictions = predict_texts(model, tokenizer, args.text, device,
                                    model_name, args.beam_size)
        for upper, lower in zip(args.text, predictions):
            print(f"上联：{upper}\n下联：{lower}\n")
        return

    if args.demo:
        predictions = predict_texts(model, tokenizer, DEMOS, device,
                                    model_name, args.beam_size)
        for upper, lower in zip(DEMOS, predictions):
            print(f"上联：{upper}\n下联：{lower}\n")
        return

    print("输入上联，回车生成下联；输入 q 退出。")
    while True:
        try:
            upper = input("上联：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if upper.lower() == "q":
            break
        if not upper:
            continue
        lower = predict_texts(model, tokenizer, [upper], device,
                              model_name, args.beam_size)[0]
        print(f"下联：{lower}")


if __name__ == "__main__":
    main()
