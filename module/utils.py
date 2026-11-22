"""Small shared helpers."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep cuDNN autotuning on: the input shapes vary from batch to batch and
    # deterministic mode is considerably slower on this workload.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(obj: Dict[str, Any], filename: Union[str, Path]) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(filename: Union[str, Path]) -> Dict[str, Any]:
    with Path(filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, *args) -> None:
        pass
