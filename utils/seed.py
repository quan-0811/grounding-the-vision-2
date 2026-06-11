"""
Reproducibility utilities.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(
    seed: int = 42,
    deterministic: bool = False,
) -> int:
    """
    Seed Python, NumPy, and PyTorch.

    deterministic=True can slow down generation and is usually not needed
    for normal LVLM inference.
    """

    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def get_torch_dtype(dtype_name: Optional[str]):
    """
    Convert string dtype from YAML into torch dtype.

    Examples:
        "float16" -> torch.float16
        "bfloat16" -> torch.bfloat16
        "float32" -> torch.float32
    """

    if dtype_name is None:
        return None

    name = str(dtype_name).lower()

    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,

        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,

        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }

    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")

    return mapping[name]