"""
Image utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image


PathLike = Union[str, Path]


def load_image(
    path: PathLike,
    mode: str = "RGB",
) -> Image.Image:
    """
    Load image with PIL.
    """

    image = Image.open(path)

    if mode is not None:
        image = image.convert(mode)

    return image