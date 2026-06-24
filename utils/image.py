"""
Image utilities.
"""

from __future__ import annotations

from PIL import Image

from utils.io import PathLike


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