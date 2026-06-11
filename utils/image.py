"""
Image utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

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


def get_image_size(path: PathLike) -> Tuple[int, int]:
    """
    Return image size as (width, height).
    """

    with Image.open(path) as img:
        return img.size


def resize_longest_edge(
    image: Image.Image,
    max_longest_edge: int,
) -> Image.Image:
    """
    Resize image so longest edge <= max_longest_edge.

    Keeps aspect ratio.
    """

    width, height = image.size

    longest = max(width, height)

    if longest <= max_longest_edge:
        return image

    scale = float(max_longest_edge) / float(longest)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return image.resize((new_width, new_height), Image.BICUBIC)


def maybe_resize_image(
    image: Image.Image,
    max_longest_edge: Optional[int] = None,
) -> Image.Image:
    """
    Optional resize helper.
    """

    if max_longest_edge is None:
        return image

    return resize_longest_edge(
        image=image,
        max_longest_edge=max_longest_edge,
    )