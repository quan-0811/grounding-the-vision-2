# utils/image_noise.py

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image


def get_diffusion_coefficients(
    noise_step: int = 500,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise_step = int(max(0, min(999, noise_step)))
    device = device or torch.device("cpu")

    betas = torch.linspace(
        -6,
        6,
        1000,
        device=device,
        dtype=torch.float32,
    )

    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)

    sqrt_alpha_prod = torch.sqrt(alphas_prod[noise_step])
    sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - alphas_prod[noise_step])

    return sqrt_alpha_prod, sqrt_one_minus_alpha_prod


def add_diffusion_noise_to_tensor(
    tensor: torch.Tensor,
    noise_step: int = 500,
) -> torch.Tensor:
    orig_dtype = tensor.dtype
    device = tensor.device

    sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_diffusion_coefficients(
        noise_step=noise_step,
        device=device,
    )

    sqrt_alpha_prod = sqrt_alpha_prod.to(dtype=orig_dtype)
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.to(dtype=orig_dtype)

    noise = torch.randn_like(tensor)

    return sqrt_alpha_prod * tensor + sqrt_one_minus_alpha_prod * noise


def add_diffusion_noise_to_pil(
    image: Image.Image,
    noise_step: int = 500,
) -> Image.Image:
    image = image.convert("RGB")

    array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array)

    sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_diffusion_coefficients(
        noise_step=noise_step,
        device=torch.device("cpu"),
    )

    noise = torch.randn_like(tensor)

    noised = sqrt_alpha_prod * tensor + sqrt_one_minus_alpha_prod * noise
    noised = torch.clamp(noised, 0.0, 1.0)

    noised_array = (noised.numpy() * 255.0).round().astype("uint8")

    return Image.fromarray(noised_array, mode="RGB")