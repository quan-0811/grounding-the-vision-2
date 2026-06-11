"""
InternVL2 wrapper.

Default target:
    OpenGVLab/InternVL2-8B

Important:
    InternVL2 cannot use a plain prompt like:

        <image>
        Describe this image.

    For raw model.generate(), InternVL2 expects the prompt to contain many
    <IMG_CONTEXT> tokens:

        <img><IMG_CONTEXT><IMG_CONTEXT>...</img>
        Describe this image.

    It also requires:
        model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

This wrapper prepares that format correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from models.base import GenerationOutput, PathLike, TensorDict
from utils.image import load_image
from utils.seed import get_torch_dtype


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


@dataclass
class InternVL2Config:
    model_id: str = "OpenGVLab/InternVL2-8B"
    torch_dtype: str = "bfloat16"
    device_map: Union[str, dict, None] = "auto"
    trust_remote_code: bool = True

    input_size: int = 448
    max_num_tiles: int = 12

    default_prompt: str = "Describe this image."


def build_transform(input_size: int):
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio

        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    """
    InternVL2 dynamic tiling.
    """

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set()

    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if min_num <= i * j <= max_num:
                    target_ratios.add((i, j))

    target_ratios = sorted(
        target_ratios,
        key=lambda x: x[0] * x[1],
    )

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio=aspect_ratio,
        target_ratios=target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]

    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))

    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )

        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


class InternVL2Wrapper:
    def __init__(self, config: Optional[InternVL2Config] = None) -> None:
        self.config = config or InternVL2Config()

        self.model = None
        self.processor = None
        self.tokenizer = None

        self.transform = build_transform(self.config.input_size)

        self.img_context_token_id: Optional[int] = None
        self.num_image_token: Optional[int] = None

    def load(self) -> "InternVL2Wrapper":
        dtype = get_torch_dtype(self.config.torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=False,
        )

        self.model = AutoModel.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
        ).eval()

        self.processor = self.tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(
            IMG_CONTEXT_TOKEN
        )

        if (
            self.img_context_token_id is None
            or self.img_context_token_id == self.tokenizer.unk_token_id
        ):
            raise ValueError(
                f"Could not resolve {IMG_CONTEXT_TOKEN} token id. "
                "InternVL2 tokenizer special tokens may not be loaded correctly."
            )

        # This is required by InternVL2 remote generate().
        self.model.img_context_token_id = self.img_context_token_id

        self.num_image_token = self._resolve_num_image_token()

        return self

    def _resolve_num_image_token(self) -> int:
        """
        Resolve number of visual context tokens per image tile.

        InternVL2 usually exposes model.num_image_token.
        Fallback formula:
            (image_size // patch_size)^2 * downsample_ratio^2
        """

        if hasattr(self.model, "num_image_token"):
            return int(self.model.num_image_token)

        vision_config = getattr(self.model.config, "vision_config", None)

        image_size = getattr(
            vision_config,
            "image_size",
            self.config.input_size,
        )

        patch_size = getattr(
            vision_config,
            "patch_size",
            14,
        )

        downsample_ratio = getattr(
            self.model.config,
            "downsample_ratio",
            0.5,
        )

        return int((image_size // patch_size) ** 2 * (downsample_ratio**2))

    def _load_images(
        self,
        image_paths: Optional[Sequence[PathLike]] = None,
        images: Optional[Sequence[Any]] = None,
    ) -> List[Image.Image]:
        if images is not None:
            return [
                img.convert("RGB") if isinstance(img, Image.Image) else img
                for img in images
            ]

        if image_paths is None:
            raise ValueError("Either image_paths or images must be provided.")

        return [load_image(path, mode="RGB") for path in image_paths]

    def _image_to_pixel_values(self, image: Image.Image) -> torch.Tensor:
        tiles = dynamic_preprocess(
            image=image,
            image_size=self.config.input_size,
            max_num=self.config.max_num_tiles,
            use_thumbnail=True,
        )

        pixel_values = [
            self.transform(tile)
            for tile in tiles
        ]

        return torch.stack(pixel_values)

    def _format_prompt(
        self,
        prompt: str,
        num_patches: int,
    ) -> str:
        """
        Build InternVL2 raw-generate prompt.

        For N image tiles:
            num_context_tokens = model.num_image_token * N
        """

        if self.num_image_token is None:
            raise RuntimeError("Call wrapper.load() first.")

        image_context = IMG_CONTEXT_TOKEN * (self.num_image_token * num_patches)

        image_prefix = (
            f"{IMG_START_TOKEN}"
            f"{image_context}"
            f"{IMG_END_TOKEN}"
        )

        return f"{image_prefix}\n{prompt}"

    def prepare_batch(
        self,
        image_paths: Optional[Sequence[PathLike]] = None,
        images: Optional[Sequence[Any]] = None,
        prompts: Union[str, Sequence[str]] = "Describe this image.",
        use_chat_template: Optional[bool] = None,
        **kwargs: Any,
    ) -> TensorDict:
        if self.tokenizer is None:
            raise RuntimeError("Call wrapper.load() first.")

        pil_images = self._load_images(
            image_paths=image_paths,
            images=images,
        )

        if isinstance(prompts, str):
            prompts = [prompts] * len(pil_images)

        prompts = list(prompts)

        if len(prompts) != len(pil_images):
            raise ValueError(
                f"prompts length {len(prompts)} != images length {len(pil_images)}"
            )

        pixel_values_list = [
            self._image_to_pixel_values(image)
            for image in pil_images
        ]

        num_patches_list = [
            pixel_values.shape[0]
            for pixel_values in pixel_values_list
        ]

        pixel_values = torch.cat(pixel_values_list, dim=0)

        formatted_prompts = [
            self._format_prompt(
                prompt=prompt,
                num_patches=num_patches,
            )
            for prompt, num_patches in zip(prompts, num_patches_list)
        ]

        tokenized = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
        )

        input_ids = tokenized["input_ids"]

        img_context_count = int(
            (input_ids == self.img_context_token_id).sum().item()
        )

        expected_img_context_count = int(
            sum(num_patches_list) * self.num_image_token
        )

        if img_context_count != expected_img_context_count:
            raise ValueError(
                "InternVL2 prompt expansion failed.\n"
                f"Found IMG_CONTEXT tokens: {img_context_count}\n"
                f"Expected IMG_CONTEXT tokens: {expected_img_context_count}\n"
                "This usually means the tokenizer did not recognize "
                "<IMG_CONTEXT> as a special token."
            )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "pixel_values": pixel_values,
            "num_patches_list": num_patches_list,
            "prompts": formatted_prompts,
        }

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        return [
            text.strip()
            for text in self.tokenizer.batch_decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
        ]

    @torch.inference_mode()
    def generate_from_inputs(
        self,
        inputs: TensorDict,
        **generate_kwargs: Any,
    ) -> GenerationOutput:
        if self.model is None:
            raise RuntimeError("Call wrapper.load() first.")
    
        # InternVL2 remote generate() requires this.
        self.model.img_context_token_id = self.img_context_token_id
    
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
    
        moved = {}
    
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    moved[key] = value.to(device=model_device, dtype=model_dtype)
                else:
                    moved[key] = value.to(device=model_device)
            else:
                moved[key] = value
    
        # ------------------------------------------------------------
        # Important InternVL2 fix:
        # Its remote generate() already passes use_cache internally.
        # If our generic decoders/scripts also pass use_cache=True,
        # language_model.generate() receives duplicate use_cache args.
        # ------------------------------------------------------------
        generate_kwargs = dict(generate_kwargs)
        generate_kwargs.pop("use_cache", None)
    
        # These are also unsafe for InternVL2 raw remote generate().
        # Stepwise/PHG is not supported for InternVL2 yet anyway.
        generate_kwargs.pop("return_dict", None)
        generate_kwargs.pop("output_attentions", None)
        generate_kwargs.pop("output_hidden_states", None)
    
        if "pad_token_id" not in generate_kwargs:
            generate_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
    
        if (
            "eos_token_id" not in generate_kwargs
            and self.tokenizer.eos_token_id is not None
        ):
            generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
    
        output_ids = self.model.generate(
            pixel_values=moved["pixel_values"],
            input_ids=moved["input_ids"],
            attention_mask=moved["attention_mask"],
            **generate_kwargs,
        )
        
        prompt_len = moved["input_ids"].shape[1]
        
        # LLaVA/Qwen HF generate usually returns:
        #     prompt + generated tokens
        #
        # InternVL2 remote generate usually returns:
        #     generated tokens only
        #
        # So only slice if the returned sequence is longer than the prompt.
        if output_ids.shape[1] > prompt_len:
            new_token_ids = output_ids[:, prompt_len:]
        else:
            new_token_ids = output_ids
        
        captions = self.batch_decode(new_token_ids)
        
        return GenerationOutput(
            captions=captions,
            sequences=output_ids,
            input_ids=moved["input_ids"],
            raw_outputs=output_ids,
        )

    @torch.inference_mode()
    def chat_generate(
        self,
        image_path: PathLike,
        prompt: str = "Describe this image.",
        max_new_tokens: int = 256,
        do_sample: bool = False,
    ) -> str:
        """
        Official-style fallback using InternVL2's remote .chat() method.
        """

        if self.model is None:
            raise RuntimeError("Call wrapper.load() first.")

        self.model.img_context_token_id = self.img_context_token_id

        image = load_image(image_path, mode="RGB")
        pixel_values = self._image_to_pixel_values(image)

        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype

        pixel_values = pixel_values.to(
            device=model_device,
            dtype=model_dtype,
        )

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        question = f"<image>\n{prompt}"

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
        )

        return response.strip()

    def get_image_grid_shape(self, inputs: Optional[TensorDict] = None):
        return None