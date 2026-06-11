"""
LLaVA-1.5 wrapper.

Default target:
    llava-hf/llava-1.5-7b-hf

Works with:
    - greedy.py
    - dola.py
    - vcd.py
    - stepwise.py
    - phg/generator.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from models.base import GenerationOutput, PathLike, TensorDict
from utils.image import load_image
from utils.seed import get_torch_dtype


@dataclass
class Llava15Config:
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    torch_dtype: str = "float16"
    device_map: Union[str, dict, None] = "auto"
    attn_implementation: Optional[str] = None

    # LLaVA-1.5 usually uses 576 image tokens = 24x24.
    image_grid_shape: tuple[int, int] = (24, 24)

    # Prompt style used in your notebooks.
    default_system_prompt: Optional[str] = None


class Llava15Wrapper:
    def __init__(self, config: Optional[Llava15Config] = None) -> None:
        self.config = config or Llava15Config()

        self.model = None
        self.processor = None
        self.tokenizer = None

    def load(self) -> "Llava15Wrapper":
        dtype = get_torch_dtype(self.config.torch_dtype)

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
        )

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": self.config.device_map,
        }

        if self.config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )

        self.tokenizer = self.processor.tokenizer

        # Important for batched generation.
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        return self

    def _format_prompt(
        self,
        prompt: str,
        use_chat_template: Optional[bool] = False,
    ) -> str:
        """
        LLaVA-1.5 notebook-compatible prompt.
        """

        if use_chat_template:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            return self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )

        return f"USER: <image>\n{prompt} ASSISTANT:"

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

    def prepare_batch(
        self,
        image_paths: Optional[Sequence[PathLike]] = None,
        images: Optional[Sequence[Any]] = None,
        prompts: Union[str, Sequence[str]] = "Describe this image.",
        use_chat_template: Optional[bool] = False,
        **kwargs: Any,
    ) -> TensorDict:
        if self.processor is None:
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

        texts = [
            self._format_prompt(prompt, use_chat_template=use_chat_template)
            for prompt in prompts
        ]

        inputs = self.processor(
            text=texts,
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )

        return dict(inputs)

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        return [
            text.strip()
            for text in self.processor.batch_decode(
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

        if "pad_token_id" not in generate_kwargs:
            generate_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        if "eos_token_id" not in generate_kwargs:
            generate_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        output_ids = self.model.generate(
            **moved,
            **generate_kwargs,
        )

        prompt_len = moved["input_ids"].shape[1]
        new_token_ids = output_ids[:, prompt_len:]

        captions = self.batch_decode(new_token_ids)

        return GenerationOutput(
            captions=captions,
            sequences=output_ids,
            input_ids=moved["input_ids"],
            raw_outputs=output_ids,
        )

    def get_image_grid_shape(self, inputs: Optional[TensorDict] = None) -> tuple[int, int]:
        return self.config.image_grid_shape