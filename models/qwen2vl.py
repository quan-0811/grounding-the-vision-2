"""
Qwen2-VL wrapper.

Default target:
    Qwen/Qwen2-VL-7B-Instruct

Also works for:
    Qwen/Qwen2.5-VL-7B-Instruct

Requires:
    pip install qwen-vl-utils

Use:
    use_chat_template=True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    process_vision_info = None

from models.base import GenerationOutput, PathLike, TensorDict
from utils.image import load_image
from utils.seed import get_torch_dtype


@dataclass
class Qwen2VLConfig:
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    torch_dtype: str = "bfloat16"
    device_map: Union[str, dict, None] = "auto"
    attn_implementation: Optional[str] = None

    # Optional Qwen processor image-resolution constraints.
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None

    default_prompt: str = "Describe this image."


class Qwen2VLWrapper:
    def __init__(self, config: Optional[Qwen2VLConfig] = None) -> None:
        self.config = config or Qwen2VLConfig()

        self.model = None
        self.processor = None
        self.tokenizer = None

    def load(self) -> "Qwen2VLWrapper":
        dtype = get_torch_dtype(self.config.torch_dtype)

        processor_kwargs = {}

        if self.config.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.config.min_pixels

        if self.config.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.config.max_pixels

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            **processor_kwargs,
        )

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": self.config.device_map,
        }

        if self.config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            **model_kwargs,
        )

        self.tokenizer = self.processor.tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if getattr(self.model.config, "pad_token_id", None) is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        return self

    def _load_images(
        self,
        image_paths: Optional[Sequence[PathLike]] = None,
        images: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        if images is not None:
            loaded = []

            for img in images:
                if isinstance(img, Image.Image):
                    loaded.append(img.convert("RGB"))
                else:
                    loaded.append(img)

            return loaded

        if image_paths is None:
            raise ValueError("Either image_paths or images must be provided.")

        return [str(path) for path in image_paths]

    def _build_messages(
        self,
        image_obj: Any,
        prompt: str,
    ) -> List[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_obj,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

    def prepare_batch(
        self,
        image_paths: Optional[Sequence[PathLike]] = None,
        images: Optional[Sequence[Any]] = None,
        prompts: Union[str, Sequence[str]] = "Describe this image.",
        use_chat_template: Optional[bool] = True,
        **kwargs: Any,
    ) -> TensorDict:
        if self.processor is None:
            raise RuntimeError("Call wrapper.load() first.")

        if process_vision_info is None:
            raise ImportError(
                "Qwen2VLWrapper requires qwen-vl-utils. "
                "Install with: pip install qwen-vl-utils"
            )

        image_objs = self._load_images(
            image_paths=image_paths,
            images=images,
        )

        if isinstance(prompts, str):
            prompts = [prompts] * len(image_objs)

        prompts = list(prompts)

        if len(prompts) != len(image_objs):
            raise ValueError(
                f"prompts length {len(prompts)} != images length {len(image_objs)}"
            )

        messages_batch = [
            self._build_messages(image_obj, prompt)
            for image_obj, prompt in zip(image_objs, prompts)
        ]

        if use_chat_template is False:
            # Qwen2-VL still needs the special vision tokens. The safest path
            # is to keep the chat template even if the generic caller passes
            # False accidentally.
            use_chat_template = True

        texts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_batch
        ]

        image_inputs = []
        video_inputs = []

        for messages in messages_batch:
            img_inputs, vid_inputs = process_vision_info(messages)
            image_inputs.extend(img_inputs or [])

            if vid_inputs is not None:
                video_inputs.extend(vid_inputs)

        processor_kwargs = {
            "text": texts,
            "images": image_inputs,
            "padding": True,
            "return_tensors": "pt",
        }

        if len(video_inputs) > 0:
            processor_kwargs["videos"] = video_inputs

        inputs = self.processor(**processor_kwargs)

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

    def get_image_grid_shape(self, inputs: Optional[TensorDict] = None) -> Optional[tuple[int, int]]:
        """
        Qwen2-VL can have dynamic image grids.

        Prefer passing inputs into grounding.resolve_image_grid_shape().
        """

        if inputs is None or "image_grid_thw" not in inputs:
            return None

        grid = inputs["image_grid_thw"]

        if torch.is_tensor(grid):
            grid = grid.detach().cpu()

        if grid.dim() == 2:
            t, h, w = grid[0].tolist()
        else:
            t, h, w = grid.tolist()

        if int(t) != 1:
            return None

        return int(h), int(w)

    def get_suppress_token_ids(self) -> list[int]:
        """
        Tokens that should not be generated as normal text.
        Useful for manual decoding methods like VCD.
        """
    
        candidate_tokens = [
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|quad_start|>",
            "<|quad_end|>",
        ]
    
        ids = []
    
        for tok in candidate_tokens:
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
    
            if tok_id is None:
                continue
    
            if tok_id == self.tokenizer.unk_token_id:
                continue
    
            ids.append(int(tok_id))
    
        return sorted(set(ids))