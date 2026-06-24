# models/qwen2vl.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

from models.base import GenerationOutput, PathLike, TensorDict
from utils.image import load_image
from utils.image_noise import add_diffusion_noise_to_pil
from utils.seed import get_torch_dtype


@dataclass
class Qwen2VLConfig:
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    torch_dtype: str = "float16"
    device_map: Union[str, dict, None] = "auto"
    attn_implementation: Optional[str] = None

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
        self.tokenizer.padding_side = "left"

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

            for image in images:
                if isinstance(image, Image.Image):
                    loaded.append(image.convert("RGB"))
                else:
                    loaded.append(image)

            return loaded

        if image_paths is None:
            raise ValueError("Either image_paths or images must be provided.")

        # Official Qwen path accepts image paths in messages.
        return [str(path) for path in image_paths]

    def _load_pil_images_for_vcd(
        self,
        image_paths: Optional[Sequence[PathLike]] = None,
        images: Optional[Sequence[Any]] = None,
    ) -> List[Image.Image]:
        if images is not None:
            loaded = []

            for image in images:
                if isinstance(image, Image.Image):
                    loaded.append(image.convert("RGB"))
                else:
                    raise TypeError(
                        "Qwen2VL VCD with in-memory images expects PIL.Image.Image."
                    )

            return loaded

        if image_paths is None:
            raise ValueError("Missing image_paths/images for Qwen2VL VCD.")

        return [load_image(path, mode="RGB") for path in image_paths]

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

    def _model_inputs_only(
        self,
        inputs: TensorDict,
    ) -> TensorDict:
        return {
            key: value
            for key, value in inputs.items()
            if not str(key).startswith("_")
        }

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
            self._build_messages(image_obj=image_obj, prompt=prompt)
            for image_obj, prompt in zip(image_objs, prompts)
        ]

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

            if img_inputs is not None:
                image_inputs.extend(img_inputs)

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

        inputs = dict(self.processor(**processor_kwargs))

        # Private metadata for Qwen VCD only.
        inputs["_qwen_prompts"] = prompts

        if image_paths is not None:
            inputs["_qwen_image_paths"] = [str(path) for path in image_paths]
            inputs["_qwen_images"] = None
        else:
            inputs["_qwen_image_paths"] = None
            inputs["_qwen_images"] = [
                image.convert("RGB") if isinstance(image, Image.Image) else image
                for image in images
            ]

        return inputs

    def prepare_vcd_inputs(
        self,
        inputs: TensorDict,
        noise_step: int = 500,
    ) -> TensorDict:
        prompts = inputs.get("_qwen_prompts", None)
        image_paths = inputs.get("_qwen_image_paths", None)
        images = inputs.get("_qwen_images", None)

        if prompts is None:
            raise ValueError(
                "Missing `_qwen_prompts`. "
                "Qwen VCD requires inputs from Qwen2VLWrapper.prepare_batch()."
            )

        pil_images = self._load_pil_images_for_vcd(
            image_paths=image_paths,
            images=images,
        )

        noised_images = [
            add_diffusion_noise_to_pil(
                image=image,
                noise_step=noise_step,
            )
            for image in pil_images
        ]

        return self.prepare_batch(
            images=noised_images,
            prompts=prompts,
            use_chat_template=True,
        )

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
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

        model_inputs = self._model_inputs_only(inputs)

        moved = {}

        for key, value in model_inputs.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    moved[key] = value.to(
                        device=model_device,
                        dtype=model_dtype,
                    )
                else:
                    moved[key] = value.to(device=model_device)
            else:
                moved[key] = value

        generate_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        generate_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        output_ids = self.model.generate(
            **moved,
            **generate_kwargs,
            trust_remote_code=True,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(moved["input_ids"], output_ids)
        ]

        new_token_ids = torch.nn.utils.rnn.pad_sequence(
            generated_ids_trimmed,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        captions = self.batch_decode(new_token_ids)

        return GenerationOutput(
            captions=captions,
            sequences=output_ids,
            input_ids=moved["input_ids"],
            raw_outputs=output_ids,
        )

    def get_image_grid_shape(
        self,
        inputs: Optional[TensorDict] = None,
    ) -> Optional[tuple[int, int]]:
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