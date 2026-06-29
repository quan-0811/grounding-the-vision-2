"""
Greedy decoding for LVLM caption generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from models.base import BaseLVLM, GenerationOutput, PathLike, TensorDict


@dataclass
class GreedyConfig:
    max_new_tokens: int = 256
    use_cache: bool = True

    do_sample: bool = False
    num_beams: int = 1

    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None

    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    extra_generate_kwargs: Dict[str, Any] = field(default_factory=dict)


class GreedyDecoder:
    def __init__(self, config: Optional[GreedyConfig] = None) -> None:
        self.config = config or GreedyConfig()

    def build_generate_kwargs(self, **override_kwargs: Any) -> Dict[str, Any]:
        cfg = self.config

        kwargs: Dict[str, Any] = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "use_cache": cfg.use_cache,
        }

        if cfg.repetition_penalty is not None:
            kwargs["repetition_penalty"] = cfg.repetition_penalty

        if cfg.length_penalty is not None:
            kwargs["length_penalty"] = cfg.length_penalty

        if cfg.no_repeat_ngram_size is not None:
            kwargs["no_repeat_ngram_size"] = cfg.no_repeat_ngram_size

        if cfg.do_sample:
            kwargs["temperature"] = cfg.temperature
            if cfg.top_p is not None:
                kwargs["top_p"] = cfg.top_p
            if cfg.top_k is not None:
                kwargs["top_k"] = cfg.top_k

        kwargs.update(cfg.extra_generate_kwargs)
        kwargs.update(override_kwargs)

        kwargs["do_sample"] = False
        kwargs["num_beams"] = 1

        return kwargs

    def generate_from_inputs(
        self,
        wrapper: BaseLVLM,
        inputs: TensorDict,
        **override_generate_kwargs: Any,
    ) -> GenerationOutput:
        return wrapper.generate_from_inputs(
            inputs=inputs,
            **self.build_generate_kwargs(**override_generate_kwargs),
        )

    def generate_batch(
        self,
        wrapper: BaseLVLM,
        image_paths: Optional[Sequence[PathLike]] = None,
        images: Optional[Sequence[Any]] = None,
        prompts: Union[str, Sequence[str]] = "Describe this image.",
        use_chat_template: Optional[bool] = None,
        **prepare_kwargs: Any,
    ) -> GenerationOutput:
        inputs = wrapper.prepare_batch(
            image_paths=image_paths,
            images=images,
            prompts=prompts,
            use_chat_template=use_chat_template,
            **prepare_kwargs,
        )

        return self.generate_from_inputs(wrapper, inputs)

    def generate_samples(
        self,
        wrapper: BaseLVLM,
        samples: Sequence[Dict[str, Any]],
        image_key: str = "image_path",
        prompt_key: str = "prompt",
        id_key: str = "id",
        caption_key: str = "caption",
        use_chat_template: Optional[bool] = False,
        include_prompt: bool = False,
        **prepare_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if len(samples) == 0:
            return []

        image_paths = [sample[image_key] for sample in samples]
        prompts = [sample.get(prompt_key, "Describe this image.") for sample in samples]

        output = self.generate_batch(
            wrapper=wrapper,
            image_paths=image_paths,
            prompts=prompts,
            use_chat_template=use_chat_template,
            **prepare_kwargs,
        )

        rows: List[Dict[str, Any]] = []

        for sample, caption in zip(samples, output.captions):
            row: Dict[str, Any] = {
                id_key: int(sample[id_key]),
                caption_key: caption,
            }

            if include_prompt:
                row[prompt_key] = sample.get(prompt_key, "Describe this image.")

            rows.append(row)

        return rows


def generate_greedy_samples(
    wrapper: BaseLVLM,
    samples: Sequence[Dict[str, Any]],
    config: Optional[GreedyConfig] = None,
    image_key: str = "image_path",
    prompt_key: str = "prompt",
    id_key: str = "id",
    caption_key: str = "caption",
    use_chat_template: Optional[bool] = False,
    include_prompt: bool = False,
    **prepare_kwargs: Any,
) -> List[Dict[str, Any]]:
    return GreedyDecoder(config).generate_samples(
        wrapper=wrapper,
        samples=samples,
        image_key=image_key,
        prompt_key=prompt_key,
        id_key=id_key,
        caption_key=caption_key,
        use_chat_template=use_chat_template,
        include_prompt=include_prompt,
        **prepare_kwargs,
    )