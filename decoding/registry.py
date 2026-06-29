# decoding/registry.py

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from models.base import BaseLVLM
from decoding.utils import is_qwen_wrapper


SUPPORTED_DECODERS = {
    "greedy",
    "dola_low",
    "vcd",
}


def build_decoder_config(
    decoding_name: str,
    config_kwargs: Optional[Dict[str, Any]] = None,
):
    config_kwargs = config_kwargs or {}
    name = decoding_name.lower()

    if name == "greedy":
        from decoding.greedy import GreedyConfig

        return GreedyConfig(**config_kwargs)

    if name in {"dola_low", "dola"}:
        from decoding.dola import DoLAConfig

        config_kwargs.setdefault("dola_layers", "low")
        return DoLAConfig(**config_kwargs)

    if name == "vcd":
        from decoding.vcd import VCDConfig

        return VCDConfig(**config_kwargs)

    raise ValueError(
        f"Unknown decoding name: {decoding_name}. "
        f"Supported decoders: {sorted(SUPPORTED_DECODERS)}"
    )


def generate_samples_with_decoder(
    decoding_name: str,
    wrapper: BaseLVLM,
    samples: Sequence[Dict[str, Any]],
    config: Any = None,
    **kwargs: Any,
):
    name = decoding_name.lower()

    if name == "greedy":
        from decoding.greedy import generate_greedy_samples

        return generate_greedy_samples(
            wrapper=wrapper,
            samples=samples,
            config=config,
            **kwargs,
        )

    if name in {"dola_low", "dola"}:
        from decoding.dola import generate_dola_samples

        return generate_dola_samples(
            wrapper=wrapper,
            samples=samples,
            config=config,
            **kwargs,
        )

    if name == "vcd":
        if is_qwen_wrapper(wrapper):
            from decoding.qwen_vcd import generate_qwen_vcd_samples

            return generate_qwen_vcd_samples(
                wrapper=wrapper,
                samples=samples,
                config=config,
                **kwargs,
            )

        from decoding.vcd import generate_vcd_samples

        return generate_vcd_samples(
            wrapper=wrapper,
            samples=samples,
            config=config,
            **kwargs,
        )

    raise ValueError(
        f"Unknown decoding name: {decoding_name}. "
        f"Supported decoders: {sorted(SUPPORTED_DECODERS)}"
    )