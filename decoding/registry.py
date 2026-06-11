"""
Decoder registry.

Use this from scripts/generate.py so the script does not need many if/else
blocks for greedy, DoLA, VCD, and PHG variants.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_decoder_config(
    decoding_name: str,
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Build config object by decoding name.
    """

    config_kwargs = config_kwargs or {}
    name = decoding_name.lower()

    if name == "greedy":
        from decoding.greedy import GreedyConfig

        return GreedyConfig(**config_kwargs)

    if name in {"dola", "dola_low"}:
        from decoding.dola import DoLAConfig

        if "dola_layers" not in config_kwargs:
            config_kwargs["dola_layers"] = "low"

        return DoLAConfig(**config_kwargs)

    if name == "vcd":
        from decoding.vcd import VCDConfig

        return VCDConfig(**config_kwargs)

    if name in {"greedy_phg", "dola_phg", "dola_low_phg", "vcd_phg"}:
        from phg import PHGConfig

        if name == "greedy_phg":
            config_kwargs.setdefault("decoding_mode", "greedy")

        elif name in {"dola_phg", "dola_low_phg"}:
            config_kwargs.setdefault("decoding_mode", "dola")
            config_kwargs.setdefault("dola_layers", "low")

        elif name == "vcd_phg":
            config_kwargs.setdefault("decoding_mode", "vcd")

        return PHGConfig(**config_kwargs)

    raise ValueError(f"Unknown decoding name: {decoding_name}")


def generate_samples_with_decoder(
    decoding_name: str,
    wrapper,
    samples,
    config: Optional[Any] = None,
    config_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Dispatch generation by decoding name.
    """

    name = decoding_name.lower()

    if config is None:
        config = build_decoder_config(
            decoding_name=name,
            config_kwargs=config_kwargs,
        )

    if name == "greedy":
        from decoding.greedy import generate_greedy_samples

        return generate_greedy_samples(
            wrapper=wrapper,
            samples=samples,
            config=config,
            **kwargs,
        )

    if name in {"dola", "dola_low"}:
        from decoding.dola import generate_dola_samples

        return generate_dola_samples(
            wrapper=wrapper,
            samples=samples,
            config=config,
            **kwargs,
        )

    if name == "vcd":
        from decoding.vcd import generate_vcd_samples

        return generate_vcd_samples(
            wrapper=wrapper,
            samples=samples,
            config=config,
            **kwargs,
        )

    if name in {"greedy_phg", "dola_phg", "dola_low_phg", "vcd_phg"}:
        from phg import generate_phg_samples

        return generate_phg_samples(
            wrapper=wrapper,
            samples=samples,
            config=config,
            **kwargs,
        )

    raise ValueError(f"Unknown decoding name: {decoding_name}")