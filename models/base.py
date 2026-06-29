"""
Base model wrapper interface.

Every LVLM wrapper should expose:

    - wrapper.model
    - wrapper.processor or wrapper.tokenizer
    - load()
    - prepare_batch()
    - generate_from_inputs()
    - batch_decode()

This allows decoding/greedy.py, decoding/dola.py, decoding/vcd.py,
decoding/stepwise.py, and phg/generator.py to stay model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union

import torch

from utils.io import PathLike

TensorDict = Dict[str, Any]


@dataclass
class GenerationOutput:
    """
    Normalized output for all wrappers.
    """

    captions: List[str]
    sequences: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    raw_outputs: Optional[Any] = None


class BaseLVLM(Protocol):
    """
    Wrapper protocol.

    Decoders depend on this interface.
    """

    model: Any

    def load(self) -> "BaseLVLM":
        ...

    def prepare_batch(
        self,
        image_paths: Optional[Sequence[PathLike]] = None,
        images: Optional[Sequence[Any]] = None,
        prompts: Union[str, Sequence[str]] = "Describe this image.",
        use_chat_template: Optional[bool] = None,
        **kwargs: Any,
    ) -> TensorDict:
        ...

    def generate_from_inputs(
        self,
        inputs: TensorDict,
        **generate_kwargs: Any,
    ) -> GenerationOutput:
        ...

    def batch_decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        ...