import torch
from models.base import TensorDict
from typing import Dict, Any, List, Sequence
from models.base import BaseLVLM

def strip_private_inputs(inputs: TensorDict) -> TensorDict:
    return {
        key: value
        for key, value in inputs.items()
        if not str(key).startswith("_")
    }

def move_inputs_to_model(
    inputs: Dict[str, Any],
    model: Any,
) -> Dict[str, Any]:
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype

    moved = {}

    for key, value in strip_private_inputs(inputs).items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value

    return moved

def get_model(wrapper: BaseLVLM) -> Any:
    if hasattr(wrapper, "model"):
        return wrapper.model

    if hasattr(wrapper, "get_model"):
        return wrapper.get_model()

    raise AttributeError("Wrapper must expose `.model` or `.get_model()`.")


def get_processor(wrapper: BaseLVLM) -> Any:
    if hasattr(wrapper, "processor"):
        return wrapper.processor

    if hasattr(wrapper, "get_processor"):
        return wrapper.get_processor()

    raise AttributeError("Wrapper must expose `.processor` or `.get_processor()`.")


def get_tokenizer(wrapper: BaseLVLM, processor: Any) -> Any:
    if hasattr(wrapper, "tokenizer"):
        return wrapper.tokenizer

    if hasattr(processor, "tokenizer"):
        return processor.tokenizer

    return processor

def get_eos_token_ids(model: Any, tokenizer: Any) -> List[int]:
    eos_token_id = getattr(model.generation_config, "eos_token_id", None)

    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    if eos_token_id is None:
        return []

    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]

    return [int(x) for x in eos_token_id]

def get_fallback_token_id(
    processor: Any,
    eos_token_ids: Sequence[int],
) -> int:
    if hasattr(processor, "tokenizer"):
        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        eos_id = getattr(processor.tokenizer, "eos_token_id", None)

        if pad_id is not None:
            return int(pad_id)

        if eos_id is not None:
            return int(eos_id)

    if len(eos_token_ids) > 0:
        return int(eos_token_ids[0])

    return 0

def is_qwen_wrapper(wrapper: BaseLVLM) -> bool:
    model_id = str(
        getattr(
            getattr(wrapper, "config", None),
            "model_id",
            "",
        )
    ).lower()

    return "qwen2-vl" in model_id