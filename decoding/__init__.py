"""
Decoding package.
"""

from decoding.greedy import (
    GreedyConfig,
    GreedyDecoder,
    generate_greedy_batch,
    generate_greedy_from_inputs,
    generate_greedy_samples,
)

from decoding.dola import (
    DoLAConfig,
    DoLADecoder,
    generate_dola_batch,
    generate_dola_from_inputs,
    generate_dola_samples,
)

from decoding.vcd import (
    VCDConfig,
    VCDDecoder,
    add_diffusion_noise,
    apply_vcd_logits,
    generate_vcd_batch,
    generate_vcd_from_inputs,
    generate_vcd_samples,
)

from decoding.stepwise import (
    StepRecord,
    StepwiseConfig,
    StepwiseDecoder,
    StepwiseOutput,
    generate_stepwise_batch,
    generate_stepwise_from_inputs,
)

from decoding.utils import (
    move_inputs_to_model,
    strip_private_inputs,
    get_model,
    get_processor,
    get_tokenizer,
    get_eos_token_ids,
    get_fallback_token_id,
    is_qwen_wrapper
)