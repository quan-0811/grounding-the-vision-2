"""
PHG package.
"""

from phg.types import (
    CandidateBranch,
    CandidateRecord,
    CheckpointState,
    PHGConfig,
    PHGDecodingMode,
    PHGOutput,
    SegmentScore,
)

from phg.memory import PHGMemory

from phg.candidates import (
    build_candidate_records,
    candidates_to_trace,
    first_boundary_candidate,
    is_sentence_end_token,
    select_candidate_tokens,
)

from phg.checkpoint import (
    append_generated_ids_to_inputs,
    build_checkpoint,
    extract_output_segment_by_abs_range,
    normalize_prefix_ids,
)

from phg.scoring import score_segment

from phg.generator import (
    PHGGenerator,
    generate_phg_batch,
    generate_phg_from_inputs,
    generate_phg_samples,
)