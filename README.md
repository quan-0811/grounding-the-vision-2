# Grounding the Vision — PHG (Post-Hoc Grounding)

Mitigating object hallucination in Large Vision-Language Models (LVLMs) during image captioning via **PHG** — a decoding-time intervention that uses cross-attention maps to verify whether generated nouns are grounded in image regions.

## How it works

PHG decodes text sentence-by-sentence. When the model becomes uncertain (low token confidence), it creates a checkpoint, scores the prefix for hallucination using attention maps, then branches alternative continuations from the top-K candidate tokens. Each branch is scored by its **Attention Dispersion Score** (ADS) — compact attention = real object; diffuse attention = hallucination. The best branch is selected, and accepted objects are tracked in memory so later sentences stay consistent.

PHG wraps three base decoding strategies: **greedy**, **DoLA** (contrastive layers), and **VCD** (Visual Contrastive Decoding).

## Quick start

```bash
# Create conda environment
conda create -n phg python=3.11 -y
conda activate phg

# Install
pip install -r requirements.txt
python scripts/download_nltk.py
python -m spacy download en_core_web_lg

# Download datasets
bash download_cocoval2017.sh     # COCO val2017 images + annotations
bash download_amber.sh           # AMBER images + metadata

# Generate captions
python scripts/generate.py \
  --model llava15_7b --decoding greedy_phg --dataset coco_val2017 \
  --output outputs/full/greedy_phg_llava15_coco.json
# Required: --model, --decoding, --dataset, --output

# Evaluate
python scripts/eval_chair.py --input outputs/full/greedy_phg_llava15_coco.json
python scripts/eval_amber.py --input outputs/full/output_amber.json
```

**Supported models:** `llava15_7b`, `qwen2vl_7b`  
**Supported decodings:** `greedy`, `dola_low`, `vcd`, `greedy_phg`, `dola_low_phg`, `vcd_phg`  
**Supported datasets:** `coco_val2017`, `amber`

## Project structure

```
scripts/generate.py         ← CLI entry point, config merging, orchestration
├── models/                 ← LVLM wrappers (BaseLVLM Protocol)
│   ├── llava15.py          ← LLaVA-1.5-7B
│   └── qwen2vl.py          ← Qwen2-VL-7B
├── decoding/               ← Decoding strategies (model-agnostic)
│   ├── greedy.py           ← Standard greedy
│   ├── dola.py             ← DoLA (contrastive layers)
│   ├── vcd.py              ← VCD (clean vs diffused-image contrast)
│   ├── qwen_vcd.py         ← VCD for Qwen (LogitsProcessor-based)
│   └── stepwise.py         ← Token-by-token decoding with attention extraction
├── phg/                    ← PHG system (the core contribution)
│   ├── generator.py        ← Main loop: rounds, uncertainty, branching, reranking
│   ├── memory.py           ← Global + per-sentence object tracking
│   ├── scoring.py          ← Noun→mask→ADS→IoU grounding pipeline
│   ├── candidates.py       ← Candidate token selection, sentence-end detection
│   └── checkpoint.py       ← Checkpoint build/restore, prefix management
├── grounding/              ← Attention→grounding utilities (model-agnostic)
│   ├── attention.py        ← Attention extraction from transformer layers
│   ├── ads.py              ← Attention Dispersion Score (low ADS = real)
│   ├── masks.py            ← Binary object masks from top-k attention heads
│   ├── iou.py              ← Mask compatibility via IoU threshold
│   ├── grid.py             ← 1D attention → 2D grid reshaping
│   └── noun_extraction.py  ← NLTK-based noun detection
├── data/                   ← Dataset loaders (COCO, AMBER)
├── evaluation/             ← CHAIR, AMBER, caption quality metrics
├── configs/                ← YAML configs (model + dataset + decoding)
└── utils/                  ← I/O, config loading, image noise, seed
```
