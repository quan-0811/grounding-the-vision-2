for DECODING in greedy dola_low vcd greedy_phg dola_low_phg vcd_phg; do
  python scripts/generate.py \
    --model llava15_7b \
    --decoding ${DECODING} \
    --dataset coco_val2017 \
    --output outputs/smoke_coco/llava15_${DECODING}_coco_n3.json \
    --dtype float16 \
    --device-map auto \
    --attn-implementation eager \
    --batch-size 1 \
    --max-samples 3 \
    --seed 42 \
    --prompt "Describe this image." \
    --max-new-tokens 64 \
    --temperature 1.0 \
    --repetition-penalty 1.2 \
    --cd-alpha 1.0 \
    --cd-beta 0.1 \
    --noise-step 500 \
    --dola-relative-top 0.1 \
    --dola-select-strategy argmax \
    --phg-max-rounds 4 \
    --phg-min-new-tokens 3 \
    --phg-top-k 3 \
    --phg-accumulate-prob 0.5 \
    --phg-iou-thresh 0.5 \
    --phg-ads-thresh 0.45 \
    --phg-ads-foreground-ratio 0.10 \
    --selected-layers=-8,-4,-1 \
    --coco-image-dir data/coco2017/val2017 \
    --coco-annotation-path data/coco2017/annotations/instances_val2017.json \
    --include-trace
done

for DECODING in greedy dola_low vcd greedy_phg dola_low_phg vcd_phg; do
  python scripts/generate.py \
    --model llava15_7b \
    --decoding ${DECODING} \
    --dataset amber \
    --output outputs/smoke_amber/llava15_${DECODING}_amber_n3.json \
    --dtype float16 \
    --device-map auto \
    --attn-implementation eager \
    --batch-size 1 \
    --max-samples 3 \
    --seed 42 \
    --prompt "Describe this image." \
    --max-new-tokens 64 \
    --temperature 1.0 \
    --repetition-penalty 1.2 \
    --cd-alpha 1.0 \
    --cd-beta 0.1 \
    --noise-step 500 \
    --dola-relative-top 0.1 \
    --dola-select-strategy argmax \
    --phg-max-rounds 4 \
    --phg-min-new-tokens 3 \
    --phg-top-k 3 \
    --phg-accumulate-prob 0.5 \
    --phg-iou-thresh 0.5 \
    --phg-ads-thresh 0.45 \
    --phg-ads-foreground-ratio 0.10 \
    --selected-layers=-8,-4,-1 \
    --amber-root data/amber \
    --amber-image-dir data/amber/image \
    --amber-query-path data/amber/query/query_generative.json \
    --amber-annotation-path data/amber/annotations.json \
    --include-trace
done

for DECODING in greedy dola_low vcd greedy_phg dola_low_phg vcd_phg; do
  python scripts/generate.py \
    --model qwen2vl_7b \
    --decoding ${DECODING} \
    --dataset coco_val2017 \
    --output outputs/smoke_coco/qwen2vl_${DECODING}_coco_n3.json \
    --dtype float16 \
    --device-map auto \
    --attn-implementation eager \
    --batch-size 1 \
    --max-samples 3 \
    --seed 42 \
    --prompt "Describe this image." \
    --max-new-tokens 64 \
    --temperature 1.0 \
    --repetition-penalty 1.2 \
    --cd-alpha 1.0 \
    --cd-beta 0.1 \
    --noise-step 500 \
    --dola-relative-top 0.1 \
    --dola-select-strategy argmax \
    --phg-max-rounds 4 \
    --phg-min-new-tokens 3 \
    --phg-top-k 3 \
    --phg-accumulate-prob 0.5 \
    --phg-iou-thresh 0.5 \
    --phg-ads-thresh 0.45 \
    --phg-ads-foreground-ratio 0.10 \
    --selected-layers=-8,-4,-1 \
    --coco-image-dir data/coco2017/val2017 \
    --coco-annotation-path data/coco2017/annotations/instances_val2017.json \
    --qwen-min-pixels 200704 \
    --qwen-max-pixels 1003520 \
    --include-trace
done

for DECODING in greedy dola_low vcd greedy_phg dola_low_phg vcd_phg; do
  python scripts/generate.py \
    --model qwen2vl_7b \
    --decoding ${DECODING} \
    --dataset amber \
    --output outputs/smoke_amber/qwen2vl_${DECODING}_amber_n3.json \
    --dtype float16 \
    --device-map auto \
    --attn-implementation eager \
    --batch-size 1 \
    --max-samples 3 \
    --seed 42 \
    --prompt "Describe this image." \
    --max-new-tokens 64 \
    --temperature 1.0 \
    --repetition-penalty 1.2 \
    --cd-alpha 1.0 \
    --cd-beta 0.1 \
    --noise-step 500 \
    --dola-relative-top 0.1 \
    --dola-select-strategy argmax \
    --phg-max-rounds 4 \
    --phg-min-new-tokens 3 \
    --phg-top-k 3 \
    --phg-accumulate-prob 0.5 \
    --phg-iou-thresh 0.5 \
    --phg-ads-thresh 0.45 \
    --phg-ads-foreground-ratio 0.10 \
    --selected-layers=-8,-4,-1 \
    --amber-root data/amber \
    --amber-image-dir data/amber/image \
    --amber-query-path data/amber/query/query_generative.json \
    --amber-annotation-path data/amber/annotations.json \
    --qwen-min-pixels 200704 \
    --qwen-max-pixels 1003520 \
    --include-trace
done