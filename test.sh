for MODEL in llava15 qwen2vl; do
  for METHOD in greedy dola_low vcd greedy_phg dola_low_phg vcd_phg; do
    python scripts/eval_output_stats.py \
      --pred outputs/full/coco_val2017/${MODEL}/${METHOD}.json \
      --instances data/coco2017/annotations/instances_val2017.json \
      --out outputs/eval/output_stats/${MODEL}/${METHOD}.json \
      --per-image-out outputs/eval/output_stats/${MODEL}/${METHOD}_per_image.csv
  done
done