python ./src/multiplication/draw_results.py \
    --data_path ./results/full_multiplication/plain/Qwen2.5-1.5B/all_scores.json \
    --out_path ./results/full_multiplication/accuracy \
    --prompt_type plain \
    --format pdf

python ./src/multiplication/draw_results.py \
    --data_path ./results/full_multiplication/full_cot/Qwen2.5-1.5B \
    --out_path ./results/full_multiplication/accuracy \
    --prompt_type full \
    --from_separate_files \
    --format pdf

python ./src/multiplication/draw_results.py \
    --data_path ./results/full_multiplication/compressed_cot/Qwen2.5-1.5B \
    --out_path ./results/full_multiplication/accuracy \
    --prompt_type compressed \
    --from_separate_files \
    --format pdf

python ./src/multiplication/draw_results.py \
    --data_path ./results/full_multiplication/latent/Qwen2.5-1.5B \
    --out_path ./results/full_multiplication/accuracy \
    --prompt_type latent \
    --from_separate_files \
    --format pdf