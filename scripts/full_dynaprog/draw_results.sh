python ./src/dynaprog/draw_results.py \
    --data_path ./results/full_dynaprog/plain/Qwen2.5-1.5B/all_scores.json \
    --out_path ./results/full_dynaprog/accuracy \
    --from_separate_files \
    --prompt_type plain \
    --format pdf

python ./src/multiplication/draw_results.py \
    --data_path ./results/full_dynaprog/full_cot/Qwen2.5-1.5B \
    --out_path ./results/full_dynaprog/accuracy \
    --prompt_type full \
    --from_separate_files \
    --format pdf

python ./src/multiplication/draw_results.py \
    --data_path ./results/full_dynaprog/latent/Qwen2.5-1.5B \
    --out_path ./results/full_dynaprog/accuracy \
    --prompt_type latent \
    --from_separate_files \
    --format pdf