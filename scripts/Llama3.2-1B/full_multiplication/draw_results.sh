python ./src/multiplication/draw_results.py \
    --data_path ./results/full_multiplication/plain/Llama-3.2-1B/all_scores.json \
    --out_path ./results/full_multiplication/accuracy/Llama-3.2-1B \
    --prompt_type plain \
    --format pdf

python ./src/multiplication/draw_results.py \
    --data_path ./results/full_multiplication/full_cot/Llama-3.2-1B \
    --out_path ./results/full_multiplication/accuracy/Llama-3.2-1B \
    --prompt_type full \
    --from_separate_files \
    --format pdf

# python ./src/multiplication/draw_results.py \
#     --data_path ./results/full_multiplication/compressed_cot/Llama-3.2-1B \
#     --out_path ./results/full_multiplication/accuracy/Llama-3.2-1B \
#     --prompt_type compressed \
#     --from_separate_files \
#     --format pdf

python ./src/multiplication/draw_results.py \
    --data_path ./results/full_multiplication/latent/Llama-3.2-1B \
    --out_path ./results/full_multiplication/accuracy/Llama-3.2-1B \
    --prompt_type latent \
    --from_separate_files \
    --format pdf