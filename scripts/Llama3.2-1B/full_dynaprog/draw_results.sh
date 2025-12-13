python ./src/dynaprog/draw_results.py \
    --data_path ./results/full_dynaprog/plain/Llama-3.2-1B \
    --out_path ./results/full_dynaprog/accuracy/Llama-3.2-1B \
    --from_separate_files \
    --prompt_type plain \
    --format pdf

python ./src/multiplication/draw_results.py \
    --data_path ./results/full_dynaprog/full_cot/Llama-3.2-1B \
    --out_path ./results/full_dynaprog/accuracy/Llama-3.2-1B \
    --prompt_type full \
    --from_separate_files \
    --format pdf

python ./src/multiplication/draw_results.py \
    --data_path ./results/full_dynaprog/latent/Llama-3.2-1B \
    --out_path ./results/full_dynaprog/accuracy/Llama-3.2-1B \
    --prompt_type latent \
    --from_separate_files \
    --format pdf