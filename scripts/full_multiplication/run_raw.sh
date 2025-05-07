python ./src/multiplication/run_raw.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/multiplication \
    --save_model_path ./models/full_multiplication/plain/Qwen2.5-1.5B \
    --save_result_path ./results/full_multiplication/raw/Qwen2.5-1.5B \
    --max_new_tokens 50 \
    --bf16 \
    --run_all \
    --seed 42