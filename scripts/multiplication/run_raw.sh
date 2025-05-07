python ./src/multiplication/run_raw.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/multiplication/4_mul_4.json \
    --save_model_path ./models/multiplication/plain/Qwen2.5-1.5B \
    --save_result_path ./results/multiplication/raw/Qwen2.5-1.5B \
    --max_new_tokens 50 \
    --seed 42