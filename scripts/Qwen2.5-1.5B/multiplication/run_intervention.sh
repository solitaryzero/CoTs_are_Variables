python ./src/multiplication/run_intervention.py \
    --base_model ./models/Qwen2.5-1.5B \
    --save_model_path ./models/multiplication/full_cot/Qwen2.5-1.5B/4_mul_4 \
    --data_path ./results/multiplication/intervened/Qwen2.5-1.5B/intervened.json \
    --save_result_path ./results/multiplication/intervened/Qwen2.5-1.5B \
    --max_new_tokens 1024 \
    --seed 42