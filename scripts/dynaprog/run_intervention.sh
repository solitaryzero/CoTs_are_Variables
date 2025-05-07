python ./src/dynaprog/run_intervention.py \
    --base_model ./models/Qwen2.5-1.5B \
    --save_model_path ./models/dynaprog/full_cot/Qwen2.5-1.5B \
    --data_path ./results/dynaprog/intervened/Qwen2.5-1.5B/intervened.json \
    --save_result_path ./results/dynaprog/intervened/Qwen2.5-1.5B \
    --max_new_tokens 1024 \
    --seed 42