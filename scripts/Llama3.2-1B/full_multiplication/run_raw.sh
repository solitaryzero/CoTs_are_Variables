CUDA_VISIBLE_DEVICES=7 python ./src/multiplication/run_raw.py \
    --base_model ./models/Llama-3.2-1B \
    --data_path ./data/multiplication \
    --save_model_path ./models/full_multiplication/plain/Llama-3.2-1B \
    --save_result_path ./results/full_multiplication/raw/Llama-3.2-1B \
    --max_new_tokens 50 \
    --bf16 \
    --run_all \
    --seed 42