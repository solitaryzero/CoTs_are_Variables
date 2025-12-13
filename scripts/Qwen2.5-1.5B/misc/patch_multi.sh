# CUDA_VISIBLE_DEVICES=4 python ./src/multiplication/run_cot.py \
#     --base_model ./models/Qwen2.5-1.5B \
#     --data_path ./data/multiplication/1_mul_1.json \
#     --save_model_path ./models/full_multiplication/full_cot/Qwen2.5-1.5B \
#     --save_result_path ./results/full_multiplication/full_cot/Qwen2.5-1.5B \
#     --do_train \
#     --learning_rate 1e-5 \
#     --weight_decay 0 \
#     --grad_clip 1.0 \
#     --train_batch_size 4 \
#     --epoch 10 \
#     --logging_steps 200 \
#     --cot_type full \
#     --max_new_tokens 1024 \
#     --seed 42

CUDA_VISIBLE_DEVICES=4 python ./src/multiplication/run_cot.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/multiplication/4_mul_5.json \
    --save_model_path ./models/full_multiplication/compressed_cot/Qwen2.5-1.5B \
    --save_result_path ./results/full_multiplication/compressed_cot/Qwen2.5-1.5B \
    --do_train \
    --learning_rate 1e-5 \
    --weight_decay 0 \
    --grad_clip 1.0 \
    --train_batch_size 4 \
    --epoch 1 \
    --bf16 \
    --logging_steps 200 \
    --cot_type compressed \
    --max_new_tokens 1024 \
    --seed 42

# CUDA_VISIBLE_DEVICES=4 python ./src/multiplication/run_latent.py \
#     --base_model ./models/Qwen2.5-1.5B \
#     --data_path ./data/multiplication/1_mul_1.json \
#     --save_model_path ./models/full_multiplication/latent/Qwen2.5-1.5B \
#     --save_result_path ./results/full_multiplication/latent/Qwen2.5-1.5B \
#     --latent_dim 20 \
#     --do_train \
#     --learning_rate 1e-5 \
#     --weight_decay 0 \
#     --grad_clip 1.0 \
#     --train_batch_size 4 \
#     --epoch 10 \
#     --bf16 \
#     --resume_interruption \
#     --logging_steps 200 \
#     --max_new_tokens 1024 \
#     --seed 42

# CUDA_VISIBLE_DEVICES=6 python ./src/multiplication/run_raw.py \
#     --base_model ./models/Qwen2.5-1.5B \
#     --data_path ./data/multiplication/1_mul_2.json \
#     --save_model_path ./models/full_multiplication/plain/Qwen2.5-1.5B \
#     --save_result_path ./results/full_multiplication/plain/Qwen2.5-1.5B \
#     --do_train \
#     --learning_rate 1e-5 \
#     --weight_decay 0 \
#     --grad_clip 1.0 \
#     --train_batch_size 8 \
#     --epoch 10 \
#     --logging_steps 200 \
#     --max_new_tokens 20 \
#     --bf16 \
#     --seed 42