# CUDA_VISIBLE_DEVICES=6 python ./src/misc/run_unsupervised_latent.py \
#     --base_model ./models/Qwen2.5-1.5B \
#     --data_path ./data/full_dynaprog/2_mul_3.json \
#     --save_model_path ./models/misc/unsupervised/Qwen2.5-1.5B/2_mul_3_none \
#     --save_result_path ./results/misc/unsupervised/Qwen2.5-1.5B \
#     --do_train \
#     --compress none \
#     --learning_rate 1e-5 \
#     --weight_decay 0 \
#     --grad_clip 1.0 \
#     --train_batch_size 4 \
#     --epoch 1 \
#     --logging_steps 200 \
#     --max_new_tokens 1024 \
#     --seed 42

CUDA_VISIBLE_DEVICES=6 python ./src/misc/run_unsupervised_latent.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/full_dynaprog/4_mul_4.json \
    --save_model_path ./models/misc/unsupervised/Qwen2.5-1.5B/4_mul_4_none \
    --save_result_path ./results/misc/unsupervised/4_mul_4/Qwen2.5-1.5B \
    --do_train \
    --compress none \
    --learning_rate 1e-5 \
    --weight_decay 0 \
    --grad_clip 1.0 \
    --train_batch_size 4 \
    --epoch 1 \
    --logging_steps 200 \
    --max_new_tokens 1024 \
    --seed 42