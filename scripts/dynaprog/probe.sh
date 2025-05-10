python ./src/dynaprog/probe.py \
    --data_path ./data/dynaprog/probe/Qwen2.5-1.5B \
    --output_path ./models/dynaprog/probe/Qwen2.5-1.5B \
    --num_layers 28 \
    --hidden_size 1536 \
    --target_dim 50 \
    --cuda \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --epoch 4 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --print_interval 2000 \
    --shuffle 

python ./src/dynaprog/probe.py \
    --data_path ./data/dynaprog/probe_row/Qwen2.5-1.5B \
    --output_path ./models/dynaprog/probe_row/Qwen2.5-1.5B \
    --num_layers 28 \
    --hidden_size 1536 \
    --target_dim 50 \
    --cuda \
    --learning_rate 1e-3 \
    --max_grad_norm 1.0 \
    --epoch 4 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --print_interval 2000 \
    --shuffle 