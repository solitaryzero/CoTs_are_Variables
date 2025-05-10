python ./src/dynaprog/probe_breakdown.py \
    --data_path ./data/dynaprog/probe_row/Qwen2.5-1.5B \
    --model_path ./models/dynaprog/probe_row/Qwen2.5-1.5B \
    --output_path ./results/dynaprog/probe_breakdown/Qwen2.5-1.5B \
    --num_layers 28 \
    --hidden_size 1536 \
    --target_dim 50 \
    --cuda \
    --format pdf \
    --seed 42