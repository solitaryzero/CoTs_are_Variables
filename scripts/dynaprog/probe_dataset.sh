python ./src/dynaprog/probing_utils.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/dynaprog/dp_data.json \
    --latent_model_path ./models/dynaprog/latent/Qwen2.5-1.5B \
    --save_result_path ./data/dynaprog/probe/Qwen2.5-1.5B \
    --latent_dim 50

python ./src/dynaprog/probing_utils.py \
    --base_model ./models/Qwen2.5-1.5B \
    --data_path ./data/dynaprog/dp_data.json \
    --latent_model_path ./models/dynaprog/latent_row/Qwen2.5-1.5B \
    --save_result_path ./data/dynaprog/probe_row/Qwen2.5-1.5B \
    --latent_dim 50