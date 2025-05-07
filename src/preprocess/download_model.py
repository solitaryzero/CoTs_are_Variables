from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B"
model_path = './models/Qwen2.5-1.5B'

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
model.save_pretrained(model_path)