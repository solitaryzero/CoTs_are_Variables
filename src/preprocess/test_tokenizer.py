from transformers import AutoTokenizer

model_path = './models/Qwen2.5-1.5B'
tokenizer = AutoTokenizer.from_pretrained(model_path)

s = '1234*5678'
print(tokenizer.tokenize(s, add_special_tokens=True))