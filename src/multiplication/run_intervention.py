import os
import json
import argparse
from tqdm import tqdm
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def run(
    args,
):  
    model = AutoModelForCausalLM.from_pretrained(args.save_model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Eval
    success, total = 0, 0
    generation_config = GenerationConfig(
        num_beams=1,
        do_sample=False,
    )

    with open(args.data_path, 'r', encoding='utf-8') as fin:
        test_dataset = json.load(fin)

    all_predictions = []
    for entry in tqdm(test_dataset, desc='Eval'):
        prompt = entry['partial_cot']
        inputs = tokenizer(
            prompt, 
            truncation=True,
            max_length=2048,
            add_special_tokens=True,
        )
        input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0).cuda()
        attention_mask = torch.tensor(inputs["attention_mask"]).unsqueeze(0).cuda()
        generation_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            max_new_tokens=args.max_new_tokens,
        )

        decoded = tokenizer.decode(generation_outputs.sequences[0], skip_special_tokens=True)
        golden = entry['golden']
        raw_prediction = entry['raw_prediction']
        simulated_prediction = entry['simulated_prediction']
        try:
            prediction = int(decoded.strip().split('Result: ')[-1])
        except:
            prediction = -1
        
        # print(prompt)
        # print(decoded)
        # print('=== Results ===')
        # print('Golden: ', golden)
        # print('Raw prediction: ', raw_prediction)
        # print('Simulated prediction: ', simulated_prediction)
        # print('Intervened prediction: ', prediction)
        # print('==========')
        # input()

        js = {
            'id': entry['id'],
            'query': entry['query'],
            'raw_output': entry['raw_output'],
            'partial_cot': prompt,
            'modification': entry['modification'],
            'output': decoded,
            'golden': golden,
            'raw_prediction': raw_prediction,
            'simulated_prediction': simulated_prediction,
            'intervened_prediction': prediction,
        }
        all_predictions.append(js)

        if (simulated_prediction == prediction):
            success += 1
        total += 1

    result = {
        'success': success,
        'total': total,
        'accuracy': success/total,
    }

    return result, all_predictions


def main(args):
    result, predictions = run(args)
    out_path = os.path.join(args.save_result_path, f'intervene_scores.json')
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(result, fout)

    out_path = os.path.join(args.save_result_path, f'intervene_predictions.json')
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(predictions, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path args
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--save_model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_result_path', type=str, required=True)

    # Model args
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')

    # Misc
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(42)
    
    main(args)