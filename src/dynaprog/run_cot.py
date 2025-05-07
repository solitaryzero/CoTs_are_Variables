import os
import json
import argparse
from tqdm import tqdm
import gc
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, GenerationConfig

from utils import parse_dataset_name, process_cot_dataset

def run(
    args,
    train_dataset,
    test_dataset,
    save_model_path,
):  
    if (args.do_train):
        model = AutoModelForCausalLM.from_pretrained(args.base_model).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.save_model_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Train
    if (args.do_train):
        dataset = process_cot_dataset(train_dataset)
        def tokenize_function(element):
            full_tokens = tokenizer(
                element['prompt']+'<|endoftext|>',
                truncation=True,
                max_length=2048,
                add_special_tokens=True,
            )
            query_tokens = tokenizer(
                element['query'],
                truncation=True,
                max_length=2048,
                add_special_tokens=False,
            )

            full_tokens['query_ids'] = query_tokens['input_ids']

            return full_tokens

        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=['query', 'prompt', 'id'],
        )

        if (len(dataset) < 10000):
            epoch = 10
        else:
            epoch = args.epoch

        train_params = TrainingArguments(
            output_dir=save_model_path,
            num_train_epochs=epoch,
            per_device_train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=1,
            optim="adamw_torch",
            save_strategy='epoch',
            logging_steps=args.logging_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            fp16=args.fp16,
            bf16=args.bf16,
            max_grad_norm=args.grad_clip,
            max_steps=-1,
            warmup_ratio=0,
            group_by_length=False,
            remove_unused_columns=False,
            lr_scheduler_type="constant",
            report_to=None,
        )

        # Train
        def collate(elements):
            tokenlist = [e["input_ids"] for e in elements]
            query_lengths = [len(e['query_ids']) for e in elements]
            tokens_maxlen = max([len(t) for t in tokenlist])

            input_ids, labels, attention_masks = [], [], []
            for tokens, query_len in zip(tokenlist, query_lengths):
                pad_len = tokens_maxlen - len(tokens)
                input_ids.append(tokens + [tokenizer.pad_token_id]*pad_len)
                attention_masks.append([1]*len(tokens) + [0]*pad_len)
                labels.append([-100]*query_len + tokens[query_len:] + [-100]*pad_len)

            batch = {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attention_masks),
            }
            return batch

        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            data_collator=collate,
            args=train_params,
            train_dataset=tokenized_dataset,
            eval_dataset=None,
            compute_metrics=None,
        )

        trainer.train()
        model.save_pretrained(save_model_path)

    # Eval
    correct, total = 0, 0
    generation_config = GenerationConfig(
        num_beams=1,
        do_sample=False,
    )

    all_predictions = []
    for entry in tqdm(test_dataset, desc='Eval'):
        prompt = entry['query']
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
        try:
            prediction = int(decoded.strip().split('Result: ')[-1])
        except:
            prediction = -1
        
        # print(prompt)
        # print(decoded)
        # print(f'No. tokens: {len(generation_outputs.sequences[0])}')
        # print(golden)
        # print(prediction)
        # print(golden == prediction)
        # print('=====')
        # input()

        js = {
            'id': entry['id'],
            'query': entry['query'],
            'output': decoded,
            'golden': golden,
            'prediction': prediction,
        }
        all_predictions.append(js)

        if (golden == prediction):
            correct += 1
        total += 1

    result = {
        'correct': correct,
        'total': total,
        'accuracy': correct/total,
    }

    return result, model, all_predictions

def main(args):
    if (args.run_all):
        files = os.listdir(args.data_path)
        all_results = {}
        if not(os.path.exists(args.save_result_path)):
            os.makedirs(args.save_result_path)

        for fn in files:
            data_path = os.path.join(args.data_path, fn)
            da, db = parse_dataset_name(data_path)
            if (da > db):
                continue

            # resume from interruption
            if (args.resume_interruption):
                out_path = os.path.join(args.save_result_path, f'{da}_mul_{db}_scores.json')
                if (os.path.isfile(out_path)): # executed beforehand
                    continue

            print(f'=== Running {da}_mul_{db} ===')

            with open(data_path, 'r', encoding='utf-8') as fin:
                js = json.load(fin)
                train_dataset, test_dataset = js['train'], js['test']

            save_model_path = os.path.join(args.save_model_path, f'{da}_mul_{db}')
            if not(os.path.exists(save_model_path)):
                os.makedirs(save_model_path)
            result, model, _ = run(args, train_dataset, test_dataset, save_model_path)
            key = f'{da},{db}'
            all_results[key] = result

            out_path = os.path.join(args.save_result_path, f'{da}_mul_{db}_scores.json')
            with open(out_path, 'w', encoding='utf-8') as fout:
                json.dump(result, fout)

            del model
            gc.collect()

        out_path = os.path.join(args.save_result_path, 'all_scores.json')
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(all_results, fout)
    else:
        with open(args.data_path, 'r', encoding='utf-8') as fin:
            js = json.load(fin)
            train_dataset, test_dataset = js['train'], js['test']
        
        save_model_path = args.save_model_path
        if not(os.path.exists(save_model_path)):
            os.makedirs(save_model_path)
        result, model, predictions = run(args, train_dataset, test_dataset, save_model_path)

        if not(os.path.exists(args.save_result_path)):
            os.makedirs(args.save_result_path)
        out_path = os.path.join(args.save_result_path, f'scores.json')
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(result, fout)

        out_path = os.path.join(args.save_result_path, f'predictions.json')
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(predictions, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path args
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_model_path', type=str)
    parser.add_argument('--save_result_path', type=str, required=True)

    # Model args
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')

    # Training args
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--logging_steps', type=int, default=2000)

    # Misc
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run_all', action='store_true')
    parser.add_argument('--resume_interruption', action='store_true')

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(42)

    main(args)