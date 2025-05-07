import os
import sys
import json
from tqdm import tqdm
import argparse
import pickle
import logging

import numpy as np
import torch
import torch.nn as nn
from utils import process_latent_dataset
from model import LatentQwen2, LatentTokenizer
from transformers import get_linear_schedule_with_warmup


def collect_embeds(
    tokenizer,
    model,
    dummy_latent,
    dataset,
    save_path,
):
    n_layers = model.config.num_hidden_layers
    all_embeds = [[] for _ in range(n_layers)]
    metadata = []

    for entry in tqdm(dataset):
        metadata.append({
            'id': entry['id'],
            'query': entry['query'],
            'golden': entry['golden'],
        })

        inputs = {
            'input_ids': None,
            'attention_mask': None,
            'latent_embeds': None,
            'token_types': None
        }
        tokenize_result = tokenizer(
            entry['prompt']+'<|endoftext|>',
            truncation=True,
            max_length=2048,
            add_special_tokens=True,
        )
        inputs['input_ids'] = tokenize_result['input_ids']
        inputs['attention_mask'] = tokenize_result['attention_mask']

        # BNE features
        full_latent_embed = []
        token_type = []
        current = 0
        for token in inputs['input_ids']:
            if (token == tokenizer.latent_token_id):
                full_latent_embed.append(entry['latent_embeds'][current])
                token_type.append(True)
                current += 1
            else:
                full_latent_embed.append(dummy_latent)
                token_type.append(False)

        inputs['latent_embeds'] = full_latent_embed
        inputs['token_types'] = token_type

        for key in inputs:
            inputs[key] = torch.tensor(inputs[key]).to(model.device)
        
        with torch.no_grad():
            model_outputs = model(
                input_ids=inputs['input_ids'].unsqueeze(0),
                attention_mask=inputs['attention_mask'].unsqueeze(0),
                latent_embeds=inputs['latent_embeds'].float().unsqueeze(0),
                token_types=inputs['token_types'].unsqueeze(0),
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = model_outputs['hidden_states'][1:] # skip embedding layer
            assert len(hidden_states) == n_layers
            for layer, hs in enumerate(hidden_states):
                token_mask = inputs['token_types'][1:]
                shifted_hidden = hs[0, :-1, :]
                x_embeds = shifted_hidden[token_mask].detach().cpu().numpy()
                y_embeds = np.array(entry['latent_embeds'])
                assert (x_embeds.shape[0] == y_embeds.shape[0])
                all_embeds[layer].append({
                    'x': x_embeds,
                    'y': y_embeds,
                })

    for layer in range(n_layers):
        with open(os.path.join(save_path, f'{layer}.embeds'), 'wb') as fout:
            pickle.dump(all_embeds[layer], fout)

    with open(os.path.join(save_path, f'metadata.json'), 'w', encoding='utf-8') as fout:
        json.dump(metadata, fout)


def make_probe_dataset(args):
    tokenizer = LatentTokenizer.from_pretrained(args.base_model)
    model = LatentQwen2.from_pretrained(args.latent_model_path, latent_dim=args.latent_dim).cuda()
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    with open(args.data_path, 'r', encoding='utf-8') as fin:
        js = json.load(fin)
        train_dataset, test_dataset = js['train'], js['test']
        train_dataset = process_latent_dataset(
            train_dataset,
            tokenizer,
            args.latent_dim,
        )
        test_dataset = process_latent_dataset(
            test_dataset,
            tokenizer,
            args.latent_dim,
        )

    dummy_latent = [0 for _ in range(args.latent_dim)]

    train_out_path = os.path.join(args.save_result_path, 'train')
    if not(os.path.exists(train_out_path)):
        os.makedirs(train_out_path)
    collect_embeds(
        tokenizer,
        model,
        dummy_latent,
        train_dataset,
        train_out_path,
    )

    test_out_path = os.path.join(args.save_result_path, 'test')
    if not(os.path.exists(test_out_path)):
        os.makedirs(test_out_path)
    collect_embeds(
        tokenizer,
        model,
        dummy_latent,
        test_dataset,
        test_out_path,
    )



class LinearProber(nn.Module):
    def __init__(self, params):
        super(LinearProber, self).__init__()
        self.probe_layer = nn.Linear(params['hidden_size'], params['target_dim'])

        self.criterion = nn.BCEWithLogitsLoss()

        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and params.get('cuda', False) else "cpu"
        )
        
        if (params['load_model_path'] is not None):
            self.load_model(params['load_model_path'])

    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.load_state_dict(state_dict)


    def predict(self,
        hidden_states,
    ):
        predictions = self.probe_layer(hidden_states)
        predictions = torch.sigmoid(predictions)
        predictions = (predictions >= 0.5).long()
        return predictions


    def forward(self, 
        hidden_states,
        labels,
    ):  
        predictions = self.probe_layer(hidden_states)
        loss = self.criterion(predictions, labels)

        return loss, predictions
    

def setup_logger(name, save_dir, filename="log.txt", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not(os.path.exists(save_dir)):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_model(model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_model_file = os.path.join(output_dir, 'checkpoint.bin')
    torch.save(model.state_dict(), output_model_file)


def ellipse(lst, max_display=5, sep='|'):
    """
    Like join, but possibly inserts an ellipsis.
    :param lst: The list to join on
    :param int max_display: the number of items to display for ellipsing.
        If -1, shows all items
    :param string sep: the delimiter to join on
    """
    # copy the list (or force it to a list if it's a set)
    choices = list(lst)
    # insert the ellipsis if necessary
    if max_display > 0 and len(choices) > max_display:
        ellipsis = '...and {} more'.format(len(choices) - max_display)
        choices = choices[:max_display] + [ellipsis]
    return sep.join(str(c) for c in choices)


def get_optimizer(model, params):
    parameters_with_decay = []
    parameters_with_decay_names = []
    parameters_without_decay = []
    parameters_without_decay_names = []
    no_decay = ['bias', 'gamma', 'beta']

    for n, p in model.named_parameters():
        if any(t in n for t in no_decay):
            parameters_without_decay.append(p)
            parameters_without_decay_names.append(n)
        else:
            parameters_with_decay.append(p)
            parameters_with_decay_names.append(n)

    print('The following parameters will be optimized WITH decay:')
    print(ellipse(parameters_with_decay_names, 5, ' , '))
    print('The following parameters will be optimized WITHOUT decay:')
    print(ellipse(parameters_without_decay_names, 5, ' , '))

    optimizer_grouped_parameters = [
        {'params': parameters_with_decay, 'weight_decay': 0.01},
        {'params': parameters_without_decay, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=params['learning_rate'],
    )

    return optimizer


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params['train_batch_size']
    epochs = params['epoch']

    num_train_steps = int(len_train_data / batch_size) * epochs
    num_warmup_steps = int(num_train_steps * params['warmup_proportion'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_train_steps,
    )
    if (logger):
        logger.info("Num optimization steps = %d" % num_train_steps)
        logger.info("Num warmup steps = %d", num_warmup_steps)
    return scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Path args
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--latent_model_path', type=str)
    parser.add_argument('--save_result_path', type=str, required=True)

    # Model args
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--latent_dim', type=int, default=50)

    args = parser.parse_args()

    make_probe_dataset(args)