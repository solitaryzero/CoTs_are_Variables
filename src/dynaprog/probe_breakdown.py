import os
import argparse
import torch
import json
from tqdm import tqdm, trange
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from probing_utils import LinearProber, setup_logger

sns.set_context(rc={"axes.labelsize":20, "legend.fontsize":16, "legend.title_fontsize":16})

def read_metadata(args, spl):
    with open(os.path.join(args.data_path, spl, 'metadata.json')) as fin:
        metadata = json.load(fin)
    return metadata


def read_embeds(args, spl, layer):
    dataset = []
    embed_path = os.path.join(args.data_path, spl)
    with open(os.path.join(embed_path, f'{layer}.embeds'), 'rb') as fin:
        while True:
            try:
                d = pickle.load(fin)
                dataset.append(d)
            except EOFError:
                break
    return dataset


def cal_accuracy(predictions, labels):
    acc_dict = {}
    for i in range(labels.shape[0]):
        scale = int(np.sum(labels[i]))
        if (scale not in acc_dict):
            acc_dict[scale] = {
                'correct': 0,
                'total': 0,
            }
        
        correct = np.all(predictions[i] == labels[i])
        if (correct):
            acc_dict[scale]['correct'] += 1
        acc_dict[scale]['total'] += 1
    
    return acc_dict


def evaluate_breakdown(
    model, 
    X_test, 
    Y_test, 
):
    device = 'cuda'
    model = model.to(device)
    with torch.no_grad():
        iter_ = tqdm(zip(X_test, Y_test), desc="Evaluation")
        results = {}
    
        for X, Y in iter_:
            hidden_states, labels = torch.FloatTensor(X), torch.FloatTensor(Y)
            hidden_states, labels = hidden_states.to(device), labels.to(device)
                
            prediction = model.predict(hidden_states.unsqueeze(0))
            prediction = prediction.squeeze(0).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            acc_dict = cal_accuracy(prediction, labels)

            for scale in acc_dict:
                if (scale not in results):
                    results[scale] = {
                        'correct': 0,
                        'total': 0,
                    }
                for k in acc_dict[scale]:
                    results[scale][k] += acc_dict[scale][k]

        return results


def draw_breakdown(results):
    df_dict = {
        'layer': [],
        'Digit Scale': [],
        'accuracy': [],
    }

    for layer in results:
        for scale in results[layer]:
            acc = results[layer][scale]['correct'] / results[layer][scale]['total']

            df_dict['layer'].append(int(layer))
            df_dict['Digit Scale'].append(int(scale))
            df_dict['accuracy'].append(acc)

    df = pd.DataFrame.from_dict(df_dict)
    df = df.pivot(index='Digit Scale', columns='layer', values='accuracy')

    plt.figure(figsize=(6,4), constrained_layout=True)
    fig = sns.heatmap(
        df, 
        annot=False, 
        # cmap='vlag_r',
        cmap='Blues',
    )
    fig.set_title("Accuracy Breakdown", fontsize=20)

    if not(os.path.exists(args.output_path)):
        os.makedirs(args.output_path)
    fig.figure.savefig(
        os.path.join(
            args.output_path, 
            f'probe_breakdown.{args.format}'), 
            format=args.format,
    )


def main(args):
    params = args.__dict__

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if (args.quickdraw):
        with open(os.path.join(args.output_path, 'breakdown_results.json'), 'r', encoding='utf-8') as fin:
            all_results = json.load(fin)
    else:
        all_results = {}

        num_layers = args.num_layers
        for layer in tqdm(range(num_layers), desc='Layer'):
            model_path = os.path.join(args.model_path, f'layer_{layer}', 'checkpoint.bin')

            test_dataset = read_embeds(args, 'test', layer)
            all_X, all_Y = [], []
            for entry in test_dataset:
                all_X.append(entry['x'])
                all_Y.append(entry['y'])
            X_test, Y_test = np.stack(all_X), np.stack(all_Y)
            
            model = LinearProber(params)
            model.load_model(model_path)
            layer_results = evaluate_breakdown(
                model, X_test, Y_test,
            )

            all_results[layer] = layer_results

        if not(os.path.exists(args.output_path)):
            os.makedirs(args.output_path)

        with open(os.path.join(args.output_path, 'breakdown_results.json'), 'w', encoding='utf-8') as fout:
            json.dump(all_results, fout)

    draw_breakdown(all_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--num_layers', type=int, default=28)

    # model arguments
    parser.add_argument('--hidden_size', type=int, default=1536)
    parser.add_argument('--target_dim', type=int, default=30)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    # picture arguments
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')
    parser.add_argument('--quickdraw', action='store_true')
    
    args = parser.parse_args()
    print(args)

    main(args)