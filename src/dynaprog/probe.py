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

from probing_utils import LinearProber, setup_logger, save_model, get_optimizer, get_scheduler


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
    total_count = labels.shape[0]*labels.shape[1]*labels.shape[2]
    correct_count = np.sum(predictions == labels)
    total_entry = labels.shape[0]*labels.shape[1]
    entry = np.all(predictions == labels, axis=-1)
    correct_entry = np.sum(entry)
    
    return total_count, correct_count, total_entry, correct_entry


def train(
    params,
    output_path,
    logger,
    X_train, 
    Y_train, 
    X_test, 
    Y_test,
):
    X_train, Y_train = torch.FloatTensor(X_train), torch.FloatTensor(Y_train)
    X_test, Y_test = torch.FloatTensor(X_test), torch.FloatTensor(Y_test)

    train_tensor_data = TensorDataset(X_train, Y_train)
    if args.shuffle:
        train_sampler = RandomSampler(train_tensor_data)
    else:
        train_sampler = SequentialSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=params['train_batch_size']
    )

    test_tensor_data = TensorDataset(X_test, Y_test)
    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(
        test_tensor_data, sampler=test_sampler, batch_size=params['eval_batch_size']
    )

    model = LinearProber(params)
    model.to(model.device)
    device = model.device

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

    best_epoch_idx = -1
    best_score = -1
    epoch_results = []
    num_train_epochs = params["epoch"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        model.train()
        tr_loss = 0
        results = None
        iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            hidden_states, labels = batch
                
            loss, predictions = model(
                hidden_states,
                labels,
            )

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"]) == 0:
                if (logger):
                    logger.info(
                        "Step %d - epoch %d average loss: %.4f; loss: %.4f" %(
                            step,
                            epoch_idx,
                            tr_loss / (params["print_interval"]),
                            loss.item(),
                        )
                    )
                tr_loss = 0
                # print(predictions)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), params["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (logger):
            logger.info("***** Saving fine - tuned model *****")

        epoch_output_folder_path = os.path.join(
            output_path, "epoch_%d" %(epoch_idx)
        )
        save_model(model, epoch_output_folder_path)

        # evaluate
        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            model, test_dataloader, device=device, logger=logger
        )
        with open(output_eval_file, 'w', encoding='utf-8') as fout:
            fout.write(json.dumps(results, indent=4))

        epoch_results.append(results)
        ls = [best_score, results["token_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        if (logger):
            logger.info("\n")

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["load_model_path"] = os.path.join(
        output_path, 
        "epoch_%d" %(best_epoch_idx),
        'checkpoint.bin',
    )

    model = LinearProber(params)
    model.to(model.device)
    save_model(model, output_path)

    print('Best results: ')
    print(epoch_results[best_epoch_idx])
    with open(os.path.join(output_path, 'best_result.json'), 'w', encoding='utf-8') as fout:
        json.dump(epoch_results[best_epoch_idx], fout)

    return model


def evaluate(
    model, 
    eval_dataloader, 
    device, 
    logger,
):
    with torch.no_grad():
        model.eval()
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

        results = {}

        total_count, correct_count, total_entry, correct_entry = 0, 0, 0, 0
    
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            hidden_states, labels = batch
                
            prediction = model.predict(hidden_states)

            prediction = prediction.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            tc, cc, te, ce = cal_accuracy(prediction, labels)

            total_count += tc
            correct_count += cc
            total_entry += te
            correct_entry += ce

        element_accuracy = correct_count / total_count
        entry_accuracy = correct_entry / total_entry
        logger.info("Element accuracy: %.5f" % element_accuracy)
        logger.info("Token accuracy: %.5f" % entry_accuracy)
        results["element_accuracy"] = element_accuracy
        results["token_accuracy"] = entry_accuracy
        return results


def main(args):
    params = args.__dict__

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = setup_logger('Prober', params['output_path'])

    # training
    num_layers = args.num_layers

    for layer in tqdm(range(num_layers), desc='Layer'):
        output_path = os.path.join(args.output_path, f'layer_{layer}')
        if not(os.path.exists(output_path)):
            os.makedirs(output_path)

        train_dataset = read_embeds(args, 'train', layer)
        all_X, all_Y = [], []
        for entry in train_dataset:
            all_X.append(entry['x'])
            all_Y.append(entry['y'])
        X_train, Y_train = np.stack(all_X), np.stack(all_Y)

        test_dataset = read_embeds(args, 'test', layer)
        all_X, all_Y = [], []
        for entry in test_dataset:
            all_X.append(entry['x'])
            all_Y.append(entry['y'])
        X_test, Y_test = np.stack(all_X), np.stack(all_Y)
        
        model = train(
            params=params, 
            output_path=output_path, 
            logger=logger, 
            X_train=X_train, 
            Y_train=Y_train, 
            X_test=X_test, 
            Y_test=Y_test,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--output_path', type=str, default='./model')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--num_layers', type=int, default=28)

    # model arguments
    parser.add_argument('--hidden_size', type=int, default=1536)
    parser.add_argument('--target_dim', type=int, default=30)
    parser.add_argument('--cuda', action='store_true')

    # training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--print_interval', type=int, default=200)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    print(args)

    main(args)