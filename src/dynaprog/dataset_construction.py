import random
import argparse
import os
import json
from tqdm import tqdm

from utils import formulate_prompt


def main(args):
    random.seed(args.seed)

    all_data = []
    for index in tqdm(range(args.n)):
        table = []
        for i in range(args.n_rows):
            table.append([])
            for j in range(args.n_cols):
                v = random.randint(args.lower_bound, args.higher_bound)
                table[i].append(v)

        query, prompt, answer = formulate_prompt(table)
        js = {
            'id': index,
            'table': table,
            'golden': answer,
            'query': query,
            'prompt': prompt,
        }
        all_data.append(js)

    train_n = int(args.train_ratio * args.n)
    dataset = {
        'train': all_data[:train_n],
        'test': all_data[train_n:],
    }

    if not(os.path.exists(args.save_path)):
        os.makedirs(args.save_path)
    
    if (args.generate_all):
        out_path = os.path.join(args.save_path, f'{args.n_rows}_mul_{args.n_cols}.json')
    else:
        out_path = os.path.join(args.save_path, f'dp_data.json')
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(dataset, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_rows', type=int, default=5)
    parser.add_argument('--n_cols', type=int, default=5)
    parser.add_argument('--n', type=int, default=100000)
    parser.add_argument('--lower_bound', type=int, default=1)
    parser.add_argument('--higher_bound', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='./data/dynaprog')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generate_all', action='store_true')
    args = parser.parse_args()

    if (args.generate_all):
        for a in range(1, 6):
            for b in range(1, 6):
                args.n_rows = a
                args.n_cols = b
                main(args)
    else:
        main(args)