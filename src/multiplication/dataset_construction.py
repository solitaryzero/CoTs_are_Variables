import random
import argparse
import os
import numpy as np
import json
from tqdm import tqdm


def main(args):
    random.seed(args.seed)

    da, db = args.num_digit_a, args.num_digit_b
    max_d = da+db
    if (max_d <= np.log10(args.n)):
        n = 10**(max_d)
    else:
        n = args.n

    all_data = []
    for i in tqdm(range(n)):
        a = random.randint(10**(da-1), (10**da)-1)
        b = random.randint(10**(db-1), (10**db)-1)
        res = a*b
        js = {
            'id': i,
            'a': a,
            'b': b,
            'golden': res,
            'query': f'{a}*{b}=',
            'prompt': f'{a}*{b}={res}',
        }
        all_data.append(js)

    train_n = int(args.train_ratio * n)
    dataset = {
        'train': all_data[:train_n],
        'test': all_data[train_n:],
    }

    if not(os.path.exists(args.save_path)):
        os.makedirs(args.save_path)
    out_path = os.path.join(args.save_path, f'{da}_mul_{db}.json')
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(dataset, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_digit_a', type=int, default=4)
    parser.add_argument('--num_digit_b', type=int, default=4)
    parser.add_argument('--n', type=int, default=100000)
    parser.add_argument('--save_path', type=str, default='./data/multiplication')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    for a in range(1, 6):
        for b in range(1, 6):
            args.num_digit_a = a
            args.num_digit_b = b
            main(args)