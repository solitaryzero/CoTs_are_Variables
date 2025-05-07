import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_context(rc={"axes.labelsize":20, "legend.fontsize":16, "legend.title_fontsize":16})

def main(args):
    if (args.from_separate_files):
        df_dict = {
            'digit_a': [],
            'digit_b': [],
            'accuracy': [],
        }
        files = os.listdir(args.data_path)
        for file_name in files:
            if ('_mul_' not in file_name):
                continue
            segs = file_name.split('_mul_')
            da = int(segs[0])
            db = int(segs[1].split('_')[0])
            if (da > db):
                continue

            with open(os.path.join(args.data_path, file_name), 'r', encoding='utf-8') as fin:
                js = json.load(fin)
                acc = js['accuracy']

            df_dict['digit_a'].append(da)
            df_dict['digit_b'].append(db)
            df_dict['accuracy'].append(acc)
    else:
        with open(args.data_path, 'r', encoding='utf-8') as fin:
            js = json.load(fin)

            df_dict = {
                'digit_a': [],
                'digit_b': [],
                'accuracy': [],
            }
            for key in js:
                a, b = key.split(',')
                a, b = int(a), int(b)
                acc = js[key]['accuracy']
                df_dict['digit_a'].append(a)
                df_dict['digit_b'].append(b)
                df_dict['accuracy'].append(acc)

    df = pd.DataFrame.from_dict(df_dict)
    df = df.pivot(index='digit_b', columns='digit_a', values='accuracy')

    plt.figure(figsize=(6,6), constrained_layout=True)
    if (args.prompt_type == 'plain'):
        mask = np.invert(np.tril(np.ones((5,5), dtype=bool)))
    else:
        mask = None
    fig = sns.heatmap(
        df, 
        mask=mask,
        vmin=0.0,
        vmax=1.0,
        annot=True, 
        annot_kws={
            "size": 18,
            "color": "black",
        },
        fmt='.2f', 
        cmap='vlag_r'
    )
    fig.set_title("Accuracy", fontsize=20)
    
    if not(os.path.exists(args.out_path)):
        os.makedirs(args.out_path)
    fig.figure.savefig(
        os.path.join(
            args.out_path, 
            f'{args.prompt_type}_accuracy.{args.format}'), 
            format=args.format,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--prompt_type', type=str)
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')
    parser.add_argument('--from_separate_files', action='store_true')

    args = parser.parse_args()
    main(args)