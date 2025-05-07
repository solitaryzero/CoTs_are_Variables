import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_context(rc={"axes.labelsize":20, "legend.fontsize":16, "legend.title_fontsize":16})

def main(args):
    df_dict = {
        'task': [],
        'accuracy': [],
    }

    with open(args.data_path_multi, 'r', encoding='utf-8') as fin:
        js = json.load(fin)
        df_dict['task'].append('Multi')
        df_dict['accuracy'].append(js['accuracy'])

    with open(args.data_path_dp, 'r', encoding='utf-8') as fin:
        js = json.load(fin)
        df_dict['task'].append('DP')
        df_dict['accuracy'].append(js['accuracy'])

    df = pd.DataFrame.from_dict(df_dict)

    fig = sns.barplot(x="task", y="accuracy", data=df, palette="Blues_d")
    if not(os.path.exists(args.out_path)):
        os.makedirs(args.out_path)
    fig.figure.savefig(
        os.path.join(
            args.out_path, 
            f'intervene_accuracy.{args.format}'), 
            format=args.format,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_multi', type=str, required=True)
    parser.add_argument('--data_path_dp', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')

    args = parser.parse_args()
    main(args)