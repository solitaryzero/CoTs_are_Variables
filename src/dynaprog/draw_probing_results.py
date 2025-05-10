import os
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context(rc={"axes.labelsize":20, "legend.fontsize":16, "legend.title_fontsize":16})

def main(args, tasks):
    df_dict = {
        'Value': [],
        'Accuracy Metric': [],
        'Layer': [],
        'Compression': [],
    }

    for task in tasks:
        model_path = tasks[task]
        layer_folders = os.listdir(model_path)
        for folder in layer_folders:
            if (folder.startswith('layer')):
                layer = int(folder.split('_')[-1])
                file_path = os.path.join(model_path, folder, 'best_result.json')
                with open(file_path, 'r', encoding='utf-8') as fin:
                    res = json.load(fin)

                    df_dict['Value'].append(res['element_accuracy'])
                    df_dict['Accuracy Metric'].append('element')
                    df_dict['Layer'].append(layer)
                    df_dict['Compression'].append(task)

                    df_dict['Value'].append(res['token_accuracy'])
                    df_dict['Accuracy Metric'].append('token')
                    df_dict['Layer'].append(layer)
                    df_dict['Compression'].append(task)

    df = pd.DataFrame.from_dict(df_dict)
    plt.figure(figsize=(6,4), constrained_layout=True)
    fig = sns.lineplot(
        x='Layer',
        y='Value',
        hue='Accuracy Metric',
        style='Compression',
        data=df,
    )
    fig.set_title("Probe Accuracy", fontsize=20)

    if not(os.path.exists(args.out_path)):
        os.makedirs(args.out_path)
    fig.figure.savefig(
        os.path.join(
            args.out_path, 
            f'probe_accuracy.{args.format}'), 
            format=args.format,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')

    tasks = {
        'none': './models/dynaprog/probe/Qwen2.5-1.5B',
        'row': './models/dynaprog/probe_row/Qwen2.5-1.5B',
    }

    args = parser.parse_args()
    main(args, tasks)