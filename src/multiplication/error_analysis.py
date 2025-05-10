import os
import json
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from intervene_utils import parse_cot, simulate_partial_cot

sns.set_context(rc={"axes.labelsize":20, "legend.fontsize":18, "legend.title_fontsize":20})

# Error types:
# 1. Addition error: error in large number addition
# 2. Reconstruct error: fail to reconstruct partial mul results from digits
# 3. Shortcut error: shortcut in *0 and *1 multiplications
# 4. Copy error: fail to copy partial mul results to addition step
# 5. Final error: the edit of final result step is not reflected (can be merged to Misc error)
# 6. Misc error: carry error, calculation error, etc,.


def find_error(
    simulated_steps,
    intervened_steps,
):
    partial_mul_results = []

    # Module 1: partial mul
    carry, step_digits, maybe_shortcut = None, None, False
    for i, int_step in enumerate(intervened_steps['partial_mul']):
        sim_step = simulated_steps['partial_mul'][i]

        if (sim_step != int_step):
            error = {
                'type': 'misc',
                'simluated_step': sim_step,
                'intervened_step': int_step,
            }
            if (maybe_shortcut):
                error['type'] = 'shortcut'
            elif (int_step['type'] == 'summary'):
                step_digits.append(carry)
                step_str_digits = ''.join([str(x) for x in reversed(step_digits)])
                if not(str(int_step['output']).startswith(step_str_digits)):
                    error['type'] = 'reconstruct'

            return error

        if (int_step['type'] == 'abstract'):
            carry = 0
            step_digits = []
            if (str(int_step['input_b']).startswith('0')) or (str(int_step['input_b']).startswith('1')):
                maybe_shortcut = True
            else:
                maybe_shortcut = False
        elif (int_step['type'] == 'substep'):
            step_digits.append(int_step['digit'])
            carry = int_step['carry']
        else:
            partial_mul_results.append(int_step['output'])

    # Module 2: addition
    for i, int_step in enumerate(intervened_steps['addition']):
        sim_step = simulated_steps['addition'][i]
        if (sim_step != int_step):
            error = {
                'type': 'misc',
                'simluated_step': sim_step,
                'intervened_step': int_step,
            }
            if (int_step['type'] == 'abstract'):
                error['type'] = 'copy'
            elif (int_step['type'] == 'substep'):
                error['type'] = 'addition'

            return error

    # Module 3: final result
    for i, int_step in enumerate(intervened_steps['final_result']):
        sim_step = simulated_steps['final_result'][i]
        if (sim_step != int_step):
            error = {
                'type': 'misc',
                'simluated_step': sim_step,
                'intervened_step': int_step,
            }

            return error
  
    return {
        'type': 'final',
    }


def show_examples(args):
    with open(args.data_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    for entry in data:
        if (entry['simulated_prediction'] != entry['intervened_prediction']):
            print('=== Intervened CoT ===')
            partial_cot = entry['partial_cot']
            intervened_cot = entry['output']
            assert intervened_cot.startswith(partial_cot)
            print(partial_cot + '-> Modified Here <-')
            print(entry['output'][len(partial_cot):])
            print('=== Diff ===')
            print('Original correctness: ', entry['raw_prediction'] == entry['golden'])
            print('Simulated: ', entry['simulated_prediction'])
            print('Intervened: ', entry['intervened_prediction'])
            input()


def draw_stats(args, stats):
    sub_type = ['addition', 'reconstruct', 'shortcut', 'copy', 'misc', 'final']
    num_error = stats['error']
    percentages = {}
    for key in sub_type:
        percentages[key] = stats[key] / num_error

    # merge final error to misc
    percentages['misc'] = percentages['misc'] + percentages['final']
    del percentages['final']

    # seaborn style
    sns.set_style("whitegrid")

    labels, sizes = [], []
    for key in percentages:
        labels.append(key)
        sizes.append(percentages[key])

    explode = []
    for key in labels:
        if (key == 'shortcut'):
            explode.append(0.1)
        else:
            explode.append(0)

    def autopct(pct):
        return f'{pct:.1f}%' if pct > 5 else ''

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.1, right=0.7)
    wedges, _, _ = plt.pie(
        sizes, 
        labels=None, 
        autopct=autopct, 
        # explode=explode,
        startangle=90, 
        textprops={'fontsize': 20},
        colors=sns.color_palette("pastel")
    )
    plt.title("Error breakdown", fontsize=20)
    plt.axis('equal')
   
    plt.legend(
        wedges, 
        labels,
        title="Error type",
        loc='upper right',
        bbox_to_anchor=(1, 1),
        bbox_transform=plt.gcf().transFigure
    )

    plt.savefig(
        os.path.join(
            args.out_path, 
            f'error_breakdown.{args.format}'), 
            format=args.format,
    )


def error_stats(args):
    with open(args.data_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    stats = {
        'total': 0,
        'success': 0,
        'error': 0,
        'addition': 0,
        'reconstruct': 0,
        'shortcut': 0,
        'copy': 0,
        'misc': 0,
        'final': 0,
    }

    for entry in data:
        stats['total'] += 1
        if (entry['simulated_prediction'] == entry['intervened_prediction']):
            stats['success'] += 1
        else:
            a, b = entry['query'][:-1].split('*')
            a, b = int(a), int(b)
            partial_steps = parse_cot(entry['partial_cot'])
            simulated_steps, _ = simulate_partial_cot(partial_steps, a, b)
            intervened_steps = parse_cot(entry['output'])

            stats['error'] += 1
            error = find_error(
                simulated_steps, 
                intervened_steps,
            )

            stats[error['type']] += 1

    print(json.dumps(stats, indent=4))

    if not(os.path.exists(args.out_path)):
        os.makedirs(args.out_path)

    with open(os.path.join(args.out_path, 'error_stats.json'), 'w', encoding='utf-8') as fout:
        json.dump(stats, fout)

    draw_stats(args, stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')

    args = parser.parse_args()
    # show_examples(args)
    error_stats(args)