import random
import json

import argparse
import os
from tqdm import tqdm

cot_example = """Find a path in the given table from the top-left corner to the bottom-right corner that maximizes the sum of the numbers on it. You can only move rightwards or downwards.

Table:
85 93 45 79 49
28 12 37 57 76
3 22 37 55 68
26 2 57 7 100
87 11 12 67 89

<tool_call>85 178 223 302 351
379 391 428 485 561
564 586 623 678 746
772 774 831 838 938
1025 1036 1048 1115 1204</tool_call>

Result: 1204"""

example_query = """Find a path in the given table from the top-left corner to the bottom-right corner that maximizes the sum of the numbers on it. You can only move rightwards or downwards.

Table:
85 93 45 79 49
28 12 37 57 76
3 22 37 55 68
26 2 57 7 100
87 11 12 67 89

"""

def parse_cot(
    text,
):
    dp_values = []
    cot = text.split('<tool_call>')[1].split('</tool_call>')[0]

    lines = cot.split('\n')
    for line in lines:
        dp_row = []
        values = line.strip().split(' ')
        for value in values:
            dp_row.append(int(value))

        dp_values.append(dp_row)

    return dp_values


def reconstruct_cot(
    dp_values,
):
    texts = []
    count = 0
    for row_values in dp_values:
        row_text = ' '.join([str(x) for x in row_values])
        texts.append(row_text)
        count += len(row_values)
    text = '\n'.join(texts)

    cot = '<tool_call>' + text
    if (count == 5*5):
        cot += '</tool_call>'
    return cot


def simulate_partial_cot(
    partial_dp_values,
    original_values,
):
    m, n = len(original_values), len(original_values[0])

    dp = [[0 for _ in range(n)] for __ in range(m)]

    for i in range(m):
        if (i >= len(partial_dp_values)): # new row
            for j in range(n):
                past = 0
                if (i > 0):
                    past = max(past, dp[i-1][j])
                if (j > 0):
                    past = max(past, dp[i][j-1])
                dp[i][j] = original_values[i][j] + past
        elif (len(partial_dp_values[i]) < n): # partial row
            for j in range(len(partial_dp_values[i])):
                dp[i][j] = partial_dp_values[i][j]
            for j in range(len(partial_dp_values[i]), n):
                past = 0
                if (i > 0):
                    past = max(past, dp[i-1][j])
                if (j > 0):
                    past = max(past, dp[i][j-1])
                dp[i][j] = original_values[i][j] + past
        else: # complete row
            dp[i] = partial_dp_values[i]

    simulation_result = dp[-1][-1]
    return dp, simulation_result


def generate_random_value(original, major=False):
    n = len(str(original))
    if n == 1:
        candidates = list(range(10))
        candidates.remove(original)
        return random.choice(candidates)
    else:
        if (major): # randomize all digits
            lower = 10**(n-1)
            upper = (10**n)-1
            new = random.randint(lower, upper)
            while (new == original):
                new = random.randint(lower, upper)
            return new
        else: # randomize all but the first digits
            base = 10**(n-1)
            rest = int(str(original)[0])*base

            lower = 10**(n-2)
            upper = (10**(n-1))-1
            new = random.randint(lower, upper)
            while (new+rest == original):
                new = random.randint(lower, upper)
            return new+rest


def modify_cot(
    text,
    major_intervention=False,
):
    try:
        dp = parse_cot(text)
        m, n = len(dp), len(dp[-1])
    
        flat_dp = []
        for i in range(m):
            for j in range(n):
                flat_dp.append({
                    'value': dp[i][j],
                    'index': (i, j),
                })
    except:
        print(text)
        return None, None
            
    sampled_step = random.choice(flat_dp)
    modification = {
        'index': sampled_step['index'],
    }

    si, sj = sampled_step['index']
    original_value = dp[si][sj]
    modified_value = generate_random_value(original_value, major_intervention)

    modification['original'] = original_value
    modification['modified'] = modified_value

    new_dp = []
    for i in range(si):
        new_dp.append(dp[i])
    
    new_line = []
    for j in range(sj):
        new_line.append(dp[si][j])
    new_line.append(modified_value)
    new_dp.append(new_line)
    
    return new_dp, modification


def unit_test():
    # random.seed(123)

    query = example_query
    table = [
        [85, 93, 45, 79, 49],
        [28, 12, 37, 57, 76],
        [3, 22, 37, 55, 68],
        [26, 2, 57, 7, 100],
        [87, 11, 12, 67, 89],
    ]
    suffix = '\n\nResult: 1204'

    dp_values = parse_cot(cot_example)
    print('=== DP Values ===')
    print(json.dumps(dp_values, indent=4))
    input()

    reconstructed = reconstruct_cot(dp_values)
    print('=== Is reconstruction successful ===')
    print((query + reconstructed + suffix) == cot_example)
    print('=== Reconstructed ===')
    print(query + reconstructed + suffix)
    input()

    partial_dp, modification = modify_cot(cot_example)
    print('=== Partial DP values (after modification) ===')
    print(json.dumps(partial_dp, indent=4))
    print('=== Modification ===')
    print(json.dumps(modification, indent=4))
    input()

    partial_reconstructed = reconstruct_cot(partial_dp)
    print('=== Reconstructed partial CoT')
    print(partial_reconstructed)
    input()

    simulation_cot_steps, simluation_result = simulate_partial_cot(partial_dp, table)
    print('=== Simuluation ===')
    print(json.dumps(simulation_cot_steps, indent=4))
    print('=== Simuluation Result ===')
    print(simluation_result)
    input()

    simulation_reconstructed = reconstruct_cot(simulation_cot_steps)
    print('=== Reconstructed simuluation CoT ===')
    print(simulation_reconstructed)
    input()


def generate_intervened_dataset(args):
    with open(args.input_file, 'r', encoding='utf-8') as fin:
        js = json.load(fin)
    
    intervened = []
    for entry in tqdm(js):
        full_cot = entry['output']
        table_text = entry['query'].split('Table:')[-1].strip()
        table = []
        for line in table_text.split('\n'):
            line_values = line.strip().split(' ')
            table.append([int(x) for x in line_values])

        new_dp_values, modification = modify_cot(full_cot, args.major_intervention)
        if (new_dp_values is None):
            continue

        reconstructed_partial_cot = reconstruct_cot(new_dp_values)
        simulated_cot_steps, simulation_result = simulate_partial_cot(new_dp_values, table)
        reconstructed_simulated_cot = reconstruct_cot(simulated_cot_steps)

        intervened_js = {
            'id': entry['id'],
            'query': entry['query'],
            'raw_output': entry['output'],
            'modification': modification,
            'partial_cot': entry['query'] + reconstructed_partial_cot,
            'simulated_cot': entry['query'] + reconstructed_simulated_cot,
            'raw_prediction': entry['prediction'],
            'golden': entry['golden'],
            'simulated_prediction': simulation_result,
        }
        intervened.append(intervened_js)

    if not(os.path.exists(args.output_path)):
        os.makedirs(args.output_path)
    with open(os.path.join(args.output_path, 'intervened.json'), 'w', encoding='utf-8') as fout:
        json.dump(intervened, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--unit_test',action='store_true')
    parser.add_argument('--major_intervention', action='store_true')
    args = parser.parse_args()

    if (args.unit_test):
        unit_test()

    random.seed(args.seed)
    generate_intervened_dataset(args)