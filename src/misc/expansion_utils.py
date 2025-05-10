from datasets import Dataset

prompt_prefix = """Find a path in the given table from the top-left corner to the bottom-right corner that maximizes the sum of the numbers on it. You can only move rightwards or downwards.

Table:
"""

def solve_dp(table):
    m, n = len(table), len(table[0])
    dp = [[0 for _ in range(n)] for __ in range(m)]

    for i in range(m):
        for j in range(n):
            past = 0
            if (i > 0):
                past = max(past, dp[i-1][j])
            if (j > 0):
                past = max(past, dp[i][j-1])

            dp[i][j] = table[i][j] + past

    return dp[-1][-1], dp


def formulate_prompt(table):
    table_rows = []
    for row in table:
        s = ' '.join([str(x) for x in row])
        table_rows.append(s)
    table_str = '\n'.join(table_rows)

    answer, _ = solve_dp(table)
    query = f'{prompt_prefix}{table_str}\n\n'
    prompt = f'{query}Result: {answer}'
    return query, prompt, answer


def parse_dataset_name(data_path):
    segs = data_path.split('/')[-1].split('.json')[0].split('_mul_')
    da, db = int(segs[0]), int(segs[1])
    return da, db


def process_dataset(raw_dataset):
    all_data = {
        'id': [], 
        'query': [], 
        'prompt': [],
        'golden': [],
    }

    for entry in raw_dataset:
        all_data['id'].append(entry['id'])
        all_data['query'].append(entry['query'])
        all_data['prompt'].append(entry['prompt'])
        all_data['golden'].append(entry['golden'])

    processed_dataset = Dataset.from_dict(all_data)
    return processed_dataset


def expand_tokenizer(tokenizer, model):
    added_tokens = []
    for i in range(10000):
        added_tokens.append(f'<{i}>')
    tokenizer.add_tokens(added_tokens)
    model.resize_token_embeddings(len(tokenizer))


def generate_expanded_cot(table):
    steps = []
    _, dp = solve_dp(table)
    for row in dp:
        steps.append(' '.join([f'<{x}>' for x in row]))

    full_cot = '\n'.join(steps)
    return full_cot


def process_expanded_dataset(
    raw_dataset,
):
    all_data = {
        'id': [], 
        'query': [], 
        'prompt': [],
        'golden': [],
    }

    for entry in raw_dataset:
        all_data['id'].append(entry['id'])
        all_data['query'].append(entry['query'])
        all_data['golden'].append(entry['golden'])

        table = entry['table']
        cot = generate_expanded_cot(table)
        prompt = entry['query'] + '<tool_call>' + cot + '</tool_call>'
        prompt += f'\n\nResult: {entry['golden']}'
        all_data['prompt'].append(prompt)

    processed_dataset = Dataset.from_dict(all_data)
    return processed_dataset


if __name__ == '__main__':
    table = [
        [15, 5, 59, 62, 22],
        [41, 61, 7, 12, 27],
        [98, 60, 34, 94, 24],
        [45, 40, 12, 77, 11],
        [56, 94, 46, 34, 45],
    ]

    res, dp = solve_dp(table)
    print(res)
    print(dp)