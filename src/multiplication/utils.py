from tqdm import tqdm
from datasets import Dataset

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


def parse_dataset_name(data_path):
    segs = data_path.split('/')[-1].split('.json')[0].split('_mul_')
    da, db = int(segs[0]), int(segs[1])
    return da, db


def generate_full_cot(a, b):
    str_a, str_b = str(a), str(b)
    steps = []
    partial_results = []
    for ib in range(len(str_b)-1, -1, -1): # from tail of b
        db = int(str_b[ib])
        n_zero = len(str_b)-ib-1
        zeros = '0'*n_zero
        current_step = f'Calculate {str_a}*{db}{zeros}\n'
        carry = 0
        for ia in range(len(str_a)-1, -1, -1): # from tail of a
            da = int(str_a[ia])
            t = da*db + carry
            rem, carry = t%10, t//10
            current_step += f'{da}*{db}={da*db}, digit {rem}, carry {carry}\n'

        current_step += f'Result of {a}*{db}{zeros}={a*db}{zeros}'
        steps.append(current_step)
        partial_results.append(a*db*(10**n_zero))

    multiply_cot = '\n'.join(steps)
    
    all_addition_str = '+'.join([str(x) for x in partial_results])
    addition_cot = f'Add up partial results: {all_addition_str}\n'
    addition_steps = []

    if (len(partial_results) == 1):
        addition_steps.append(f'{partial_results[0]}={partial_results[0]}')
    else:
        partial_addition = partial_results[0]
        for i in range(1, len(partial_results)):
            prev_step_str = '+'.join([str(partial_addition)]+[str(x) for x in partial_results[i:]])
            partial_addition += partial_results[i]
            if (i == len(partial_results)-1):
                suffix = ''
            else:
                suffix = '+'+'+'.join([str(x) for x in partial_results[i+1:]])
            step = f'{prev_step_str}={partial_addition}{suffix}'
            addition_steps.append(step)

    addition_cot += '\n'.join(addition_steps) 
    final_result = f'The final result is: {a}*{b}={a*b}'
        
    full_cot = multiply_cot + '\n\n' + addition_cot + '\n\n' + final_result

    return full_cot


def generate_compressed_cot(a, b):
    str_a, str_b = str(a), str(b)
    steps = []
    partial_results = []
    for ib in range(len(str_b)-1, -1, -1): # from tail of b
        db = int(str_b[ib])
        n_zero = len(str_b)-ib-1
        zeros = '0'*n_zero
        current_step = f'{str_a}*{db}{zeros}\n'
        carry = 0
        for ia in range(len(str_a)-1, -1, -1): # from tail of a
            da = int(str_a[ia])
            t = da*db + carry
            rem, carry = t%10, t//10
            current_step += f'{da}*{db} {rem} {carry}\n'

        current_step += f'{a}*{db}{zeros}={a*db}{zeros}'
        steps.append(current_step)
        partial_results.append(a*db*(10**n_zero))

    multiply_cot = '\n'.join(steps)
    
    all_addition_str = '+'.join([str(x) for x in partial_results])
    addition_cot = f'{all_addition_str}\n'
    addition_steps = []

    if (len(partial_results) == 1):
        addition_steps.append(f'{partial_results[0]}={partial_results[0]}')
    else:
        partial_addition = partial_results[0]
        for i in range(1, len(partial_results)):
            prev_step_str = '+'.join([str(partial_addition)]+[str(x) for x in partial_results[i:]])
            partial_addition += partial_results[i]
            if (i == len(partial_results)-1):
                suffix = ''
            else:
                suffix = '+'+'+'.join([str(x) for x in partial_results[i+1:]])
            step = f'{prev_step_str}={partial_addition}{suffix}'
            addition_steps.append(step)

    addition_cot += '\n'.join(addition_steps) 
    final_result = f'{a}*{b}={a*b}'
        
    full_cot = multiply_cot + '\n\n' + addition_cot + '\n\n' + final_result

    return full_cot


def process_cot_dataset(raw_dataset, cot_type='full'):    
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

        a, b = entry['a'], entry['b']
        if (cot_type == 'full'):
            cot = generate_full_cot(a, b)
        elif (cot_type == 'compressed'):
            cot = generate_compressed_cot(a, b)
        prompt = entry['query'] + '<tool_call>' + cot + '</tool_call>'
        prompt += f'\n\nResult: {str(a*b)}'
        all_data['prompt'].append(prompt)

    processed_dataset = Dataset.from_dict(all_data)
    return processed_dataset


def generate_latent_cot(a, b, tokenizer, latent_dim):
    latent_token = tokenizer.latent_token
    latent_embeds = []

    str_a, str_b = str(a), str(b)
    steps = []
    partial_results = []
    for ib in range(len(str_b)-1, -1, -1): # from tail of b
        db = int(str_b[ib])
        n_zero = len(str_b)-ib-1
        zeros = '0'*n_zero
        current_step = f'{str_a}*{db}{zeros}\n'
        carry = 0
        for ia in range(len(str_a)-1, -1, -1): # from tail of a
            da = int(str_a[ia])
            t = da*db + carry
            rem, carry = t%10, t//10
            current_step += f'{latent_token}'

            # Latent embeds
            emb = [0 for _ in range(latent_dim)]
            emb[rem] = 1
            emb[(latent_dim//2)+carry] = 1
            latent_embeds.append(emb)

        current_step += f'|{a*db}{zeros}'
        steps.append(current_step)
        partial_results.append(a*db*(10**n_zero))

    multiply_cot = '\n'.join(steps)
    
    all_addition_str = '+'.join([str(x) for x in partial_results])
    addition_cot = f'{all_addition_str}\n'
    addition_steps = []

    if (len(partial_results) == 1):
        addition_steps.append(f'{partial_results[0]}={partial_results[0]}')
    else:
        partial_addition = partial_results[0]
        for i in range(1, len(partial_results)):
            prev_step_str = '+'.join([str(partial_addition)]+[str(x) for x in partial_results[i:]])
            partial_addition += partial_results[i]
            if (i == len(partial_results)-1):
                suffix = ''
            else:
                suffix = '+'+'+'.join([str(x) for x in partial_results[i+1:]])
            step = f'{prev_step_str}={partial_addition}{suffix}'
            addition_steps.append(step)

    addition_cot += '\n'.join(addition_steps) 
    final_result = f'{a}*{b}={a*b}'
        
    full_cot = multiply_cot + '\n\n' + addition_cot + '\n\n' + final_result

    return full_cot, latent_embeds


def process_latent_dataset(
    raw_dataset, 
    tokenizer,
    latent_dim=64,
):
    all_data = {
        'id': [], 
        'query': [], 
        'prompt': [],
        'latent_embeds': [],
        'golden': [],
    }

    for entry in raw_dataset:
        all_data['id'].append(entry['id'])
        all_data['query'].append(entry['query'])
        all_data['golden'].append(entry['golden'])

        a, b, golden = entry['a'], entry['b'], entry['golden']
        cot, latent_embeds = generate_latent_cot(a, b, tokenizer, latent_dim)
        prompt = entry['query'] + '<tool_call>' + cot + '</tool_call>'
        prompt += f'\n\nResult: {str(a*b)}'
        all_data['prompt'].append(prompt)
        all_data['latent_embeds'].append(latent_embeds)

    processed_dataset = Dataset.from_dict(all_data)
    return processed_dataset


if __name__ == '__main__':
    a, b = 1234, 5678
    full_cot = generate_full_cot(a, b)

    print(f'{a}*{b}={a*b}')
    print('=== COT ===')
    print(full_cot)