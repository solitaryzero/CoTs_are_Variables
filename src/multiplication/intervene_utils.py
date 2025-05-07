import random
import json

import argparse
import os
from tqdm import tqdm

modules = ['partial_mul', 'addition', 'final_result']
step_types = {
    'partial_mul': ['abstract', 'substep', 'summary'],
    'addition': ['abstract', 'substep'],
    'final_result': ['abstract'],
}

cot_example = """3773*6821=<tool_call>Calculate 3773*1
3*1=3, digit 3, carry 0
7*1=7, digit 7, carry 0
7*1=7, digit 7, carry 0
3*1=3, digit 3, carry 0
Result of 3773*1=3773
Calculate 3773*20
3*2=6, digit 6, carry 0
7*2=14, digit 4, carry 1
7*2=14, digit 5, carry 1
3*2=6, digit 7, carry 0
Result of 3773*20=75460
Calculate 3773*800
3*8=24, digit 4, carry 2
7*8=56, digit 8, carry 5
7*8=56, digit 1, carry 6
3*8=24, digit 0, carry 3
Result of 3773*800=3018400
Calculate 3773*6000
3*6=18, digit 8, carry 1
7*6=42, digit 3, carry 4
7*6=42, digit 6, carry 4
3*6=18, digit 2, carry 2
Result of 3773*6000=22638000

Add up partial results: 3773+75460+3018400+22638000
3773+75460+3018400+22638000=79233+3018400+22638000
79233+3018400+22638000=3097633+22638000
3097633+22638000=25735633

The final result is: 3773*6821=25735633</tool_call>

Result: 25735633"""

def parse_cot(
    text,
):
    # cot_parts = {
    #     'partial_mul': [],
    #     'addition': [],
    #     'final_result': [],
    # }
    cot_parts = {
        'partial_mul': [],
    }
    cot = text.split('<tool_call>')[1].split('</tool_call>')[0]
    segs = cot.split('\n\n')

    # Module 1: partial multiplication
    steps = []
    lines = segs[0].split('\n')
    for line in lines:
        if (line.startswith('Calculate')):
            t = line.split('Calculate ')[1]
            a, b = t.split('*')
            assert (len(b) == 1) or (int(b[1:]) == 0)
            steps.append({
                'step': 'partial_mul',
                'type': 'abstract',
                'input_a': int(a),
                'input_b': int(b),
            })
        elif (line.startswith('Result of')):
            t = line.split('Result of ')[1]
            tt, res = t.split('=')
            a, b = tt.split('*')
            assert (len(b) == 1) or (int(b[1:]) == 0)
            steps.append({
                'step': 'partial_mul',
                'type': 'summary',
                'input_a': int(a),
                'input_b': int(b),
                'output': int(res),
            })
        else:
            equation, t = line.split(', digit ')
            digit, carry = t.split(', carry ')
            steps.append({
                'step': 'partial_mul',
                'type': 'substep',
                'equation': equation,
                'digit': int(digit),
                'carry': int(carry),
            })

    cot_parts['partial_mul'] = steps

    # Module 2: addition
    if (len(segs) == 1):
        return cot_parts
    steps = []
    lines = segs[1].split('\n')
    for line in lines:
        if (line.startswith('Add up partial results: ')):
            equation = line.split('Add up partial results: ')[1]
            elements = [int(x) for x in equation.split('+')]
            steps.append({
                'step': 'addition',
                'type': 'abstract',
                'elements': elements,
            })
        else:
            inp, output = line.split('=')
            input_elements = [int(x) for x in inp.split('+')]
            output_elements = [int(x) for x in output.split('+')]
            steps.append({
                'step': 'addition',
                'type': 'substep',
                'input_elements': input_elements,
                'output_elements': output_elements,
            })

    cot_parts['addition'] = steps

    # Module 3: final result
    if (len(segs) == 2):
        return cot_parts
    line = segs[2].strip()
    assert (line.startswith('The final result is: '))
    t = line.split('The final result is: ')[1]
    inp, result = t.split('=')
    steps = [{
        'step': 'final_result',
        'type': 'abstract',
        'input': inp,
        'output': int(result),
    }]
    cot_parts['final_result'] = steps

    return cot_parts


def reconstruct_cot(
    steps,
):
    cot = ''
    # Module 1: partial multiplication
    step_texts = []
    for step in steps['partial_mul']:
        if (step['type'] == 'abstract'):
            text = f"Calculate {step['input_a']}*{step['input_b']}"
            step_texts.append(text)
        elif (step['type'] == 'substep'):
            text = f"{step['equation']}, digit {step['digit']}, carry {step['carry']}"
            step_texts.append(text)
        elif (step['type'] == 'summary'):
            text = f"Result of {step['input_a']}*{step['input_b']}={step['output']}"
            step_texts.append(text)
        else:
            raise NotImplementedError
        
    cot += '\n'.join(step_texts)

    # Module 2: addition
    if ('addition' in steps):
        cot += '\n\n'
        
        step_texts = []
        for step in steps['addition']:
            if (step['type'] == 'abstract'):
                inp = '+'.join([str(x) for x in step['elements']])
                text = f"Add up partial results: {inp}"
                step_texts.append(text)
            elif (step['type'] == 'substep'):
                inp = '+'.join([str(x) for x in step['input_elements']])
                output = '+'.join([str(x) for x in step['output_elements']])
                text = f"{inp}={output}"
                step_texts.append(text)
            else:
                raise NotImplementedError
            
        cot += '\n'.join(step_texts)

    # Module 3: final result
    if ('final_result' in steps):
        cot += '\n\n'

        assert len(steps['final_result']) == 1
        step = steps['final_result'][0]
        text = f"The final result is: {step['input']}={step['output']}"

        cot += (text+'</tool_call>')

    cot = '<tool_call>' + cot
    return cot


def simulate_partial_cot(
    partial_cot,
    original_a,
    original_b,
):
    full_cot = {
        'partial_mul': [],
        'addition': [],
        'final_result': [],
    }
    simulation_result = None

    if ('addition' not in partial_cot): # modification at partial multiplication
        partial_mul_results = []
        step_digits = []
        a_digit_index, b_digit_index = 0, 0
        carry = 0
        last_step_type = None
        str_a, str_b = str(original_a), str(original_b)
        len_a, len_b = len(str_a), len(str_b)

        for step in partial_cot['partial_mul']:
            full_cot['partial_mul'].append(step)
            last_step_type = step['type']
            if (step['type'] == 'abstract'):
                carry = 0
                step_digits = []
                a_digit_index = 0
            elif (step['type'] == 'substep'):
                digit, carry = step['digit'], step['carry']
                step_digits.append(digit)
                a_digit_index += 1
            elif (step['type'] == 'summary'):
                output = step['output']
                step_digits.append(carry)
                step_str_digits = [str(x) for x in reversed(step_digits)]
                prediction = ''.join(step_str_digits)
                zeros = '0'*(b_digit_index)
                assert (int(prediction + zeros) == output)

                partial_mul_results.append(step['output'])
                b_digit_index += 1
            else:
                raise NotImplementedError
            
        # complete partial multiplication module
        # 1. complete current loop
        if (last_step_type == 'abstract'): # should not appear
            raise NotImplementedError
        elif (last_step_type == 'substep'):
            # complete substeps
            for i in range(a_digit_index, len_a):
                da = str_a[-i-1]
                db = str_b[-b_digit_index-1]
                tt = int(da)*int(db)
                t = tt+carry
                digit, carry = t%10, t//10
                equation = f"{da}*{db}={tt}"
                step = {
                    'step': 'partial_mul',
                    'type': 'substep',
                    'equation': equation,
                    'digit': digit,
                    'carry': carry,
                }
                step_digits.append(digit)
                full_cot['partial_mul'].append(step)
            
            # complete summary
            zeros = '0'*(b_digit_index)
            step_b = str_b[-b_digit_index-1]+str(zeros)
            step_digits.append(carry)
            step_str_digits = [str(x) for x in reversed(step_digits)]
            res = ''.join(step_str_digits)+zeros
            step = {
                'step': 'partial_mul',
                'type': 'summary',
                'input_a': original_a,
                'input_b': int(step_b),
                'output': int(res),
            }
            partial_mul_results.append(step['output'])
            b_digit_index += 1
            full_cot['partial_mul'].append(step)

        # 2. complete remaining loops
        for j in range(b_digit_index, len_b):
            zeros = '0'*(j)
            step_b = str_b[-j-1]+str(zeros)

            # abstract
            carry = 0
            step_digits = []

            step = {
                'step': 'partial_mul',
                'type': 'abstract',
                'input_a': original_a,
                'input_b': int(step_b),
            }
            full_cot['partial_mul'].append(step)

            # substeps
            for i in range(len_a):
                da = str_a[-i-1]
                db = str_b[-j-1]
                tt = int(da)*int(db)
                t = tt+carry
                digit, carry = t%10, t//10
                equation = f"{da}*{db}={tt}"
                step = {
                    'step': 'partial_mul',
                    'type': 'substep',
                    'equation': equation,
                    'digit': digit,
                    'carry': carry,
                }
                step_digits.append(digit)
                full_cot['partial_mul'].append(step)
            
            # summary
            step_digits.append(carry)
            step_str_digits = [str(x) for x in reversed(step_digits)]
            res = ''.join(step_str_digits)+zeros
            step = {
                'step': 'partial_mul',
                'type': 'summary',
                'input_a': original_a,
                'input_b': int(step_b),
                'output': int(res),
            }
            partial_mul_results.append(step['output'])
            b_digit_index += 1
            full_cot['partial_mul'].append(step)

        # addition & final result
        elements = partial_mul_results
        step = {
            'step': 'addition',
            'type': 'abstract',
            'elements': elements,
        }
        full_cot['addition'].append(step)

        while (len(elements) > 1): # complete addition module
            output_elements = []
            output_elements.append(elements[0] + elements[1])
            output_elements.extend(elements[2:])
            step = {
                'step': 'addition',
                'type': 'substep',
                'input_elements': elements,
                'output_elements': output_elements,
            }
            full_cot['addition'].append(step)
            elements = output_elements

        addition_result = elements[0]

        # complete final result module
        full_cot['final_result'] = [{
            'step': 'final_result',
            'type': 'abstract',
            'input': f"{original_a}*{original_b}",
            'output': int(addition_result),
        }]

        simulation_result = addition_result

    elif ('final_result' not in partial_cot): # modification at serial addition
        full_cot['partial_mul'] = partial_cot['partial_mul']

        elements = None
        for step in partial_cot['addition']:
            full_cot['addition'].append(step)
            if (step['type'] == 'abstract'):
                elements = step['elements']
            elif (step['type'] == 'substep'):
                elements = step['output_elements']

        while (len(elements) > 1): # complete addition module
            output_elements = []
            output_elements.append(elements[0] + elements[1])
            output_elements.extend(elements[2:])
            step = {
                'step': 'addition',
                'type': 'substep',
                'input_elements': elements,
                'output_elements': output_elements,
            }
            full_cot['addition'].append(step)
            elements = output_elements

        addition_result = elements[0]

        # complete final result module
        full_cot['final_result'] = [{
            'step': 'final_result',
            'type': 'abstract',
            'input': f"{original_a}*{original_b}",
            'output': int(addition_result),
        }]

        simulation_result = addition_result
    else: # modification at final result
        full_cot['partial_mul'] = partial_cot['partial_mul']
        full_cot['addition'] = partial_cot['addition']
        full_cot['final_result'] = partial_cot['final_result']

        assert len(partial_cot['final_result']) == 1
        simulation_result = partial_cot['final_result'][0]['output']

    return full_cot, simulation_result


def generate_random_value(original):
    n = len(str(original))
    if n == 1:
        candidates = list(range(10))
        candidates.remove(original)
        return random.choice(candidates)
    else:
        lower = 10**(n-1)
        upper = (10**n)-1
        new = random.randint(lower, upper)
        while (new == original):
            new = random.randint(lower, upper)
        return new


def modify_cot(
    text,
):
    try:
        cot_steps = parse_cot(text)
    except:
        print(text)
        return None, None
    
    flat_steps = []
    for module in cot_steps:
        for i, step in enumerate(cot_steps[module]):
            if not((module == 'partial_mul') and ((step['type'] == 'abstract') or (step['type'] == 'summary'))): # Avoid vague modification
                flat_steps.append({
                    'module': module,
                    'index': i,
                })
            
    sampled_step = random.choice(flat_steps)
    new_cot_steps = {}
    step = cot_steps[sampled_step['module']][sampled_step['index']]
    modification = {
        'module': sampled_step['module'],
        'type': step['type'],
        'index': sampled_step['index'],
    }

    if (sampled_step['module'] == 'partial_mul'):
        if (step['type'] == 'substep'):
            element_name = random.choice(['digit', 'carry'])
            element = step[element_name]
            modified_value = generate_random_value(element)
            step[element_name] = modified_value
        # elif (step['type'] == 'summary'): # FIXME: 0 tails
        #     element_name = 'output'
        #     element = step[element_name]
        #     modified_value = generate_random_value(element)
        #     step[element_name] = modified_value

        modification['original'] = element
        modification['modified'] = modified_value

        new_cot_steps = {
            'partial_mul': cot_steps['partial_mul'][:sampled_step['index']+1],
        }
    elif (sampled_step['module'] == 'addition'):
        if (step['type'] == 'abstract'):
            element_name = 'elements'
            element = step[element_name]

            modification['original'] = element.copy()

            index = random.choice(range(len(element)))
            modified_value = generate_random_value(element[index])
            element[index] = modified_value

            modification['modified'] = element

        elif (step['type'] == 'substep'):
            element_name = 'output_elements'
            element = step[element_name]

            modification['original'] = element.copy()

            index = 0
            element = step[element_name]
            modified_value = generate_random_value(element[index])
            element[index] = modified_value

            modification['modified'] = element

        new_cot_steps = {
            'partial_mul': cot_steps['partial_mul'],
            'addition': cot_steps['addition'][:sampled_step['index']+1],
        }
    elif (sampled_step['module'] == 'final_result'):
        assert (step['type'] == 'abstract')
        element_name = 'output'
        element = step[element_name]
        modified_value = generate_random_value(element)
        step[element_name] = modified_value

        modification['original'] = element
        modification['modified'] = modified_value

        new_cot_steps = {
            'partial_mul': cot_steps['partial_mul'],
            'addition': cot_steps['addition'],
            'final_result': [step],
        }
    else:
        raise NotImplementedError
    
    return new_cot_steps, modification


def unit_test():
    # random.seed(123)

    query = '3773*6821='
    oa, ob = 3773, 6821
    suffix = '\n\nResult: 25735633'

    cot_steps = parse_cot(cot_example)
    print('=== CoT steps ===')
    print(json.dumps(cot_steps, indent=4))
    input()

    reconstructed = reconstruct_cot(cot_steps)
    print('=== Is reconstruction successful ===')
    print((query + reconstructed + suffix) == cot_example)
    print('=== Reconstructed ===')
    print(query + reconstructed + suffix)
    input()

    partial_cot, modification = modify_cot(cot_example)
    print('=== Partial CoT steps (after modification) ===')
    print(json.dumps(partial_cot, indent=4))
    print('=== Modification ===')
    print(json.dumps(modification, indent=4))
    input()

    partial_reconstructed = reconstruct_cot(partial_cot)
    print('=== Reconstructed partial CoT')
    print(partial_reconstructed)
    input()

    simulation_cot_steps, simluation_result = simulate_partial_cot(partial_cot, original_a=oa, original_b=ob)
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
        a, b = entry['query'][:-1].split('*')
        a, b = int(a), int(b)
        new_cot_steps, modification = modify_cot(full_cot)
        if (new_cot_steps is None):
            continue

        reconstructed_partial_cot = reconstruct_cot(new_cot_steps)
        simulated_cot_steps, simulation_result = simulate_partial_cot(new_cot_steps, a, b)
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
    args = parser.parse_args()

    if (args.unit_test):
        unit_test()

    random.seed(args.seed)
    generate_intervened_dataset(args)