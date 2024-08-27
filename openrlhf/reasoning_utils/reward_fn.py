from pebble import ProcessPool
from tqdm import tqdm
import signal
import sympy
import math

TIMEOUT = 10
global_restricted = {lib: globals()[lib] for lib in ['sympy', 'math']}
# del global_restricted['sympy'].init_session
local_restricted = {}

DATASET = "gsm8k"
COT_MODE = "nl"

def setup_reward(dataset, cot_mode):
    global DATASET
    DATASET = dataset
    global COT_MODE
    COT_MODE = cot_mode

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def floatify(x):
    try:
        return float(x)
    except:
        return None

def run(code_piece, expr):
    _global_vars, _local_vars = {}, {}
    for lib in ['sympy', 'math']:
        _global_vars[lib] = global_restricted[lib]
        if lib in local_restricted:
            _local_vars[lib] = local_restricted[lib]
    exec(code_piece, _global_vars, _local_vars)
    result = eval(expr, _global_vars, _local_vars)
    return result

def process_code(code_gen, truncate_first_return=False):
    ## deal with blacklist keyword
    if 'sys.exit' in code_gen:
        code_gen = code_gen.replace('sys.exit', 'print')
    snippet = code_gen.split('\n')
    ## post process the code
    updated_code_snippet = ['import math', 'import sympy']
    for snippet_line in snippet:
        if snippet_line.startswith('def solution'):
            updated_code_snippet.append(snippet_line)
            continue
        if snippet_line.strip() == "":
            break
        if truncate_first_return:
            if snippet_line == "    return result":
                break
        updated_code_snippet.append(snippet_line)
    updated_code_gen = '\n'.join(updated_code_snippet)
    return updated_code_gen

def run_python_code(programs, TIMEOUT: float, safe=True):
    is_single_program = False
    if not isinstance(programs, list):
        is_single_program = True
        programs = [programs]
    updated_programs = [process_code(code) for code in programs]
    if safe:
        # Safer -- executed code can't affect main code (e.g numpy.random.seed(...))
        # But it is slow ... 
        with ProcessPool(max_workers=8) as pool:
            futures = [pool.schedule(run, args=[code,  'solution()'], timeout=TIMEOUT) for code in updated_programs]
            results = []
            for i, f in tqdm(enumerate(futures), total=len(futures), disable=True):
                try:
                    res = f.result()
                except Exception as e:
                    print(str(e)) #, updated_programs[i])
                    res = None
                results.append(res)
    else:
        results = []
        for code in tqdm(updated_programs, disable=True):
            with timeout(seconds=int(TIMEOUT)):
                try:
                    res = run(code_piece=code, expr="solution()")
                except Exception as e:
                    print(str(e), code)
                    res = None
                results.append(res)

    if is_single_program:
        assert len(results) == 1, len(results)
        return results[0]

    return results

# def get_post_process_answer_cot_fn(dataset, cot_mode):
#     answer_trigger = "\nTherefore, the answer is: "
#     return {
#         "python": {
#             "gsm8k": lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)], 
#             "svamp": lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)], 
#             "mathqa": lambda answer_cot: [str(res).lower().replace('"','').replace("'","").strip() for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)], 
#             "mathqa-numeric": lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)], 
#         }, 
#         "nl": {
#             "gsm8k": lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
#             "svamp": lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
#             "mathqa": lambda answer_cot: [res.split(answer_trigger)[-1].lower().replace('"','').replace("'",'').strip() for res in answer_cot],
#             "mathqa-numeric": lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
#         }
#     }.get(cot_mode).get(dataset)

answer_trigger = "\nTherefore, the answer is: "
post_process_answer_cot_fn = {
    "python": {
        "gsm8k": lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)], 
        "svamp": lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)], 
        "mathqa": lambda answer_cot: [str(res).lower().replace('"','').replace("'","").strip() for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)], 
        "mathqa-numeric": lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)], 
    }, 
    "nl": {
        "gsm8k": lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
        "svamp": lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
        "mathqa": lambda answer_cot: [res.split(answer_trigger)[-1].lower().replace('"','').replace("'",'').strip() for res in answer_cot],
        "mathqa-numeric": lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    }
}
    
# def get_compare_answer_fn(dataset):
#     return {
#         "gsm8k": lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
#         "svamp": lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
#         "mathqa": lambda extracted_ans, target_answer: extracted_ans == target_answer,
#         "mathqa-numeric": lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
#     }.get(dataset)

compare_answer_fn = {
    "gsm8k": lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    "svamp": lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    "mathqa": lambda extracted_ans, target_answer: extracted_ans == target_answer,
    "mathqa-numeric": lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
}

# def get_post_process_answer_value_fn(dataset):
#     return {
#         'gsm8k': lambda x: float(x.replace(',','').strip()),
#         'svamp': lambda x: float(x.replace(',','').strip()),
#         'mathqa': lambda x: x.lower().replace('"','').replace("'",'').strip(),
#         'mathqa-numeric': lambda x: float(x),
#     }.get(dataset)
    
post_process_answer_value_fn = {
    'gsm8k': lambda x: float(x.replace(',','').strip()),
    'svamp': lambda x: float(x.replace(',','').strip()),
    'mathqa': lambda x: x.lower().replace('"','').replace("'",'').strip(),
    'mathqa-numeric': lambda x: float(x),
}


def calculate_reward(generated_texts, answer_values):
    dataset = DATASET
    cot_mode = COT_MODE
    pred_values = post_process_answer_cot_fn[cot_mode][dataset](generated_texts)
    correctness = []
    for pred_value, target_value in zip(pred_values, answer_values):
        target_value = post_process_answer_value_fn[dataset](target_value)
        if pred_value is not None:
            if compare_answer_fn[dataset](pred_value, target_value):
                is_correct = 1
            else:
                is_correct = 0.1
        else:
            is_correct = 0
        correctness.append(is_correct)
    return correctness

def calculate_accuracy(generated_texts, answer_values):
    dataset = DATASET
    cot_mode = COT_MODE
    pred_values = post_process_answer_cot_fn[cot_mode][dataset](generated_texts)
    correctness = []
    for pred_value, target_value in zip(pred_values, answer_values):
        target_value = post_process_answer_value_fn[dataset](target_value)
        if pred_value is not None:
            if compare_answer_fn[dataset](pred_value, target_value):
                is_correct = 1
            else:
                is_correct = 0
        else:
            is_correct = 0
        correctness.append(is_correct)
    return correctness
    