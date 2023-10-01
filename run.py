import argparse
import json
import os
import sys
from time import time
from functools import partial

import autogen

from ecoassistant import RetrievalAgent, \
    initial_assistant_hierarchy, model_evaluation_function, human_evaluation_function, mixed_evaluation_function

KEY_LOC = "./"
here = os.path.abspath(os.path.dirname(__file__))


class Transcript(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.logfile.write(message)
        self.logfile.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def start(filename):
    """Start transcript, appending print output to given filename"""
    sys.stdout = Transcript(filename)


def stop():
    """Stop transcript and return print functionality to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal


def get_llm_config(model, openai_key=None):
    if openai_key is None:
        config_list = autogen.config_list_from_models(key_file_path=KEY_LOC, model_list=[model])
    else:
        config_list = [{'api_key': openai_key, 'model': model}]
    return config_list


def get_llm_configs(models, openai_key=None):
    config_lists = []
    for model in models:
        if model == 'Llama-2-13b-chat-hf':
            config_list = [{
                'api_key'   : 'llama',
                'api_base'  : 'http://localhost:8000/v1',
                'model'     : 'meta-llama/Llama-2-13b-chat-hf',
                'max_tokens': 945,
            }]
        else:
            config_list = get_llm_config(model, openai_key=openai_key)
        config_lists.append(config_list)
    return config_lists


def main(args, queries):
    APIs = json.load(open(args.path_to_key, 'r'))

    API_tokens = {
        'google places': '181dbb37',
        'weatherapi'   : 'b4d5490d',
        'alphavantage' : 'af8fb19b'
    }

    openai_key = APIs.pop('openai')

    models = args.model.split(',')

    config_lists = get_llm_configs(models, openai_key=openai_key)
    assistant = initial_assistant_hierarchy(models, config_lists, seed=args.seed, name='assistant')

    if args.eval == 'human':
        evaluation_func = partial(human_evaluation_function, cache=os.path.join(args.output_dir, 'human_eval_cache.json'))
    else:
        config_list = get_llm_config('gpt-4', openai_key=openai_key)
        llm_config = {
            "model"          : 'gpt-4',
            "request_timeout": 600,
            "seed"           : 43,
            "config_list"    : config_list
        }
        if args.eval == 'llm':
            evaluation_func = model_evaluation_function(llm_config)
        elif args.eval == 'mix':
            evaluation_func = mixed_evaluation_function(llm_config, cache=os.path.join(args.output_dir, 'human_eval_cache.json'))
        else:
            raise NotImplementedError

    code_execution_config = {
        "work_dir"  : args.work_dir,
        "use_docker": "python:3" if args.docker else False,
        "timeout"   : 300,
    }

    user = RetrievalAgent(
        "user",

        chain_of_thought=args.cot,

        apis=APIs if args.api else {},
        api_tokens=API_tokens if args.api else {},

        solution_demonstration=args.solution_demonstration,
        retrieval_topk=args.retrieval_topk,
        retrieval_threshold=args.retrieval_threshold,

        evaluation_function=evaluation_func,
        max_consecutive_auto_reply=5,
        code_execution_config=code_execution_config,
    )

    start(os.path.join(args.output_dir, f'{args.name}.txt'))
    print(args)
    logs = {}
    time_start = time()
    logs['args'] = args.__dict__
    for i, raw_query in enumerate(queries):
        print(f'>>>>>>>> Query {i}: {raw_query}')
        log = user.initiate_chat(raw_query, assistant)
        logs[raw_query] = log
    run_time = time() - time_start
    logs['run_time'] = run_time
    print(f'Run time: {run_time}')
    stop()

    return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo,gpt-4")
    parser.add_argument('--eval', type=str, default="llm", choices=['human', 'llm', 'mix'])
    parser.add_argument('--docker', action="store_true", default=False)

    parser.add_argument('--api', action="store_true", default=False)
    parser.add_argument('--cot', action="store_true", default=False)
    parser.add_argument('--solution_demonstration', action="store_true", default=False)
    parser.add_argument('--retrieval_threshold', type=float, default=0.5)
    parser.add_argument('--retrieval_topk', type=int, default=1)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--path_to_key', type=str, default='./keys.json')
    parser.add_argument('--data', type=str, default='mixed_100')
    args = parser.parse_args()

    output_dir = os.path.join('results/')
    os.path.exists(output_dir) or os.makedirs(output_dir)
    args.output_dir = output_dir

    args.name = f'{args.data}_{args.eval}_{args.seed}_{args.model}'
    if args.api:
        args.name += '_api'
    if args.cot:
        args.name += '_cot'
    if args.solution_demonstration:
        args.name += '_soldemo'

    data_path = './dataset/'
    queries = json.load(open(os.path.join(data_path, f'{args.data}.json')))

    args.work_dir = os.path.join('workdir/', f'{args.name}_workdir')
    os.path.exists(args.work_dir) or os.makedirs(args.work_dir)
    logs = main(args, queries)

    json.dump(logs, open(os.path.join(output_dir, f'{args.name}.json'), 'w'), indent=4)
