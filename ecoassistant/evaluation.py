import json
import os

from autogen import oai

HUMAN_EVAL_CACHE = './human_eval_cache.json'

JUDGE_PROMPT = """
USER INPUT: {query}
CONVERSATION HISTORY:
{conversation_history}
"""

JUDGE_SYSTEM_MESSAGE = """You are a fair AI judge.
1. You will be given a user input and a conversation history between the user and one AI assistant. Below is the schema:
    USER INPUT: <user_input>

    CONVERSATION HISTORY: 
    ASSISTANT 1: <assistant_utterance_1>
    USER 2: <user_utterance_2>
    ASSISTANT 2: <assistant_utterance_2>
    ...
2. Your task is to judge whether the user's task is successfully completed.
3. The AI assistant may suggest code for the user to execute. If the user does not execute the code, the task is not successfully completed.
4. Your output should be the judgement (Yes or No) only.
"""


def format_conversation(conversation):
    formatted_conversation = ""
    for idx, (i, o) in enumerate(conversation):
        idx += 1
        if idx == 1:
            formatted_conversation += f"ASSISTANT {idx}: {o}\n"
        else:
            formatted_conversation += f"USER {idx}: {i}\n"
            formatted_conversation += f"ASSISTANT {idx}: {o}\n"
    return formatted_conversation


def judge(query, conversation, llm_config):
    formatted_conversation = format_conversation(conversation)
    message = JUDGE_PROMPT.format(query=query, conversation_history=formatted_conversation)
    messages = [{"content": JUDGE_SYSTEM_MESSAGE, "role": "system"}, {"content": message, "role": "user"}]
    responses = oai.ChatCompletion.create(messages=messages, **llm_config)
    prompt_tokens, completion_tokens = responses['usage']['prompt_tokens'], responses['usage']['completion_tokens']
    return oai.ChatCompletion.extract_text(responses)[0], prompt_tokens, completion_tokens


def model_evaluation_function(llm_config):
    def success_check(query, conversation):
        result = {}
        try:
            response, prompt_tokens, completion_tokens = judge(query, conversation, llm_config)
            succeed = response
            result['prompt_tokens'] = prompt_tokens
            result['completion_tokens'] = completion_tokens
            assert completion_tokens == 1
        except Exception as e:
            print(f"GPT-4 evaluation failed: {e}")
            succeed = 'no'
        if "yes" in succeed.lower():
            succeed = True
        else:
            succeed = False
        print(f"\n>>>>>>>> GPT-4 JUDGEMENT: {'SUCCEED' if succeed else 'FAIL'}", flush=True)
        result['model'] = succeed
        result['final'] = succeed
        return result

    return success_check


def human_evaluation_function(query, conversation, cache=HUMAN_EVAL_CACHE):
    formatted_conversation = format_conversation(conversation)
    if os.path.exists(cache):
        human_eval_cache = json.load(open(cache, 'r'))
    else:
        human_eval_cache = {}
    if query in human_eval_cache and formatted_conversation in human_eval_cache[query]:
        result = human_eval_cache[query][formatted_conversation]
        print(f"\n>>>>>>>> HUMAN JUDGEMENT: {'SUCCEED' if result['final'] else 'FAIL'}", flush=True)
        return result
    else:
        succeed = input(f"{query}\nwhether the AI assistant successfully completes the user's task (y/n):")

        if "y" == succeed.lower():
            succeed = True
        else:
            succeed = False
        print(f"\n>>>>>>>> HUMAN JUDGEMENT: {'SUCCEED' if succeed else 'FAIL'}", flush=True)
        result = {
            'final': succeed,
            'human': succeed,
        }
        if query in human_eval_cache:
            human_eval_cache[query][formatted_conversation] = result
        else:
            human_eval_cache[query] = {formatted_conversation: result}
        json.dump(human_eval_cache, open(cache, 'w'), indent=4)
        return result


def mixed_evaluation_function(llm_config, cache=HUMAN_EVAL_CACHE):
    model_eval_func = model_evaluation_function(llm_config)

    def success_check(query, conversation):
        model_result = model_eval_func(query, conversation)
        human_result = human_evaluation_function(query, conversation, cache)
        model_result.update(human_result)
        return model_result

    return success_check
