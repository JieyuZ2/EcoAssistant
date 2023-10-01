import logging
from typing import Dict, List, Optional, Union

from autogen import oai
from autogen.agentchat import Agent, AssistantAgent

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
    In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute. You must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly. Solve the task step by step if you need to.
    If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
    If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
    When you find an answer, verify the answer carefully. 
    Reply "TERMINATE" in the end when everything is done.
"""


def initial_assistant_hierarchy(models, config_lists, name='assistant', seed=42, request_timeout=600):
    sub_assistants = []
    for model, config_list in zip(models, config_lists):
        llm_config = {
            "model"          : model,
            "request_timeout": request_timeout,
            "seed"           : seed,
            "config_list"    : config_list
        }
        sub_assistants.append(SubAssistant(
            f"{model}_{seed}",
            model=model,
            system_message=DEFAULT_SYSTEM_MESSAGE,
            llm_config=llm_config,
        ))

    assistant = HyperAssistantAgent(
        name,
        sub_agents=sub_assistants,
    )

    return assistant


class SubAssistant(AssistantAgent):
    def __init__(self, name, model, system_message=DEFAULT_SYSTEM_MESSAGE, **config):
        super().__init__(name, system_message, **config)
        self.model = model

    def generate_oai_reply(self, messages: List[Dict]):
        response = oai.ChatCompletion.create(
            context=messages[-1].pop("context", None), messages=self._oai_system_message + messages, **self.llm_config
        )
        prompt_tokens, completion_tokens = response['usage']['prompt_tokens'], response['usage']['completion_tokens']
        return oai.ChatCompletion.extract_text_or_function_call(response)[0], prompt_tokens, completion_tokens


class HyperAssistantAgent(AssistantAgent):
    def __init__(self,
                 name,
                 sub_agents,
                 ):
        super().__init__(name)
        self.sub_agents = sub_agents

        self._current_agent_pointer = 0
        self.model_call_counter = {}
        self.model_prompt_tokens_counter = {}
        self.model_completion_tokens_counter = {}
        self.model_messages = {agent.name: [] for agent in self.sub_agents}

    @property
    def _current_agent(self):
        return self.sub_agents[self._current_agent_pointer]

    @property
    def name(self):
        name = f"{self._name} [{self._current_agent.name}]"
        return name

    @property
    def if_last_agent(self):
        return self._current_agent_pointer == len(self.sub_agents) - 1

    def _switch_agent(self):
        self._current_agent_pointer = self._current_agent_pointer + 1
        super().reset()

    def get_conversation(self):
        conversation = []
        for message in self.model_messages[self._current_agent.name]:
            i = message["input"]
            o = message["output"]
            conversation.append((i, o))
        return conversation

    def receive(
            self,
            message: Union[Dict, str],
            sender: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
    ):

        self._process_received_message(message, sender, silent)

        if message == "FAIL":
            last_agent_name = self._current_agent.name
            self._switch_agent()
            if self._current_agent_pointer >= len(self.sub_agents):
                print("\n>>>>>>>> ENUMERATED ALL SUB-AGENTS. TERMINATING...", flush=True)
                self._reset_agents()
                self.send("FAIL", sender)
            else:
                print(f"\n>>>>>>>> SWITCH FROM {last_agent_name} TO {self._current_agent.name}", flush=True)
                self.send("RESTART", sender)
            return

        try:
            response, prompt_tokens, completion_tokens = self._current_agent.generate_oai_reply(self.chat_messages[sender])
        except Exception as err:
            logging.error(err)
            response, prompt_tokens, completion_tokens = "ERROR", 0, 0

        self.model_call_counter[self._current_agent.model] = self.model_call_counter.get(self._current_agent.model, 0) + 1
        self.model_prompt_tokens_counter[self._current_agent.model] = self.model_prompt_tokens_counter.get(self._current_agent.model, 0) + prompt_tokens
        self.model_completion_tokens_counter[self._current_agent.model] = self.model_completion_tokens_counter.get(self._current_agent.model, 0) + completion_tokens

        if isinstance(message, Dict):
            if callable(message["content"]):
                message = message["content"](message['context'])
            else:
                message = message["content"]

        self.model_messages[self._current_agent.name].append({"input": message, "output": response})

        self.send(response, sender)

    def _reset_agents(self):
        self._current_agent_pointer = 0
        for agent in self.sub_agents:
            agent.reset()

    def reset(self):
        super().reset()

        self.model_call_counter = {}
        self.model_prompt_tokens_counter = {}
        self.model_completion_tokens_counter = {}
        self.model_messages = {agent.name: [] for agent in self.sub_agents}

        self._reset_agents()
