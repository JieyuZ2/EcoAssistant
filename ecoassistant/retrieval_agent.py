import ast
import secrets
import traceback as tb
from typing import Dict, Optional, Union

from autogen.agentchat import Agent, UserProxyAgent, ConversableAgent
from autogen.code_utils import extract_code, execute_code

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

# Let GPT use APIs to solve the task
API_INSTRUCTION = """You can use the API keys in the following dictionary (key: API, value: API key):
{apis}
Directly use the provided API key in your code. Do not use placeholders of API key in the code.

"""

# Let GPT use tools to solve the task
DEMO_INSTRUCTION = """We provide some examples of query and python code used to solve the query below. They may be helpful as references. 
----------
"""

DEMO_TEMPLATE = """
query: {query}

code:
```python
{code}
```
----------

"""

GENERAL_INSTRUCTION = """Begin!

{query}"""

COT_INSTRUCTION = """Begin!

{query}

Let's think step by step."""


def has_code(message):
    code_blocks = extract_code(message)
    for lang, code in code_blocks:
        if lang in ["bash", "shell", "sh", "python", "Python"]:
            return True
    return False


def has_input(code_string):
    tree = ast.parse(code_string)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == 'input':
                    return True
    return False


def is_termination_msg(msg):
    msg = msg.get("content", "")
    if isinstance(msg, str):
        return msg.endswith("TERMINATE")
    return False

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import chromadb.utils.embedding_functions as ef


class Retriever:
    def __init__(self, topk, threshold, db_name="query", model_name='multi-qa-mpnet-base-dot-v1', metadata={"hnsw:space": "ip"}):
        self.n_results = topk
        self.threshold = threshold

        client = chromadb.Client()
        embedding_function = ef.SentenceTransformerEmbeddingFunction(model_name, normalize_embeddings=True)
        self.collection = client.get_or_create_collection(name=db_name, embedding_function=embedding_function, metadata=metadata)

    def is_empty(self):
        return self.collection.count() == 0

    def add(self, query):
        self.collection.add(
            documents=query,
            ids=query,
        )

    def delete(self, query):
        self.collection.delete(
            ids=query,
        )

    def query(self, query_text):
        results = self.collection.query(
            query_texts=query_text,
            n_results=self.n_results
        )
        retrieved = []
        for distance, document in zip(results['distances'][0], results['documents'][0]):
            if distance < self.threshold:
                retrieved.append(document)
        return retrieved


class RetrievalAgent(UserProxyAgent):

    def __init__(
            self,
            name,

            apis={},
            api_tokens={},

            solution_demonstration=True,
            retrieval_topk=1,
            retrieval_threshold=0.5,

            chain_of_thought=False,

            evaluation_function=None,

            is_termination_msg=is_termination_msg,
            default_auto_reply='Reply "TERMINATE" if you think everything is done.',
            **config,
    ):
        super().__init__(name, human_input_mode='NEVER', is_termination_msg=is_termination_msg, default_auto_reply=default_auto_reply, **config)

        self.chain_of_thought = chain_of_thought

        self.record = []
        self._evaluation_record = []
        self._evaluation_function = evaluation_function

        if solution_demonstration:
            self.retriever = Retriever(retrieval_topk, retrieval_threshold)
        else:
            self.retriever = None

        self.retrieval_log = {}
        self.query_to_code = {}

        self.api_keys = apis
        self.apis, self.api_token_to_key = {}, {}
        for api, key in apis.items():
            if api in api_tokens:
                token = api_tokens[api]
            else:
                token = secrets.token_hex(4)
            self.apis[api] = token
            self.api_token_to_key[token] = key

        # cache and flags
        self._initial_query = None
        self._code_logs = []

        self._context = {}

    def run_code(self, code, lang="python", **kwargs):
        if lang == "python":
            # check python syntax
            try:
                ast.parse(code)
            except Exception as e:
                error_string = ''.join(tb.format_exception_only(type(e), e))
                return 1, f"{type(e).__name__}: {error_string}", self._code_execution_config["use_docker"]
            # check whether the code uses `input` function
            if has_input(code):
                return 1, "The python code used `input` function, which is not supported.", self._code_execution_config["use_docker"]

            if len(self.apis) > 0:
                for api, token in self.apis.items():
                    if token in code:
                        code = code.replace(token, self.api_token_to_key[token])

        if "last_n_messages" in kwargs:
            kwargs.pop("last_n_messages")

        return execute_code(code, lang=lang, **kwargs)

    def execute_code_blocks(self, code_blocks):
        """Execute the code blocks and return the result."""
        logs_all = ""
        python_codes, python_code_outputs = [], []
        for i, code_block in enumerate(code_blocks):
            lang, code = code_block
            # if not lang:
            #     lang = infer_lang(code)
            print(colored(f"\n>>>>>>>> EXECUTING CODE BLOCK {i} (inferred language is {lang})...", "red"), flush=True)
            if lang in ["bash", "shell", "sh"]:
                exitcode, logs, image = self.run_code(code, lang=lang, **self._code_execution_config)
            elif lang in ["python", "Python"]:
                if code.startswith("# filename: "):
                    filename = code[11: code.find("\n")].strip()
                else:
                    filename = None
                exitcode, logs, image = self.run_code(
                    code,
                    lang="python",
                    filename=filename,
                    **self._code_execution_config,
                )
                python_codes.append(code)
                python_code_outputs.append(logs)
            else:
                # In case the language is not supported, we return an error message.
                # exitcode, logs, image = 1, f"unknown language {lang}", self._code_execution_config["use_docker"]
                exitcode, logs, image = 0, f"", self._code_execution_config["use_docker"]
                # raise NotImplementedError
            self._code_execution_config["use_docker"] = image
            logs_all += "\n" + logs
            if exitcode != 0:
                self._code_logs.append({
                    'exitcode'   : exitcode,
                    'code'       : python_codes,
                    'code_output': python_code_outputs
                })
                return exitcode, logs_all
        self._code_logs.append({
            'exitcode'   : exitcode,
            'code'       : python_codes,
            'code_output': python_code_outputs
        })
        return exitcode, logs_all

    def _evaluate(self, sender):
        eval_res = {}

        if len(self._code_logs) > 0 and self._code_logs[-1]['exitcode'] != 0:
            eval_res = {
                'final'      : False,
                'explanation': 'The last code execution failed.'
            }

        elif self._evaluation_function is not None:
            conversation = sender.get_conversation()
            eval_res = self._evaluation_function(self._initial_query, conversation)

        self._evaluation_record.append(eval_res)
        return eval_res

    def _reset(self):
        self.reset()
        self._initial_query = None
        self._code_logs = []

    def receive(
            self,
            message: Union[Dict, str],
            sender: Agent,
            request_reply: Optional[bool] = None,
            silent: Optional[bool] = False,
    ):
        """Receive a message from the sender agent.
        Once a message is received, this function sends a reply to the sender or simply stop.
        The reply can be generated automatically or entered manually by a human.
        """
        self._process_received_message(message, sender, silent)
        message = self._message_to_dict(message)

        if "content" in message:

            if message["content"] == 'FAIL':
                print("\n>>>>>>>> FAILED. TERMINATING...", flush=True)
                self.record.append({
                    'task'        : "fail",
                    'eval_results': self._evaluation_record
                })
                self._evaluation_record = []
                return

            if message["content"] == 'ERROR':
                print("\n>>>>>>>> ERROR...", flush=True)
                self._evaluation_record.append({
                    'final'      : False,
                    'explanation': 'error.'
                })
                self.send("FAIL", sender)
                return

            if message["content"] == 'RESTART':
                # restart with a new agent
                print("\n>>>>>>>> SWITCH SUB-AGENT. RESTARTING...", flush=True)

                self.reset()

                self._code_logs = []
                initial_message = self.generate_init_message(self._initial_query)
                self.send(initial_message, sender)
                return

        if self._consecutive_auto_reply_counter[
            sender.name
        ] >= self._max_consecutive_auto_reply or (self._is_termination_msg(message) and not has_code(message.get("content", ""))):
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender.name] = 0

            eval_res = self._evaluate(sender)

            if eval_res['final']:
                # succeed
                self.record.append({
                    'task'        : "success",
                    'eval_results': self._evaluation_record
                })
                self._evaluation_record = []
                return

            else:
                # fail
                self.send("FAIL", sender)
                return

        self._consecutive_auto_reply_counter[sender.name] += 1
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender,
                                    exclude=[ConversableAgent.generate_oai_reply, ConversableAgent.check_termination_and_human_reply])

        # only use demonstration for the initial message to save the number of inputting tokens
        self._context['retrieval'] = False
        reply = {
            "content": reply,
            "role"   : "user",
            "context": self._context
        }

        self.send(reply, sender)

    def set_query(self, query: str):
        self._initial_query = query

    def _retrieve_example(self, query):
        if self.retriever is None or self.retriever.is_empty():
            return []
        else:
            return self.retriever.query(query)

    def _get_api_instruction(self) -> str:
        return API_INSTRUCTION.format(apis=self.apis)

    def get_solution_demonstration(self, query: str) -> str:
        instruction = ""

        results = self._retrieve_example(query)

        if len(results) > 0:
            instruction = DEMO_INSTRUCTION
            for prev_query in results:
                code = self.query_to_code[prev_query]
                instruction += DEMO_TEMPLATE.format(
                    query=prev_query,
                    code=code,
                )
        return instruction

    def generate_init_message(self, query: str):
        self.set_query(query)

        def content(context):
            instruction = ""
            if context['api']:
                instruction += self._get_api_instruction()
            if context['retrieval']:
                instruction += self.get_solution_demonstration(query)
            if self.chain_of_thought:
                return instruction + COT_INSTRUCTION.format(query=query)
            else:
                return instruction + GENERAL_INSTRUCTION.format(query=query)

        self._context = {
            "api"      : len(self.apis) > 0,
            "retrieval": len(self.query_to_code) > 0,
        }

        message = {
            "content": content,
            "role"   : "user",
            "context": self._context
        }

        return message

    def initiate_chat(self, query, assistant):

        initial_message = self.generate_init_message(query)

        examples = self._retrieve_example(query)
        for prev_query in examples:
            self.retrieval_log[prev_query][0] += 1

        assistant.receive(initial_message, self)
        logs = {
            'messages'         : assistant.model_messages,
            'call'             : assistant.model_call_counter,
            'prompt_tokens'    : assistant.model_prompt_tokens_counter,
            'completion_tokens': assistant.model_completion_tokens_counter,
            'result'           : self.record[-1],
        }
        assistant.reset()

        if self.retriever is not None and self.record[-1]['task'] == 'success':
            if len(self._code_logs) > 0 and self._code_logs[-1]['exitcode'] == 0:
                self.query_to_code[self._initial_query] = '\n'.join(self._code_logs[-1]['code'])
                self.retriever.add(self._initial_query)
                self.retrieval_log[self._initial_query] = [0, 0]  # retrieved cnt, # success cnt

            for prev_query in examples:
                self.retrieval_log[prev_query][1] += 1

        self._reset()

        return logs
