import autogen
from autogen.code_utils import extract_code


class EcoAssistant:
	DEMO_INSTRUCTION = """We provide one example query and python code used to solve the query below. They may be helpful as references. 
    ----------
    query: {example}

    code:
    ```python
    {code}
    ```
    ----------
    {query}
    """

	def __init__(self, assistant_hierarchy, user_proxy, threshold=0.5):
		self.assistant_hierarchy = assistant_hierarchy
		self.user_proxy = user_proxy

		self.solution_store = {}
		self.threshold = threshold

		# __import__('pysqlite3')
		# import sys
		# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

		import chromadb
		import chromadb.utils.embedding_functions as ef

		client = chromadb.Client()
		embedding_function = ef.SentenceTransformerEmbeddingFunction("multi-qa-mpnet-base-dot-v1", normalize_embeddings=True)
		self.collection = client.get_or_create_collection(name="solution", embedding_function=embedding_function, metadata={"hnsw:space": "ip"})

	def _is_empty(self):
		return self.collection.count() == 0

	def _add_solution(self, query):
		self.collection.add(
			documents=query,
			ids=query,
		)

	def _query_solution(self, query_text):
		results = self.collection.query(
			query_texts=query_text,
			n_results=1
		)
		retrieved = []
		for distance, document in zip(results['distances'][0], results['documents'][0]):
			if distance < self.threshold:
				retrieved.append(document)
		return retrieved

	def extract_last_code(self, chat_history):
		code = None
		for msg in chat_history[::-1]:
			code = extract_code(msg['content'])[0]
			if code[0] == 'python':
				code = code[1]
				code = "```python\n" + code + "\n```"
				break
		return code

	def solve(self, queries, solution_demonstration=True):
		cost = 0
		success = []

		for query in queries:

			if solution_demonstration and not self._is_empty():
				retrieved = self._query_solution(query)
				if len(retrieved) > 0:
					example = retrieved[0]
					solution_code = self.solution_store[example]
					query = self.DEMO_INSTRUCTION.format(example=example, code=solution_code, query=query)

			success.append(False)
			for assistant in self.assistant_hierarchy:
				chat_res = self.user_proxy.initiate_chat(assistant, message=query)
				cost += chat_res.cost['usage_including_cached_inference']['total_cost']
				if input("Succeed? (y/n)") == "y":
				# if True:
					code = self.extract_last_code(chat_res.chat_history)
					self.solution_store[query] = code
					self._add_solution(query)
					success[-1] = True
					break


		return cost, sum(success) / len(success)


config_path = "./OAI_CONFIG_LIST"

config_list1 = autogen.config_list_from_json(
	config_path,
	filter_dict={"tags": ["gpt-3.5-turbo"]},  # comment out to get all
)
assistant1 = autogen.AssistantAgent(
	name="assistant1",
	llm_config={
		"cache_seed" : 41,  # seed for caching and reproducibility
		"config_list": config_list1,  # a list of OpenAI API configurations
		"temperature": 0,  # temperature for sampling
	},  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
)

config_list2 = autogen.config_list_from_json(
	config_path,
	filter_dict={"tags": ["gpt-4"]},  # comment out to get all
)
assistant2 = autogen.AssistantAgent(
	name="assistant2",
	llm_config={
		"cache_seed" : 41,  # seed for caching and reproducibility
		"config_list": config_list2,  # a list of OpenAI API configurations
		"temperature": 0,  # temperature for sampling
	},  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
	name="user_proxy",
	human_input_mode="NEVER",  # get user feedback
	max_consecutive_auto_reply=10,
	is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
	code_execution_config={
		# the executor to run the generated code
		"work_dir"  : "coding",
		"use_docker": False,
	},
)

eco = EcoAssistant(assistant_hierarchy=[assistant1, assistant2], user_proxy=user_proxy)

queries = [
    """What date is today? Compare the year-to-date gain for META and TESLA."""
    """What date is today? Compare the year-to-date gain for META and TESLA."""
]

cost1, success_rate1 = eco.solve(queries)

cost2, success_rate2 = eco.solve(queries, solution_demonstration=False)

print(f"Cost: {cost1}")
print(f"Success rate: {success_rate1}")

print(f"Cost: {cost2}")
print(f"Success rate: {success_rate2}")

