# EcoAssistant

EcoAssistant: using LLM assistant more affordably and accurately


## Introduction

### What is EcoAssistant?

EcoAssistant is a framework that can make the LLM assistant more affordable and accurate for code-driven question answering. It is based on the idea of **assistant hierarchy** and **solution demonstration**.
It is built upon [AutoGen](https://github.com/microsoft/autogen).

### What is LLM assistant?

LLM assistant is an assistant agent backed by conversational LLM such as ChatGPT and GPT-4 and is able to address user query in a conversation.

### What is code-driven question answering?

Code-driven question answering is a task that requires the LLM assistant to write code to call external APIs to answer the question. 
For example, given the question "What is the average temperature of the city X in the next 5 days?", the assistant needs to write code to get the weather information via certain APIS and calculate the average temperature of the city X in the next 5 days.


### What is assistant hierarchy?

Assistant hierarchy is a hierarchy of assistants, where the LLM assistants are ranked by their cost (e.g., GPT-3.5-turbo -> GPT-4). 
When addressing a user query, the EcoAssistant first asks the cheapest assistant to answer the query. 
Only when it fails, we invoke the more expensive assistant. It is designed to save the cost by reducing the usage of expensive assistants.

### What is solution demonstration?

Solution demonstration is a technique that leverage the past successful query-code pair to help future queries.
Everytime when a query is successfully addressed, we save the query-code pair to a database. 
When a new query comes, we retrieve the most similar query from the database, and then use the query and its associated code as in-context demonstration.
It is designed to improve the accuracy by leveraging the past successful query-code pairs.

**The combination of assistant hierarchy and solution demonstration amplifies the individual benefits because the solution from high-performing model would be naturally leveraged to guide weaker model without specific designs.**

### Why use EcoAssistant?

For queries about weather, stock, and places, EcoAssistant surpass individual GPT-4 assistant by 10 points of success rate with less than 50% of GPT-4's cost.
More details can be found in our paper.

## Preparation

All the data are included in this repository.

You only need to set your API keys in `keys.json`

Install required libraries (we recommend Python3.10):

```bash
pip3 install -r requirements.txt
```

## Instructions

We use the **Mixed-100** dataset as an example. For other dataset, just change the dataset name to google_places/stock/weather/mixed_1/mixed_2/mixed_3 in the following commands.

The outputting results can be found in the `results` folder.

The following commands are for the autonomous systems without human feedback described in the Section 4.5.


**Run the GPT-3.5-turbo assistant**

```bash
python3 run.py --data mixed_100 --seed 0 --api --model gpt-3.5-turbo 
```

**Run the GPT-3.5-turbo assistant + Chain-of-Thought**

turn on `cot`

```bash
python3 run.py --data mixed_100 --seed 0 --api --cot --model gpt-3.5-turbo 
```


**Run the GPT-3.5-turbo assistant + solution demonstration**

turn on `solution_demonstration`

```bash
python3 run.py --data mixed_100 --seed 0 --api --solution_demonstration --model gpt-3.5-turbo 
```


**Run the assistant hierarchy (GPT-3.5-turbo + GPT-4)**

set `model` to `gpt-3.5-turbo,gpt-4` 

```bash
python3 run.py --data mixed_100 --seed 0 --api --model gpt-3.5-turbo,gpt-4
```


**Run the EcoAssistant: assistant hierarchy (GPT-3.5-turbo + GPT-4) + solution demonstration**

```bash
python3 run.py --data mixed_100 --seed 0 --api --solution_demonstration --model gpt-3.5-turbo,gpt-4
```

**Enable human feedback**

For systems with human judgement, please set `eval` to `human` (which is by default `llm`) like the following example command.

```bash
python3 run.py --data mixed_100 --seed 0 --api --solution_demonstration --model gpt-3.5-turbo,gpt-4 --eval human
```

**Run the gold code for Mixed-100 we collect as described in Section 4.4**

This script would print the code outputs.

```bash
python3 run_gold_code_for_mix_100.py
```