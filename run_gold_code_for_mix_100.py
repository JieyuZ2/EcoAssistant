import json

API_tokens = {
    'google places': '181dbb37',
    'weatherapi': 'b4d5490d',
    'alphavantage': 'af8fb19b'
}

APIs = json.load(open('./keys.json', 'r'))
query_to_code = json.load(open('./dataset/query2code.json', 'r'))
queries = json.load(open('./dataset/mixed_100.json', 'r'))

for query in queries:
    code = query_to_code[query]
    print(query)
    print('-'*50)
    for api, token in API_tokens.items():
        if token in code:
            code = code.replace(token, APIs[api])
    print(code)
    print('-' * 50)
    exec(code)
    print('='*50)
    print('\n\n')

