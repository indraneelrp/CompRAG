from pypdf import PdfReader
import requests
import json

reader = PdfReader("./test_doc.pdf")

input_doc = []
for page in reader.pages:
    input_doc.append(page.extract_text())

llm_url = 'http://localhost:11434/api/generate'

prompt = """
You are a bot that reads, understands, searches through and analyses documents. You help users with these documents based on the query. You do not include extra information not found in the supplied documents. If you cannot find information, state that it cannot be found.
This is the supplied document: {supplied_document}
The query is: {query}
Give an answer based on the query and the document.
"""

prompt = """
You are a bot that reads, understands and searches through documents. You help users with these documents based on the query. You do not include extra information not found in the supplied documents.
This is the supplied document: {supplied_document}
The query is: {query}
Give an answer based on the query and the document.
"""


query = """Tell me the title of the document and the llm model name (your name)"""

data = {
    "model": "gemma3",
    "prompt": prompt.format(query=query, supplied_document=input_doc),
    "options": {
        "temperature": 0
    }
}

headers = {'Content-Type': 'application/json'}

full_response = []

response = requests.post(llm_url, data=json.dumps(data), headers=headers, stream=True)

try:
    count = 0
    for line in response.iter_lines():
        # filter out keep-alive new lines
        # count += 1
        # if count % 5== 0:
        #     print(decoded_line['response']) # print every fifth token
        if line:
            decoded_line = json.loads(line.decode('utf-8'))
            
            full_response.append(decoded_line['response'])
finally:
    response.close()

print(''.join(full_response))
