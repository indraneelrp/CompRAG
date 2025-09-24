# CompRAG

Comp is for COMPrehension and COMPosition: Retrieval that comprehends any source text with composition and graph methods.

## Test that you are ready to run this repo

To check that you are ready to use this repo, in a new environment, do the following:

- Install Ollama
- In a new terminal run `Ollama run gemma3`
- Then run `Ollama run mxbai-embed-large`. This installs the necessary models for the test script.
- Then run `Ollama serve` and leave the terminal window running.
- In a new terminal cd to CompRAG and run `pip install -r requirements.txt`
- Navigate to sanity-check-ollama and run `python TEST1_Ollama_RAG_test.py`
- Depending on your CPU, it might take a while to produce some output. If all works well, the model will produce a response that states the document and its own name in its response.
