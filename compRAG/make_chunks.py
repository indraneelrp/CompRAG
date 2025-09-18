from datasets import load_dataset

'''
Read in Hotpot QA or any source data
The size of the chunk should be an argument
'''

ds = load_dataset("BeIR/hotpotqa", "corpus")

C_SIZE = 1

def make_chunks(size, data):
    pass

corpus = ds["corpus"]
print(corpus[0])