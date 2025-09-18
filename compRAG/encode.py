'''
Take in triples, convert to 3 vectors via sentence transformer (So that we can handle 
multiple word entities E.g 'Pink Floyd', 'Attention is all you need')
Squash 3 vectors to 1 vector using the implementation given in HRR_pytorch.py
Save a triplets' HRR vector
'''
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from make_triplets import main_generate_triplets_hardcoded
from HRR_pytorch import projection, binding, unbinding
from typing import Any
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')   # dim 384
# Note- Maybe we should use word2vec and check with that as well. for phrases, just average the word2vec embeddings

# using hardcoded values from make_triplets.py, to be refactored 

# get embeddings for triplets
all_triplets = main_generate_triplets_hardcoded()

def doc_triplets2embeddings(doc_triplets: list[list[tuple[str, str, str]]]):
    doc_triplets_embed = []
    for chunk_triplets in doc_triplets:
        chunk_triplets_embed = []
        for x,R,y in chunk_triplets:
            embed_x = embedding_model.encode([x])[0]
            embed_R = embedding_model.encode([R])[0]
            embed_y = embedding_model.encode([y])[0]
            chunk_triplets_embed.append([embed_x, embed_R, embed_y])
        doc_triplets_embed.append(chunk_triplets_embed)
    return doc_triplets_embed

def doc_embeddings2hrr(doc_triplet_embeddings: list[list[list[Any]]]):
    doc_hrr_vectors = []
    for chunk_embeds in doc_triplet_embeddings:
        chunk_bound_vectors = []
        for x_e, R_e, y_e in chunk_embeds:
            x_e = torch.from_numpy(x_e).float().to(device)
            R_e = torch.from_numpy(R_e).float().to(device)
            y_e = torch.from_numpy(y_e).float().to(device)

            x_hrr = projection(x_e, dim=-1)
            R_hrr = projection(R_e, dim=-1)
            y_hrr = projection(y_e, dim=-1)

            b1 = binding(x_hrr, R_hrr, dim=-1)
            b2 = binding(b1, y_hrr, dim=-1)
            chunk_bound_vectors.append(b2)
        doc_hrr_vectors.append(chunk_bound_vectors)
    return doc_hrr_vectors

def chunk_triplets2embeddings(chunk_triplets: list[tuple[str, str, str]]):
    chunk_triplets_embed = []
    for x,R,y in chunk_triplets:
        embed_x = embedding_model.encode([x])[0]
        embed_R = embedding_model.encode([R])[0]
        embed_y = embedding_model.encode([y])[0]
        chunk_triplets_embed.append([embed_x, embed_R, embed_y])
    chunk_triplets_embed.append(chunk_triplets_embed)
    return chunk_triplets_embed

def chunk_embeddings2hrr(chunk_triplet_embeddings: list[list[Any]]):
    chunk_hrr_vectors = []
    for x_e, R_e, y_e in chunk_triplet_embeddings:
        x_e = torch.from_numpy(x_e).float().to(device)
        R_e = torch.from_numpy(R_e).float().to(device)
        y_e = torch.from_numpy(y_e).float().to(device)

        x_hrr = projection(x_e, dim=-1)
        R_hrr = projection(R_e, dim=-1)
        y_hrr = projection(y_e, dim=-1)

        b1 = binding(x_hrr, R_hrr, dim=-1)
        b2 = binding(b1, y_hrr, dim=-1)
        chunk_hrr_vectors.append(b2)
    return chunk_hrr_vectors