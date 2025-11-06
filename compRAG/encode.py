'''
Take in triples, convert to 3 vectors via sentence transformer (So that we can handle 
multiple word entities E.g 'Pink Floyd', 'Attention is all you need')
Squash 3 vectors to 1 vector using the implementation given in HRR_pytorch.py
Save a triplets' HRR vector
'''
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any
import torch
import numpy as np

from .HRR_pytorch import projection, binding, unbinding

device = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')   # dim 384 

# Directional role vectors and seed
HRR_DIM = 384 
torch.manual_seed(42)

# Role vectors
R_SUBJ = projection(torch.randn(HRR_DIM, device=device), dim=-1)
R_REL = projection(torch.randn(HRR_DIM, device=device), dim=-1)
R_OBJ  = projection(torch.randn(HRR_DIM, device=device), dim=-1)

def role_bind(base_vec: torch.Tensor, role_vec: torch.Tensor) -> torch.Tensor:
    return binding(role_vec, base_vec, dim=-1) 

def doc_triplets2embeddings(doc_triplets: list[list[tuple[str, str, str]]]) -> list[list[list[np.ndarray]]]:
    doc_triplets_embed = []
    for chunk_triplets in doc_triplets:
        chunk_triplets_embed = []
        for x,R,y in chunk_triplets:
            embed_x = EMBEDDING_MODEL.encode([x])[0]
            embed_R = EMBEDDING_MODEL.encode([R])[0]
            embed_y = EMBEDDING_MODEL.encode([y])[0]
            chunk_triplets_embed.append([embed_x, embed_R, embed_y])
        doc_triplets_embed.append(chunk_triplets_embed)
    return doc_triplets_embed

def doc_embeddings2hrr(doc_triplet_embeddings: list[list[list[Any]]])-> list[list[torch.Tensor]]:
    doc_hrr_vectors = []
    for chunk_embeds in doc_triplet_embeddings:
        chunk_bound_vectors = []
        for x_e, R_e, y_e in chunk_embeds:
            x_e = torch.from_numpy(x_e).float().to(device)
            R_e = torch.from_numpy(R_e).float().to(device)
            y_e = torch.from_numpy(y_e).float().to(device)

            x_p = projection(x_e, dim=-1)
            R_p = projection(R_e, dim=-1)
            y_p = projection(y_e, dim=-1)

            hrr_subj = role_bind(x_p, R_SUBJ)
            hrr_rel= role_bind(R_p, R_REL)
            hrr_obj  = role_bind(y_p, R_OBJ)

            H = hrr_subj + hrr_rel + hrr_obj    # superposition: choose to add rather than convolve 
                                                # since adding vectors together makes it point to new meaning
            chunk_bound_vectors.append(H)
        doc_hrr_vectors.append(chunk_bound_vectors)
    return doc_hrr_vectors

def chunk_triplets2embeddings(chunk_triplets: list[tuple[str, str, str]])-> list[list[np.ndarray]]:
    chunk_triplets_embed = []
    for x,R,y in chunk_triplets:
        embed_x = EMBEDDING_MODEL.encode([x])[0]
        embed_R = EMBEDDING_MODEL.encode([R])[0]
        embed_y = EMBEDDING_MODEL.encode([y])[0]
        chunk_triplets_embed.append([embed_x, embed_R, embed_y])
    return chunk_triplets_embed

def chunk_embeddings2hrr(chunk_triplet_embeddings: list[list[Any]])-> list[torch.Tensor]:
    chunk_hrr_vectors = []
    for x_e, R_e, y_e in chunk_triplet_embeddings:
        x_e = torch.from_numpy(x_e).float().to(device)
        R_e = torch.from_numpy(R_e).float().to(device)
        y_e = torch.from_numpy(y_e).float().to(device)

        x_p = projection(x_e, dim=-1)
        R_p = projection(R_e, dim=-1)
        y_p = projection(y_e, dim=-1)

        hrr_subj = role_bind(x_p, R_SUBJ)
        hrr_rel = role_bind(R_p, R_REL)
        hrr_obj  = role_bind(y_p, R_OBJ)

        H = hrr_subj + hrr_rel + hrr_obj        
        chunk_hrr_vectors.append(H)
    return chunk_hrr_vectors