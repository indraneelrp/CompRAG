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

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')   # dim 384
# Note- Maybe we should use word2vec and check with that as well. for phrases, just average the word2vec embeddings

# using hardcoded values, to be refactored 

# get embeddings for triplets
def triplet2embeddings():
    all_triplets = main_generate_triplets_hardcoded()
    all_triplets_embed = []
    for chunk_triplets in all_triplets:
        chunk_triplets_embed = []
        for x,R,y in chunk_triplets:
            embed_x = embedding_model.encode([x])[0]
            embed_R = embedding_model.encode([R])[0]
            embed_y = embedding_model.encode([y])[0]
            chunk_triplets_embed.append([embed_x, embed_R, embed_y])
        all_triplets_embed.append(chunk_triplets_embed)
    return all_triplets_embed

# Apply HRR
def embedding2hrr(all_triplets_embeddings):
    all_hrr_bound_vectors = []
    for chunk_embeds in all_triplets_embeddings:
        chunk_bound_vectors = []
        for x_e, R_e, y_e in chunk_embeds:
            x_hrr = projection(x_e, dim=-1)
            R_hrr = projection(R_e, dim=-1)
            y_hrr = projection(y_e, dim=-1)

            b1 = binding(x_hrr, R_hrr, dim=-1)
            b2 = binding(b1, y_hrr, dim=-1)
            chunk_bound_vectors.append(b2)
        all_hrr_bound_vectors.append(chunk_bound_vectors)
    return all_hrr_bound_vectors

