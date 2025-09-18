'''
Call HNSW search given a vector (which should already be a HRR triple vector)
'''
import hnswlib
import numpy as np
from make_triplets import main_generate_triplets_hardcoded
from encode import doc_triplets2embeddings, doc_embeddings2hrr

def initialise_hnsw(dim, max_elems):
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=max_elems, ef_construction=200, M=16)
    return index

def add_items(index, vecs, ids):
    vecs = np.array(vecs, dtype=np.float32)
    ids = np.array(ids, dtype=np.int32)
    index.add_items(vecs, ids)


if __name__ == "__main__":
    t = main_generate_triplets_hardcoded()
    t_e = doc_triplets2embeddings(t)
    t_hrr = doc_embeddings2hrr(t_e)

    index = initialise_hnsw(384, 10000)

    vecs = []
    ids = []
    for i, chunk_hrrs in enumerate(t_hrr, 100):
        for j, triplet_hrr in enumerate(chunk_hrrs, 10):
            vecs.append(triplet_hrr)
            ids.append(i*10000+j)
    
    add_items(index, vecs, ids)