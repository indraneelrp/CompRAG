'''
Call HNSW search given query HRR vectors
'''
import torch
import hnswlib
import numpy as np
from .make_triplets import main_generate_triplets, main_generate_triplets_from_list
from .encode import doc_triplets2embeddings, doc_embeddings2hrr
from typing import Sequence
from compRAG.text_samples import get_hardcoded_texts


def initialise_hnsw(dim: int, max_elems: int)-> hnswlib.Index:
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=max_elems, ef_construction=300, M=16)
    return index

def add_items(index: hnswlib.Index, 
              vecs: Sequence[Sequence[float]] | np.ndarray, 
              ids: Sequence[int] | np.ndarray)-> None:
    vecs = np.array(vecs, dtype=np.float32)
    ids = np.array(ids, dtype=np.int32)
    index.add_items(vecs, ids)

def get_nearest_triplet_ids(query_hrrs: list[torch.Tensor], index: hnswlib.Index, k=2)-> list[int]:
    matched_triplet_ids: list[int] = []
    if not query_hrrs:
        return matched_triplet_ids
    for query_hrr in query_hrrs:     
        if isinstance(query_hrr, torch.Tensor):
            query_hrr = query_hrr.detach().cpu().numpy().astype(np.float32)
        else:
            query_hrr = np.asarray(query_hrr, dtype=np.float32)

        # hnswlib expects shape (n_queries, dim)
        labels, distances = index.knn_query(query_hrr.reshape(1, -1), k=k)
        matched_triplet_ids.extend(labels[0].tolist()) # labels[0] bcause shape of labels is [[l1, l2,...]]
    return matched_triplet_ids

def test_top_k_hardcoded():
    print("===========\ncheck top-k\n----------")
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')   # dim 384

    vanilla_vecs = []
    vanilla_ids = []
    vanilla_id_dict = {}
    h_txt = get_hardcoded_texts()
    for i, txt in enumerate(h_txt, 1):
        vanilla_vecs.append(embedding_model.encode([txt])[0])
        vanilla_ids.append(i)
        vanilla_id_dict[i] = txt
    print("--encoded texts vanilla--")
    
    indexVanilla = initialise_hnsw(384, 5000)
    add_items(indexVanilla, vanilla_vecs, vanilla_ids)
    print("--initialised & added to vanilla hnsw index--")

    q_txt = '''what was the thing the team did differently in publishing their paper?'''
    vanilla_q_e = embedding_model.encode([q_txt])[0]
    print("--preprocessed query vanilla--")

    qv = np.asarray(vanilla_q_e, dtype=np.float32).reshape(1, -1)
    labels, distances = indexVanilla.knn_query(qv, k=2)
    for l in labels[0]:
        print(vanilla_id_dict[l])


if __name__ == "__main__":
    import spacy
    nlp = spacy.load("en_core_web_sm")

    h = get_hardcoded_texts()
    t = main_generate_triplets_from_list(nlp, h)
    t_e = doc_triplets2embeddings(t)
    t_hrr = doc_embeddings2hrr(t_e)

    index = initialise_hnsw(384, 10000)
    print("--initialised hnsw--")

    vecs = []
    ids = []
    id_to_chunk = {}
    id_to_t = {}
    for i, chunk_hrrs in enumerate(t_hrr, 0):
        for j, triplet_hrr in enumerate(chunk_hrrs, 0):
            vecs.append(triplet_hrr)
            ids.append((i+1)*10000+j)
            id_to_chunk[(i+1)*10000+j] = t[i]
            id_to_t[(i+1)*10000+j] = t[i][j]
    
    add_items(index, vecs, ids)
    print("--added items to hnsw index--")

    # q_txt = '''what was thing the team did differently in publishing their paper?'''
    q_txt = '''how long did the team work before they had a breakthrough?'''
    # q_txt = '''what was the university of southern california's group called where Vaswani earned his degree'''



    q_t = main_generate_triplets(nlp, q_txt)
    q_t = [q_t]
    q_t_e = doc_triplets2embeddings(q_t)
    q_t_hrr = doc_embeddings2hrr(q_t_e)
    print("--preprocessed query--")

    print(q_t)
    # print(len(q_t_e))        # checking shape of q_t_e
    # print(len(q_t_hrr[0]))   # checking shape of q_t_hrr
    if len(q_t_hrr[0]) != 0:
        for query_hrr in q_t_hrr[0]:
            labels, distances = index.knn_query(query_hrr, k=2)
            print(labels)
            for d in distances[0]:
                print(1 - d)
            for l in labels[0]:
                print(id_to_t[l])    # print the exact triplet that matched
                # print(id_to_chunk[l])   # print the entire chunk from where the matched triplet came