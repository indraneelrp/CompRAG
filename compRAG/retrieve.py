'''
Call HNSW search given query HRR vectors
'''
import hnswlib
import numpy as np
from make_triplets import main_generate_triplets_hardcoded, main_generate_triplets, get_hardcoded_texts
from encode import doc_triplets2embeddings, doc_embeddings2hrr

def initialise_hnsw(dim, max_elems):
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=max_elems, ef_construction=200, M=16)
    return index

def add_items(index, vecs, ids):
    vecs = np.array(vecs, dtype=np.float32)
    ids = np.array(ids, dtype=np.int32)
    index.add_items(vecs, ids)


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
    t = main_generate_triplets_hardcoded()
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

    q_txt = '''what was thing the team did differently in publishing their paper?'''

    import spacy
    nlp = spacy.load("en_core_web_sm")

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
            for l in labels[0]:
                print(id_to_t[l])    # print the exact triplet that matched
                # print(id_to_chunk[l])   # print the entire chunk from where the matched triplet came