import sqlite3
from compRAG.make_triplets import main_generate_triplets
from compRAG.encode import chunk_triplets2embeddings, chunk_embeddings2hrr
import spacy
from datasets import load_dataset
from dotenv import load_dotenv
import os
import numpy as np
import torch

load_dotenv()
db_path = os.getenv("HOTPOT_DB")

# {
#   "title": [
#     "Radio City (Indian radio station)",
#     "History of Albanian football",
#     "Echosmith" ....
#   ],
#   "sentences": [
#     [
#       "Radio City is India's first private FM radio station and was started on 3 July 2001.",
#       " It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengaluru (started first in 2001), Lucknow and New Delhi (since 2003).",
#       " It plays Hindi, English and regional songs.",
#       " It was launched in Hyderabad in March 2006, in Chennai on 7 July 2006 and in Visakhapatnam October 2007.", .....
#     ],
#     [
#       "Football in Albania existed before the Albanian Football Federation (FSHF) was created.", ....
#     ],
#     [
#       "Echosmith is an American, Corporate indie pop band formed in February 2009 in Chino, California.",
#       " Originally formed as a quartet of siblings, the band currently consists of Sydney, Noah and Graham Sierota, following the departure of eldest sibling Jamie in late 2016.",
#       " Echosmith started first as \"Ready Set Go!\"",
#       " until they signed to Warner Bros.", .....
#     ].......
#   ]
# }

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY, title TEXT, text TEXT NOT NULL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS triplets (
        id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id TEXT, 
        subject TEXT NOT NULL, relation TEXT NOT NULL, object TEXT NOT NULL,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS hrr_vectors (
        hrr_id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id TEXT, triplet_id INTEGER, hrr_vector BLOB,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id),
        FOREIGN KEY (triplet_id) REFERENCES triplets(id))''')
    
    conn.commit()
    return conn

def chunk_text(text, chunk_size=500):
    """Split text into chunks of roughly chunk_size characters"""
    words = text.split()
    chunks = []
    current = []
    current_len = 0
    
    for word in words:
        word_len = len(word) + 1
        if current_len + word_len > chunk_size and current:
            chunks.append(' '.join(current))
            current = [word] # start a new chunk w this word
            current_len = word_len
        else:
            current.append(word)
            current_len += word_len
    
    if current: # for the last chunk
        chunks.append(' '.join(current))
    
    return chunks


# EXAMPLE
# Iteration 1
# title = 'Radio City'
# sentences = ['Radio City is...', 'It broadcasts...']
# para_text = 'Radio City is... It broadcasts...'
# chunk0: title='Radio City', text='Radio City is...'
# chunk1: title='Radio City', text='It broadcasts...'

# Iteration 2
# title = 'Echosmith'
# sentences = ['Echosmith is...', 'They are best known...']
# para_text = 'Echosmith is... They are best known...'
# chunk3: title='Echosmith', text='Echosmith is...'


def process_example_no_embeddings(example, chunk_size=500, nlp=None):
    """Extract triplets only - no embeddings yet"""
    import time
    question_id = example['id']
    titles = example['context']['title']
    paragraphs = example['context']['sentences']

    results = []
    chunk_counter = 0

    for title, sentences in zip(titles, paragraphs):
        para_text = ' '.join(sentences)
        chunks = chunk_text(para_text, chunk_size)

        for chunk in chunks:
            chunk_id = f"{question_id}_chunk{chunk_counter}"
            chunk_counter += 1

            start_time = time.time()
            triplets = main_generate_triplets(nlp, chunk)
            elapsed = time.time() - start_time

            if elapsed > 10:  # Warn if REBEL takes > 10 seconds
                print(f"    WARNING: REBEL took {elapsed:.1f}s for chunk {chunk_id}", flush=True)

            results.append({
                'chunk_id': chunk_id,
                'title': title,
                'text': chunk,
                'triplets': triplets
            })

    return results

def batch_encode_hrrs(results_batch):
    """Encode all triplets in batch"""
    import time
    all_triplets = []
    chunk_map = []  # Track which chunk each triplet belongs to

    for i, result in enumerate(results_batch):
        for triplet in result['triplets']:
            all_triplets.append(triplet)
            chunk_map.append(i)

    if not all_triplets:
        return results_batch

    # Single batched encoding call
    try:
        start_time = time.time()
        all_embeddings = chunk_triplets2embeddings(all_triplets)
        embed_time = time.time() - start_time
        print(f"    Embedding {len(all_triplets)} triplets took {embed_time:.1f}s", flush=True)

        start_time = time.time()
        all_hrrs = chunk_embeddings2hrr(all_embeddings)
        hrr_time = time.time() - start_time
        print(f"    HRR encoding took {hrr_time:.1f}s", flush=True)

        # Distribute HRRs back to their chunks
        for result in results_batch:
            result['hrr_vectors'] = []

        for hrr_vec, chunk_idx in zip(all_hrrs, chunk_map):
            if isinstance(hrr_vec, torch.Tensor):
                hrr_vec = hrr_vec.cpu().numpy()
            hrr_blob = hrr_vec.astype(np.float32).tobytes()
            results_batch[chunk_idx]['hrr_vectors'].append(hrr_blob)
    except Exception as e:
        print(f"Batch HRR error: {e}")
        for result in results_batch:
            result['hrr_vectors'] = []

    return results_batch

# def process_example(example, chunk_size=500):
#     """Process a HotpotQA example into chunks with triplets and HRRs"""
#     question_id = example['id']
#     titles = example['context']['title']
#     paragraphs = example['context']['sentences']
    
#     results = []
#     chunk_counter = 0
    
#     # ONE paragraph at a time with its corresponding title
#     for title, sentences in zip(titles, paragraphs):
#         para_text = ' '.join(sentences)
#         chunks = chunk_text(para_text, chunk_size)
        
#         for chunk in chunks:
#             chunk_id = f"{question_id}_chunk{chunk_counter}"
#             chunk_counter += 1
           
#             triplets = main_generate_triplets(_worker_nlp, chunk)
#             triplets_db = [(chunk_id, s, r, o) for s, r, o in triplets]
            
#             hrr_vectors = []
#             if triplets:
#                 try:
#                     triplet_embeddings = chunk_triplets2embeddings(triplets)
#                     hrr_vecs = chunk_embeddings2hrr(triplet_embeddings)
                    
#                     for hrr_vec in hrr_vecs:
#                         if isinstance(hrr_vec, torch.Tensor):
#                             hrr_vec = hrr_vec.cpu().numpy()
#                         hrr_blob = hrr_vec.astype(np.float32).tobytes()
#                         hrr_vectors.append((chunk_id, hrr_blob))
#                 except Exception as e:
#                     print(f"HRR error for {chunk_id}: {e}")
            
#             results.append({
#                 'chunk': {'id': chunk_id, 'title': title, 'text': chunk},
#                 'triplets': triplets_db,
#                 'hrr': hrr_vectors
#             })
    
#     return results

# check which HotpotQA examples are already processed (so we can skip them on restart)
def get_processed_ids(conn):
    """Get IDs that are already processed"""
    c = conn.cursor()
    c.execute("SELECT DISTINCT substr(id, 1, instr(id, '_chunk')-1) FROM chunks WHERE id LIKE '%_chunk%'")
    return set(row[0] for row in c.fetchall() if row[0])

def process_dataset(split="validation", limit=None, chunk_size=500,
                   encoding_batch=128):
    from tqdm import tqdm
    
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")[split]
    total = limit if limit else len(ds)
    
    # connect to DB and check what's already done
    conn = init_db(db_path)
    processed_ids = get_processed_ids(conn)
    
    # filter out already processed
    to_process = [ex for ex in (ds.select(range(limit)) if limit else ds) 
                  if ex['id'] not in processed_ids]
    
    if not to_process:
        print("All examples already processed!")
        conn.close()
        return

    print(f"Processing {len(to_process)}/{total} examples ({len(processed_ids)} already done)")

    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

    # Combined pipeline: extract triplets and encode HRRs in mini-batches, then
    # write to DB after each mini-batch. This ensures progress is checkpointed
    # continuously so that a job restart resumes from the last committed batch
    # rather than starting over from scratch.
    #
    # ProcessPoolExecutor cannot be used here because REBEL is a CUDA model and
    # CUDA contexts cannot be safely forked into child processes.
    print(f"Extracting triplets and encoding HRRs ({encoding_batch} examples per checkpoint batch)...")

    total_chunks = 0
    for i in tqdm(range(0, len(to_process), encoding_batch), desc="Processing examples"):
        example_batch = to_process[i:i + encoding_batch]
        print(f"\n[Batch {i//encoding_batch + 1}/{(len(to_process)-1)//encoding_batch + 1}] Processing examples {i} to {i+len(example_batch)}")

        # Stage 1: extract triplets for this mini-batch
        batch_results = []
        for j, ex in enumerate(example_batch):
            try:
                print(f"  [{j+1}/{len(example_batch)}] Extracting triplets for {ex['id']}...", flush=True)
                results = process_example_no_embeddings(ex, chunk_size, nlp)
                batch_results.extend(results)
                print(f"  [{j+1}/{len(example_batch)}] Got {len(results)} chunks", flush=True)
            except Exception as e:
                print(f"Error processing {ex['id']}: {e}")

        if not batch_results:
            continue

        # Stage 2: encode HRRs for this mini-batch (batched GPU call)
        print(f"  Encoding HRRs for {len(batch_results)} chunks ({sum(len(r['triplets']) for r in batch_results)} triplets)...", flush=True)
        batch_results = batch_encode_hrrs(batch_results)
        print(f"  HRR encoding complete", flush=True)

        # Write entire mini-batch to DB in one transaction, then commit.
        # INSERT OR IGNORE means safe to re-run if the job is killed mid-batch.
        chunks_rows   = [(r['chunk_id'], r['title'], r['text']) for r in batch_results]
        triplet_rows  = [(r['chunk_id'], s, rel, o)
                         for r in batch_results for s, rel, o in r['triplets']]
        hrr_rows      = [(r['chunk_id'], blob)
                         for r in batch_results for blob in r.get('hrr_vectors', [])]

        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO chunks (id, title, text) VALUES (?, ?, ?)",
                chunks_rows)
            conn.executemany(
                "INSERT OR IGNORE INTO triplets (chunk_id, subject, relation, object) VALUES (?, ?, ?, ?)",
                triplet_rows)
            conn.executemany(
                "INSERT OR IGNORE INTO hrr_vectors (chunk_id, hrr_vector) VALUES (?, ?)",
                hrr_rows)
        conn.commit()

        total_chunks += len(batch_results)

    conn.close()
    print(f"Processing complete! Processed {total_chunks} chunks total.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing with 10 examples, 300 char chunks...")
        process_dataset(limit=10, chunk_size=300, encoding_batch=32)
    else:
        # For full run, use larger batches for more efficient GPU utilisation
        # Reduce to 64 if you experience GPU OOM or extreme slowness
        process_dataset(limit=None, chunk_size=500, encoding_batch=64)