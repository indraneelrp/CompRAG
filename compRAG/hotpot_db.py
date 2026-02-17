import sqlite3
from compRAG.make_triplets import main_generate_triplets
from compRAG.encode import chunk_triplets2embeddings, chunk_embeddings2hrr
import spacy
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
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

_worker_nlp = None

def _init_worker():
    """Initialize spaCy model once per worker"""
    global _worker_nlp
    if _worker_nlp is None:
        _worker_nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

def process_example_no_embeddings(example, chunk_size=500):
    """Extract triplets only - no embeddings yet"""
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
            triplets = main_generate_triplets(_worker_nlp, chunk)
            
            results.append({
                'chunk_id': chunk_id,
                'title': title,
                'text': chunk,
                'triplets': triplets
            })
    
    return results

def batch_encode_hrrs(results_batch):
    """Encode all triplets in batch"""
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
        all_embeddings = chunk_triplets2embeddings(all_triplets)
        all_hrrs = chunk_embeddings2hrr(all_embeddings)
        
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
                   batch_size=500, num_workers=8, encoding_batch=128):
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
    
    all_results = []
    
    # Stage 1: Parallel triplet extraction (CPU-bound)
    print("Stage 1: Extracting triplets in parallel...")
    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(process_example_no_embeddings, ex, chunk_size): ex['id'] 
                   for ex in to_process}
        
        with tqdm(total=len(to_process), desc="Extracting triplets") as pbar:
            for future in as_completed(futures):
                try:
                    results = future.result()
                    all_results.extend(results)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error: {e}")
                    pbar.update(1)
    
    # Stage 2: Batch encode HRRs (single-threaded but batched for model efficiency)
    print(f"Stage 2: Encoding {len(all_results)} chunks in batches of {encoding_batch}...")
    chunks_batch = []
    triplets_batch = []
    hrr_batch = []
    
    for i in tqdm(range(0, len(all_results), encoding_batch), desc="Encoding HRRs"):
        batch = all_results[i:i+encoding_batch]
        batch = batch_encode_hrrs(batch)
        
        # Prepare DB inserts
        for result in batch:
            chunk_id = result['chunk_id']
            chunks_batch.append((chunk_id, result['title'], result['text']))
            
            for s, r, o in result['triplets']:
                triplets_batch.append((chunk_id, s, r, o))
            
            for hrr_blob in result.get('hrr_vectors', []):
                hrr_batch.append((chunk_id, hrr_blob))
        
        # Bulk DB write periodically
        if len(chunks_batch) >= batch_size:
            with conn:
                conn.executemany("INSERT OR IGNORE INTO chunks (id, title, text) VALUES (?, ?, ?)", 
                               chunks_batch)
                conn.executemany("INSERT OR IGNORE INTO triplets (chunk_id, subject, relation, object) VALUES (?, ?, ?, ?)", 
                               triplets_batch)
                conn.executemany("INSERT OR IGNORE INTO hrr_vectors (chunk_id, hrr_vector) VALUES (?, ?)", 
                               hrr_batch)
            conn.commit()
            chunks_batch.clear()
            triplets_batch.clear()
            hrr_batch.clear()
    
    # Final write for remaining
    if chunks_batch:
        with conn:
            conn.executemany("INSERT OR IGNORE INTO chunks (id, title, text) VALUES (?, ?, ?)", 
                           chunks_batch)
            conn.executemany("INSERT OR IGNORE INTO triplets (chunk_id, subject, relation, object) VALUES (?, ?, ?, ?)", 
                           triplets_batch)
            conn.executemany("INSERT OR IGNORE INTO hrr_vectors (chunk_id, hrr_vector) VALUES (?, ?)", 
                           hrr_batch)
        conn.commit()
    
    conn.close()
    print(f"Processing complete! Processed {len(all_results)} chunks total.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing with 10 examples, 300 char chunks...")
        process_dataset(limit=10, chunk_size=300, batch_size=50, num_workers=4, encoding_batch=32)
    else:
        # For full run, use larger batches
        process_dataset(limit=None, chunk_size=500, batch_size=1000, num_workers=1, encoding_batch=256)