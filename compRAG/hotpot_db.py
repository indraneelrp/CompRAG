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

def process_example(example, chunk_size=500):
    """Process a HotpotQA example into chunks with triplets and HRRs"""
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    
    question_id = example['id']
    titles = example['context']['title']
    paragraphs = example['context']['sentences']
    
    results = []
    chunk_counter = 0
    
    # ONE paragraph at a time with its corresponding title
    for title, sentences in zip(titles, paragraphs):
        para_text = ' '.join(sentences)
        chunks = chunk_text(para_text, chunk_size)
        
        for chunk in chunks:
            chunk_id = f"{question_id}_chunk{chunk_counter}"
            chunk_counter += 1
           
            triplets = main_generate_triplets(nlp, chunk)
            triplets_db = [(chunk_id, s, r, o) for s, r, o in triplets]
            
            hrr_vectors = []
            if triplets:
                try:
                    triplet_embeddings = chunk_triplets2embeddings(triplets)
                    hrr_vecs = chunk_embeddings2hrr(triplet_embeddings)
                    
                    for hrr_vec in hrr_vecs:
                        if isinstance(hrr_vec, torch.Tensor):
                            hrr_vec = hrr_vec.cpu().numpy()
                        hrr_blob = hrr_vec.astype(np.float32).tobytes()
                        hrr_vectors.append((chunk_id, hrr_blob))
                except Exception as e:
                    print(f"HRR error for {chunk_id}: {e}")
            
            results.append({
                'chunk': {'id': chunk_id, 'title': title, 'text': chunk},
                'triplets': triplets_db,
                'hrr': hrr_vectors
            })
    
    return results

# check which HotpotQA examples are already processed (so we can skip them on restart)
def get_processed_ids(conn):
    """Get IDs that are already processed"""
    c = conn.cursor()
    c.execute("SELECT DISTINCT substr(id, 1, instr(id, '_chunk')-1) FROM chunks WHERE id LIKE '%_chunk%'")
    return set(row[0] for row in c.fetchall() if row[0])

def process_dataset(split="train", limit=None, chunk_size=500, batch_size=50, num_workers=8):
    from tqdm import tqdm
    
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")[split]
    total = limit if limit else len(ds)
    
    # cnnect to DB and check what's already done
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
    
    chunks_batch = []
    triplets_batch = []
    hrr_batch = []
    
    # process in parallel using multiple CPU cores
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_example, ex, chunk_size): ex['id'] 
                   for ex in to_process}
        
        with tqdm(total=len(to_process), desc="Processing") as pbar:
            for future in as_completed(futures):
                try:
                    results = future.result()
                    
                    for result in results:
                        chunks_batch.append(result['chunk'])
                        triplets_batch.extend(result['triplets'])
                        hrr_batch.extend(result['hrr'])
                    
                    if len(chunks_batch) >= batch_size:
                        with conn:
                            conn.executemany("INSERT OR IGNORE INTO chunks (id, title, text) VALUES (?, ?, ?)",
                                           [(c['id'], c['title'], c['text']) for c in chunks_batch])
                            conn.executemany("INSERT OR IGNORE INTO triplets (chunk_id, subject, relation, object) VALUES (?, ?, ?, ?)",
                                           triplets_batch)
                            conn.executemany("INSERT OR IGNORE INTO hrr_vectors (chunk_id, hrr_vector) VALUES (?, ?)",
                                           hrr_batch)
                        conn.commit()
                        chunks_batch.clear()
                        triplets_batch.clear()
                        hrr_batch.clear()
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"Error: {e}")
                    pbar.update(1)
    
    # save remaining
    if chunks_batch:
        with conn:
            conn.executemany("INSERT OR IGNORE INTO chunks (id, title, text) VALUES (?, ?, ?)",
                           [(c['id'], c['title'], c['text']) for c in chunks_batch])
            conn.executemany("INSERT OR IGNORE INTO triplets (chunk_id, subject, relation, object) VALUES (?, ?, ?, ?)",
                           triplets_batch)
            conn.executemany("INSERT OR IGNORE INTO hrr_vectors (chunk_id, hrr_vector) VALUES (?, ?)",
                           hrr_batch)
        conn.commit()
    
    conn.close()
    print(f"Processing complete! Total chunks: {len(chunks_batch)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing with 10 examples, 300 char chunks...")
        process_dataset(limit=10, chunk_size=300, batch_size=20, num_workers=4)
    else:
        process_dataset(limit=100, chunk_size=500, batch_size=50, num_workers=4)