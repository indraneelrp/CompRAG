import sqlite3
from compRAG.make_triplets import main_generate_triplets
import spacy
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
import os
import numpy as np
import torch
from compRAG.encode import chunk_triplets2embeddings, chunk_embeddings2hrr
from typing import Dict, Any

load_dotenv()

db_path = os.getenv("HOTPOT_DB")

nlp = spacy.load("en_core_web_sm")


def init_db(db_path):
    if not db_path:
        raise ValueError("Set HOTPOT_DB in your .env file")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Table for chunks (use existing HF IDs)
    c.execute('''
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        title TEXT,
        text TEXT NOT NULL
    )
    ''')

    # Table for triplets
    c.execute('''
    CREATE TABLE IF NOT EXISTS triplets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id TEXT,
        subject TEXT NOT NULL,
        relation TEXT NOT NULL,
        object TEXT NOT NULL,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id)
    )
    ''')

    # Table for embeddings
    c.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        embed_id INTEGER PRIMARY KEY AUTOINCREMENT,
        triplet_id INTEGER,           -- foreign key to triplets.id
        sub_emb BLOB,                 -- serialized subject embedding
        rel_emb BLOB,                 -- serialized relation embedding
        obj_emb BLOB,                 -- serialized object embedding
        FOREIGN KEY (triplet_id) REFERENCES triplets(id)
    )
    ''')

    # Table for HRR vectors
    c.execute('''
    CREATE TABLE IF NOT EXISTS hrr_vectors (
        hrr_id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id TEXT,
        triplet_id TEXT,        
        hrr_vector BLOB,              -- serialized HRR vector
        FOREIGN KEY (chunk_id) REFERENCES chunks(id),
        FOREIGN KEY (triplet_id) REFERENCES triplets(id)
    )
    ''')

    conn.commit()
    return conn

def save_chunks_batch(conn, chunk_batch):
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO chunks (id, title, text) VALUES (?, ?, ?)",
            [(chunk['id'], chunk.get('title', ''), chunk['text']) for chunk in chunk_batch]
        )
    conn.commit()

def save_triplets_batch(conn,triplets_batch):
    with conn:
        conn.executemany("INSERT OR IGNORE INTO triplets (chunk_id, subject, relation, object) VALUES (?, ?, ?, ?)", triplets_batch)

def save_hrr_vectors_batch(conn, hrr_batch):
    """Save HRR vectors to database"""
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO hrr_vectors (chunk_id, hrr_vector) VALUES (?, ?)", 
            hrr_batch
        )
    conn.commit()

def get_processed_chunk_ids(conn):
    """Return set of already processed chunk IDs"""
    c = conn.cursor()
    c.execute("SELECT id FROM chunks")
    return set(row[0] for row in c.fetchall())

def process_chunk(chunk):
    """Run in separate process"""
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    chunk_id = chunk['_id']
    text = chunk['text']
    triplets = main_generate_triplets(nlp, text)
    triplets_db = [(chunk_id, s, r, o) for s, r, o in triplets]
    
    # Generate HRR vectors for this chunk's triplets
    hrr_vectors = []
    if triplets:
        try:
            # Convert triplets to HRR vectors
            triplet_embeddings = chunk_triplets2embeddings(triplets)
            hrr_vecs = chunk_embeddings2hrr(triplet_embeddings)
            print(f"HRR vectors generated: {len(hrr_vecs)}")
            
            # Serialize HRR vectors for database storage
            for hrr_vec in hrr_vecs:
                if isinstance(hrr_vec, torch.Tensor):
                    hrr_vec = hrr_vec.cpu().numpy()
                hrr_blob = hrr_vec.astype(np.float32).tobytes()
                hrr_vectors.append((chunk_id, hrr_blob))
        except Exception as e:
            print(f"Error generating HRR for chunk {chunk_id}: {e}")
    
    return {'id': chunk_id, 'title': chunk.get('title',''), 'text': text}, triplets_db, hrr_vectors


def process_dataset(dataset_name="BeIR/hotpotqa", subset="corpus", limit=None,
                    batch_size=400, num_workers=4):
    ds_dict = load_dataset(dataset_name, subset)
    ds = ds_dict[subset]

    if limit:
        ds = ds.select(range(limit))

    conn = init_db(db_path)
    processed_ids = get_processed_chunk_ids(conn)

    chunks_batch = []
    triplets_batch = []
    hrr_batch = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for chunk in ds:
            chunk: Dict[str, Any]
            if chunk['_id'] in processed_ids:
                continue  # skip already processed
            future = executor.submit(process_chunk, chunk)
            futures[future] = chunk['_id']

        for i, future in enumerate(as_completed(futures), 1):
            chunk_data, triplets_data, hrr_data = future.result()
            chunks_batch.append(chunk_data)
            triplets_batch.extend(triplets_data)
            hrr_batch.extend(hrr_data)

            # Batch insert
            if i % batch_size == 0:
                save_chunks_batch(conn, chunks_batch)
                save_triplets_batch(conn, triplets_batch)
                save_hrr_vectors_batch(conn, hrr_batch)
                chunks_batch.clear()
                triplets_batch.clear()
                hrr_batch.clear()
                print(f"Processed {i} new chunks...")

    # Insert remaining
    if chunks_batch:
        save_chunks_batch(conn, chunks_batch)
        save_triplets_batch(conn, triplets_batch)
        save_hrr_vectors_batch(conn, hrr_batch)

    conn.close()
    print("Dataset processing complete.")

def get_all_hrr_vectors(conn):
    """Retrieve all HRR vectors from database for building search index"""
    c = conn.cursor()
    c.execute('''
        SELECT hrr_id, chunk_id, hrr_vector 
        FROM hrr_vectors 
        WHERE hrr_vector IS NOT NULL
    ''')
    
    vectors = []
    ids = []
    chunk_mapping = {}
    
    for row in c.fetchall():
        hrr_id, chunk_id, hrr_blob = row
        if hrr_blob:
            # Deserialize HRR vector
            hrr_vector = np.frombuffer(hrr_blob, dtype=np.float32)
            vectors.append(hrr_vector)
            ids.append(hrr_id)
            chunk_mapping[hrr_id] = chunk_id
    
    return vectors, ids, chunk_mapping

def get_chunks_by_ids(conn, chunk_ids):
    """Retrieve chunk texts by their IDs"""
    if not chunk_ids:
        return []
    
    c = conn.cursor()
    placeholders = ','.join('?' * len(chunk_ids))
    c.execute(f'''
        SELECT id, title, text FROM chunks 
        WHERE id IN ({placeholders})
    ''', list(chunk_ids))
    
    chunks = []
    for row in c.fetchall():
        chunk_id, title, text = row
        chunks.append({
            'id': chunk_id,
            'title': title or '',
            'text': text
        })
    
    return chunks

def get_database_stats(conn):
    """Get statistics about the database contents"""
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM triplets")
    triplet_count = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM hrr_vectors WHERE hrr_vector IS NOT NULL")
    hrr_count = c.fetchone()[0]
    
    return {
        'chunks': chunk_count,
        'triplets': triplet_count,
        'hrr_vectors': hrr_count
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test with a small dataset
        print("Testing database processing with small dataset...")
        process_dataset(limit=10, batch_size=5, num_workers=2)
        
        # Show stats
        conn = init_db(db_path)
        stats = get_database_stats(conn)
        conn.close()
        
        print(f"Database stats: {stats}")
        print("Test completed successfully!")
    else:
        # Default processing
        process_dataset()

# ds = load_dataset("BeIR/hotpotqa","corpus")
# print(ds["corpus"][0])