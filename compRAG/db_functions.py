import sqlite3
from make_triplets import main_generate_triplets
import spacy
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

nlp = spacy.load("en_core_web_sm")



def init_db(db_path="hotpot_qa.db"):
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
        conn.executemany("INSERT INTO triplets (chunk_id, subject, relation, object) VALUES (?, ?, ?, ?)", triplets_batch)

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
    return {'id': chunk_id, 'title': chunk.get('title',''), 'text': text}, triplets_db


def process_dataset(dataset_name="BeIR/hotpotqa", subset="corpus", limit=None,
                    batch_size=100, num_workers=4):
    ds_dict = load_dataset(dataset_name, subset)
    ds = ds_dict[subset]

    if limit:
        ds = ds.select(range(limit))

    conn = init_db()
    processed_ids = get_processed_chunk_ids(conn)

    chunks_batch = []
    triplets_batch = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for chunk in ds:
            if chunk['_id'] in processed_ids:
                continue  # skip already processed
            future = executor.submit(process_chunk, chunk)
            futures[future] = chunk['_id']

        for i, future in enumerate(as_completed(futures), 1):
            chunk_data, triplets_data = future.result()
            chunks_batch.append(chunk_data)
            triplets_batch.extend(triplets_data)

            # Batch insert
            if i % batch_size == 0:
                save_chunks_batch(conn, chunks_batch)
                save_triplets_batch(conn, triplets_batch)
                chunks_batch.clear()
                triplets_batch.clear()
                print(f"Processed {i} new chunks...")

    # Insert remaining
    if chunks_batch:
        save_chunks_batch(conn, chunks_batch)
        save_triplets_batch(conn, triplets_batch)

    conn.close()
    print("Dataset processing complete.")

if __name__ == "__main__":
    process_dataset()

# ds = load_dataset("BeIR/hotpotqa","corpus")
# print(ds["corpus"][0])