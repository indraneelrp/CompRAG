import sqlite3
from make_triplets import main_generate_triplets
import spacy
from datasets import load_dataset


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

def save_chunk(conn, chunk):
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO chunks (id, title, text) VALUES (?, ?, ?)",
        (chunk['id'], chunk.get('title', ''), chunk['text'])
    )
    conn.commit()

def save_triplets(conn, chunk_id, triplets):
    c = conn.cursor()
    for subj, rel, obj in triplets:
        c.execute(
            "INSERT INTO triplets (chunk_id, subject, relation, object) VALUES (?, ?, ?, ?)",
            (chunk_id, subj, rel, obj)
        )
    conn.commit()

def process_dataset(dataset_name="BeIR/hotpotqa", subset = "corpus", split="corpus", limit=None):
    ds = load_dataset(dataset_name,subset)
    ds = ds[split]
    if limit:
        ds = ds.select(range(limit))
    conn = init_db()
    
    for chunk in ds:
        save_chunk(conn, {'id': chunk['_id'], 'title': chunk.get('title', ''), 'text': chunk['text']})
        triplets = main_generate_triplets(nlp, chunk['text'])
        save_triplets(conn, chunk['_id'], triplets)
    
    conn.close()

if __name__ == "__main__":
    process_dataset()

# ds = load_dataset("BeIR/hotpotqa","corpus")
# print(ds["corpus"][0])