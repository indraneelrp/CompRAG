import sqlite3
import ujson as json
import numpy as np
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, os.path.abspath('.'))

from compRAG.make_triplets import main_generate_triplets
from compRAG.encode import chunk_triplets2embeddings, chunk_embeddings2hrr
from compRAG.retrieve import initialise_hnsw, add_items, local_similarity_graph_match
import spacy

load_dotenv()
db_path = os.getenv("HOTPOT_DB")

# Global variables
nlp = None
conn = None
model = None
tokenizer = None
index = None
hrr_id_to_embedding = {}
vector_to_chunk = {}

def init_worker(use_graph):
    """Initialize per-worker resources"""
    global nlp, conn, model, tokenizer, index, hrr_id_to_embedding, vector_to_chunk
    
    print(f"Initializing worker (PID: {os.getpid()})...")
    
    # Load spaCy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    conn = sqlite3.connect(db_path)
    
    # Load HuggingFace model
    print("Loading HuggingFace model (flan-t5-base)...")
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
    print("Model loaded")
    
    # Build HNSW index if using graph retrieval
    if use_graph:
        print("Building HNSW index for graph retrieval...")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT hrr_id, chunk_id, hrr_vector 
            FROM hrr_vectors 
            WHERE hrr_vector IS NOT NULL
        """)
        hrr_data = cursor.fetchall()
        
        if not hrr_data:
            raise RuntimeError("No HRR vectors found in database")
        
        vectors = []
        vector_ids = []
        hrr_id_to_embedding = {}
        vector_to_chunk = {}
        
        for hrr_id, chunk_id, hrr_blob in hrr_data:
            hrr_vector = np.frombuffer(hrr_blob, dtype=np.float32)
            vectors.append(hrr_vector)
            vector_ids.append(hrr_id)
            hrr_id_to_embedding[hrr_id] = hrr_vector
            vector_to_chunk[hrr_id] = chunk_id
        
        # Use function from retrieve.py
        index = initialise_hnsw(dim=384, max_elems=len(vectors))
        add_items(index, vectors, vector_ids)
        
        print(f"Index built with {len(vectors)} vectors")
    else:
        index = None

def retrieve_chunks_baseline(query, question_id, k=10):
    """Baseline retrieval: Top-k similarity within question's chunks"""
    global nlp, conn
    
    # Extract query triplets
    query_triplets = main_generate_triplets(nlp, query)
    if not query_triplets:
        return []
    
    # Encode query
    query_embeddings = chunk_triplets2embeddings(query_triplets)
    query_hrrs = chunk_embeddings2hrr(query_embeddings)
    
    query_hrr_list = []
    for hrr in query_hrrs:
        if isinstance(hrr, torch.Tensor):
            hrr = hrr.cpu().numpy()
        query_hrr_list.append(hrr)
    
    if not query_hrr_list:
        return []
    
    query_hrr = np.mean(query_hrr_list, axis=0)
    
    # Search chunks for this question
    c = conn.cursor()
    c.execute("""
        SELECT chunk_id, hrr_vector 
        FROM hrr_vectors 
        WHERE chunk_id LIKE ?
    """, (f"{question_id}_chunk%",))
    
    similarities = []
    for row in c.fetchall():
        chunk_id, hrr_blob = row
        chunk_hrr = np.frombuffer(hrr_blob, dtype=np.float32)
        
        norm_query = np.linalg.norm(query_hrr)
        norm_chunk = np.linalg.norm(chunk_hrr)
        
        if norm_query > 0 and norm_chunk > 0:
            sim = np.dot(query_hrr, chunk_hrr) / (norm_query * norm_chunk)
            similarities.append((chunk_id, float(sim)))
    
    if not similarities:
        return []
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_chunk_ids = [chunk_id for chunk_id, _ in similarities[:k]]
    
    # Fetch chunk details
    chunks = []
    for chunk_id in top_chunk_ids:
        c.execute("SELECT id, title, text FROM chunks WHERE id=?", (chunk_id,))
        row = c.fetchone()
        if row:
            chunks.append({
                'id': row[0],
                'title': row[1],
                'text': row[2]
            })
    
    return chunks

def retrieve_chunks_graph(query, question_id, depth=2, seed_k=10, branching_k=4):
    """Graph-based retrieval: BFS on similarity graph using retrieve.py"""
    global nlp, conn, index, hrr_id_to_embedding, vector_to_chunk
    
    # Extract query triplets
    query_triplets = main_generate_triplets(nlp, query)
    if not query_triplets:
        return []
    
    # Encode query
    query_embeddings = chunk_triplets2embeddings(query_triplets)
    query_hrrs = chunk_embeddings2hrr(query_embeddings)
    
    query_hrr_list = []
    for hrr in query_hrrs:
        if isinstance(hrr, torch.Tensor):
            hrr = hrr.cpu().numpy()
        query_hrr_list.append(hrr)
    
    if not query_hrr_list:
        return []
    
    # Use function from retrieve.py
    hrr_ids = local_similarity_graph_match(
        query_hrr_vectors=query_hrr_list,
        index=index,
        label_to_embedding=hrr_id_to_embedding,
        depth=depth,
        seed_k=seed_k,
        branching_k=branching_k
    )
    
    # Map to chunk IDs
    chunk_ids = set()
    for hrr_id in hrr_ids:
        if hrr_id in vector_to_chunk:
            chunk_ids.add(vector_to_chunk[hrr_id])
    
    # Filter to only this question's chunks
    question_chunk_ids = [cid for cid in chunk_ids if cid.startswith(f"{question_id}_chunk")]
    
    if not question_chunk_ids:
        return []
    
    # Fetch chunk details
    c = conn.cursor()
    chunks = []
    for chunk_id in question_chunk_ids:
        c.execute("SELECT id, title, text FROM chunks WHERE id=?", (chunk_id,))
        row = c.fetchone()
        if row:
            chunks.append({
                'id': row[0],
                'title': row[1],
                'text': row[2]
            })
    
    return chunks

def generate_answer_hf(query, chunks, max_length=100):
    """Generate answer using HuggingFace model"""
    global model, tokenizer
    
    if not chunks:
        return "No relevant information found."
    
    # Use top 3 chunks, truncate text
    context = "\n\n".join([
        f"[{i+1}] {chunk['title']}: {chunk['text'][:500]}"
        for i, chunk in enumerate(chunks[:3])
    ])
    
    prompt = f"""Answer the question based on the documents.

Documents:
{context}

Question: {query}

Instructions:
- Answer directly and concisely
- For yes/no questions, respond with just "yes" or "no"
- Use only information from the documents
- Keep answer to 1-2 sentences

Answer:"""
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=2,
                early_stopping=True,
                temperature=0.7
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
    
    except Exception as e:
        return f"Error: {str(e)}"

def process_question(item, k=10, use_graph=False, graph_params=None):
    """Process a single question"""
    question_id = item['id']
    question = item['question']
    gold_answer = item['answer']
    
    try:
        # Retrieve chunks based on mode
        if use_graph:
            params = graph_params or {'depth': 2, 'seed_k': 10, 'branching_k': 4}
            chunks = retrieve_chunks_graph(
                question, 
                question_id, 
                depth=params['depth'],
                seed_k=params['seed_k'],
                branching_k=params['branching_k']
            )
        else:
            chunks = retrieve_chunks_baseline(question, question_id, k=k)
        
        # Generate answer
        predicted_answer = generate_answer_hf(question, chunks)
        
        return {
            'id': question_id,
            'question': question,
            'predicted_answer': predicted_answer,
            'gold_answer': gold_answer,
            'num_chunks_retrieved': len(chunks),
            'retrieved_titles': [c['title'] for c in chunks],
            'retrieval_method': 'graph' if use_graph else 'baseline'
        }
    
    except Exception as e:
        return {
            'id': question_id,
            'question': question,
            'predicted_answer': "",
            'gold_answer': gold_answer,
            'error': str(e),
            'retrieval_method': 'graph' if use_graph else 'baseline'
        }

def generate_predictions(eval_file, output_file, k=10, use_graph=False, graph_params=None):
    """Generate predictions (single process due to GPU constraints)"""
    
    # Load questions
    print(f"Loading questions from {eval_file}...")
    with open(eval_file) as f:
        eval_data = json.load(f)
    
    print(f"Total questions: {len(eval_data)}")
    
    mode = "GRAPH" if use_graph else "BASELINE"
    print(f"Mode: {mode}")
    
    if use_graph:
        params = graph_params or {'depth': 2, 'seed_k': 10, 'branching_k': 4}
        print(f"Graph parameters: depth={params['depth']}, seed_k={params['seed_k']}, branching_k={params['branching_k']}")
    else:
        print(f"Retrieving top-{k} chunks per question")
    
    print()
    
    # Initialize resources
    init_worker(use_graph)
    
    # Process questions sequentially
    predictions = []
    for item in tqdm(eval_data, desc="Processing questions"):
        pred = process_question(item, k=k, use_graph=use_graph, graph_params=graph_params)
        predictions.append(pred)
    
    # Save predictions
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n✓ Saved {len(predictions)} predictions to {output_file}")
    
    # Quick stats
    errors = sum(1 for p in predictions if 'error' in p)
    print(f"Successful: {len(predictions) - errors}")
    print(f"Errors: {errors}")
    
    return predictions

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python generate_preds.py <eval_file> <output_file> [k] [--graph] [--depth D] [--seed_k S] [--branching_k B]")
        print("\nExamples:")
        print("  # Baseline mode")
        print("  python generate_preds.py hotpot_eval_test.json predictions_baseline.json 10")
        print()
        print("  # Graph mode")
        print("  python generate_preds.py hotpot_eval_test.json predictions_graph.json 10 --graph")
        print()
        print("  # Graph mode with custom parameters")
        print("  python generate_preds.py hotpot_eval_test.json predictions_graph.json 10 --graph --depth 3 --seed_k 7 --branching_k 5")
        sys.exit(1)
    
    eval_file = sys.argv[1]
    output_file = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    # Check for graph mode flag
    use_graph = '--graph' in sys.argv
    
    # Parse graph parameters
    graph_params = None
    if use_graph:
        graph_params = {'depth': 2, 'seed_k': 10, 'branching_k': 4}
        
        try:
            if '--depth' in sys.argv:
                idx = sys.argv.index('--depth')
                graph_params['depth'] = int(sys.argv[idx + 1])
            
            if '--seed_k' in sys.argv:
                idx = sys.argv.index('--seed_k')
                graph_params['seed_k'] = int(sys.argv[idx + 1])
            
            if '--branching_k' in sys.argv:
                idx = sys.argv.index('--branching_k')
                graph_params['branching_k'] = int(sys.argv[idx + 1])
        except (IndexError, ValueError) as e:
            print(f"Error parsing graph parameters: {e}")
            sys.exit(1)
    
    generate_predictions(eval_file, output_file, k=k, use_graph=use_graph, graph_params=graph_params)
