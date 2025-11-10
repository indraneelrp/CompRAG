import sqlite3
import ujson as json
import numpy as np
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import os
import subprocess
import sys
sys.path.insert(0, os.path.abspath('.'))


from compRAG.make_triplets import main_generate_triplets
from compRAG.encode import chunk_triplets2embeddings, chunk_embeddings2hrr
import spacy

load_dotenv()
db_path = os.getenv("HOTPOT_DB")

# Load spaCy once
nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

def retrieve_chunks(query, conn, question_id, k=3):
    """
    Retrieve chunks for a SPECIFIC question's context.
    """
    # 1. Extract query triplets
    query_triplets = main_generate_triplets(nlp, query)
    if not query_triplets:
        print(f"Warning: No triplets for query: {query}")
        return []
    
    # 2. Encode query
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
    
    # 3. Search ONLY chunks for this question_id
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
        print(f"Warning: No chunks found for question {question_id}")
        return []
    
    # Sort by similarity
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


def generate_answer(query, chunks):
    """
    Generate answer using Ollama LLM.
    """
    if not chunks:
        return "No relevant information found."
    
    # Build context from top chunks
    context = "\n\n".join([
        f"[{i+1}] {chunk['title']}: {chunk['text']}"
        for i, chunk in enumerate(chunks[:5])  # Use top 5 for context
    ])
    
    # Create prompt
    prompt = f"""You are answering questions based on the provided documents. Read carefully and answer directly.

Documents:
{context}

Query: {query}
Instructions:
- Answer the question directly and concisely
- If the answer is a yes/no question, respond with "yes" or "no" only
- If the answer requires reasoning across documents, do it
- Use only information from the documents above
- Give a short, specific answer (1-2 sentences max)
- If truly no relevant information exists, say "insufficient information"

Answer:"""
    
    try:
        # Call Ollama
        result = subprocess.run(
            ['ollama', 'run', 'llama3.2', prompt],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        answer = result.stdout.strip()
        
        # Clean up common LLM verbosity
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        if answer.startswith("Based on"):
            # Try to get just the answer part
            lines = answer.split('\n')
            for line in lines:
                cleaned = line.strip()
                if cleaned and not cleaned.lower().startswith(("based on", "according to", "the documents")):
                    answer = cleaned
                    break
        return answer
    
    except subprocess.TimeoutExpired:
        print("LLM timeout")
        return "Timeout"
    except Exception as e:
        print(f"Error: {e}")
        return "Error"


def generate_predictions(eval_file, output_file, k=3):
    """
    Generate predictions for all questions in eval file.
    """
    # Load questions
    print(f"Loading questions from {eval_file}...")
    with open(eval_file) as f:
        eval_data = json.load(f)
    
    # Connect to database
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    predictions = []
    
    print(f"\nGenerating predictions for {len(eval_data)} questions...")
    print(f"Retrieving top-{k} chunks per question\n")
    
    for item in tqdm(eval_data):
        question_id = item['id']
        question = item['question']
        gold_answer = item['answer']
        
        try:
            # Retrieve chunks
            chunks = retrieve_chunks(question, conn, question_id, k=k)
            
            # Generate answer
            predicted_answer = generate_answer(question, chunks)
            
            predictions.append({
                'id': question_id,
                'question': question,
                'predicted_answer': predicted_answer,
                'gold_answer': gold_answer,
                'num_chunks_retrieved': len(chunks),
                'retrieved_titles': [c['title'] for c in chunks]
            })
        
        except Exception as e:
            print(f"\nError on {question_id}: {e}")
            predictions.append({
                'id': question_id,
                'question': question,
                'predicted_answer': "",
                'gold_answer': gold_answer,
                'error': str(e)
            })
    
    # Save predictions
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\n✓ Saved {len(predictions)} predictions to {output_file}")
    conn.close()
    
    return predictions


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python generate_predictions.py <eval_file> <output_file> [k]")
        print("\nExample:")
        print("  python generate_predictions.py hotpot_eval_test.json predictions.json 10")
        sys.exit(1)
    
    eval_file = sys.argv[1]
    output_file = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    generate_predictions(eval_file, output_file, k=k)