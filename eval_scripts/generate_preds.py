import ujson as json
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

sys.path.insert(0, os.path.abspath('.'))
from main import CompRAGSystem

load_dotenv()

# Global variables
model = None
tokenizer = None
comprag_system = None

def init_system(use_graph):
    """Initialize CompRAG system and HuggingFace model"""
    global model, tokenizer, comprag_system

    print(f"Initializing system...")

    # Create args namespace for CompRAGSystem
    args = argparse.Namespace(graph=use_graph, see_chunks=False)

    # Initialize CompRAG system (reuses existing database and index building logic)
    comprag_system = CompRAGSystem(args=args)
    comprag_system.build_search_index(dim=384, max_elements=1000000)

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
    """Process a single question using CompRAGSystem"""
    global comprag_system

    question_id = item['id']
    question = item['question']
    gold_answer = item['answer']

    try:
        query_triplets = comprag_system.process_query(question)

        if not query_triplets:
            return {
                'id': question_id,
                'question': question,
                'predicted_answer': "",
                'gold_answer': gold_answer,
                'num_chunks_retrieved': 0,
                'retrieved_titles': [],
                'retrieval_method': 'graph' if use_graph else 'baseline'
            }

        # Convert to HRR vectors
        query_hrr_vectors = comprag_system.query_to_hrr_vectors(query_triplets)

        if not query_hrr_vectors:
            return {
                'id': question_id,
                'question': question,
                'predicted_answer': "",
                'gold_answer': gold_answer,
                'num_chunks_retrieved': 0,
                'retrieved_titles': [],
                'retrieval_method': 'graph' if use_graph else 'baseline'
            }

        # Retrieve chunks
        if use_graph:
            from compRAG.retrieve import local_similarity_graph_match
            params = graph_params or {'depth': 2, 'seed_k': 10, 'branching_k': 4}

            label_list = local_similarity_graph_match(
                query_hrr_vectors,
                comprag_system.index,
                comprag_system.hrr_id_to_embedding,
                depth=params['depth'],
                seed_k=params['seed_k'],
                branching_k=params['branching_k']
            )

            chunk_ids = set()
            for hrr_id in label_list:
                if hrr_id in comprag_system.vector_to_chunk:
                    chunk_id = comprag_system.vector_to_chunk[hrr_id]
                    # Filter to only this question's chunks
                    if chunk_id.startswith(f"{question_id}_chunk"):
                        chunk_ids.add(chunk_id)
        else:
            # Use baseline retrieval from CompRAGSystem
            all_chunk_ids = comprag_system.retrieve_similar_chunks(query_hrr_vectors, k=k)
            # Filter to only this question's chunks
            chunk_ids = set(cid for cid in all_chunk_ids if cid.startswith(f"{question_id}_chunk"))

        # Get chunk contexts using CompRAGSystem method
        contexts = comprag_system.get_chunk_contexts(chunk_ids)

        # Generate answer using HuggingFace
        predicted_answer = generate_answer_hf(question, contexts)

        return {
            'id': question_id,
            'question': question,
            'predicted_answer': predicted_answer,
            'gold_answer': gold_answer,
            'num_chunks_retrieved': len(contexts),
            'retrieved_titles': [c['title'] for c in contexts],
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
    """Generate predictions using CompRAGSystem"""

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

    # Initialize system (calls CompRAGSystem and loads HF model)
    init_system(use_graph)

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
