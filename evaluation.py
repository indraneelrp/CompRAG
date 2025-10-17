"""
Evaluation Script for CompRAG Project
Evaluates retrieval quality for multi-hop question answering on HotpotQA dataset.

Compares three retrieval methods:
1. Baseline: Top-K retrieval (vanilla embeddings)
2. Method 1: Graph-based retrieval
3. Method 2: Triple-based HRR retrieval
"""

import sqlite3
import numpy as np
import time
from typing import List, Dict, Tuple, Any
from collections import Counter
import json
from datasets import load_dataset
from dotenv import load_dotenv
import os

# Import your retrieval methods
from ablations.top_K import TopKRetriever
# from ablations.graph_walk import GraphRetriever  # Uncomment when implemented
# from ablations.triple_hrr import TripleHRRRetriever  # Uncomment when implemented

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()


class HotpotQAEvaluator:
    """
    Evaluator for multi-hop RAG systems on HotpotQA dataset.
    Focuses on retrieval quality metrics.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize evaluator.
        
        Args:
            db_path: Path to SQLite database with chunks
        """
        self.db_path = db_path or os.getenv("HOTPOT_DB")
        if not self.db_path:
            raise ValueError("Set HOTPOT_DB in .env file")
        
        # For semantic similarity evaluation
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Map chunk IDs to titles for supporting fact matching
        self.chunk_id_to_title = self._load_chunk_titles()
    
    def _load_chunk_titles(self) -> Dict[str, str]:
        """Load mapping of chunk IDs to their titles"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, title FROM chunks")
        mapping = {row[0]: row[1] for row in c.fetchall()}
        conn.close()
        return mapping
    
    # ==================== RETRIEVAL METRICS ====================
    
    def compute_recall_at_k(self, 
                           retrieved_chunks: List[Dict], 
                           supporting_facts: List[List[Any]]) -> float:
        """
        Compute Recall@k: What fraction of gold supporting facts were retrieved?
        
        Args:
            retrieved_chunks: List of retrieved chunk dicts with 'id', 'title', 'text'
            supporting_facts: List of [title, sentence_id] pairs from HotpotQA
            
        Returns:
            Recall score between 0 and 1
        """
        if not supporting_facts:
            return 0.0
        
        # Extract titles from retrieved chunks
        retrieved_titles = set()
        for chunk in retrieved_chunks:
            chunk_id = chunk.get('id')
            title = chunk.get('title') or self.chunk_id_to_title.get(chunk_id, '')
            if title:
                retrieved_titles.add(title.lower().strip())
        
        # Extract gold titles from supporting facts
        gold_titles = set()
        for fact in supporting_facts:
            title = fact[0].lower().strip()
            gold_titles.add(title)
        
        # Calculate recall
        if not gold_titles:
            return 0.0
        
        overlap = retrieved_titles.intersection(gold_titles)
        recall = len(overlap) / len(gold_titles)
        
        return recall
    
    def compute_precision_at_k(self, 
                              retrieved_chunks: List[Dict], 
                              supporting_facts: List[List[Any]]) -> float:
        """
        Compute Precision@k: What fraction of retrieved chunks are relevant?
        
        Args:
            retrieved_chunks: List of retrieved chunk dicts
            supporting_facts: List of [title, sentence_id] pairs
            
        Returns:
            Precision score between 0 and 1
        """
        if not retrieved_chunks:
            return 0.0
        
        # Extract titles from retrieved chunks
        retrieved_titles = []
        for chunk in retrieved_chunks:
            chunk_id = chunk.get('id')
            title = chunk.get('title') or self.chunk_id_to_title.get(chunk_id, '')
            if title:
                retrieved_titles.append(title.lower().strip())
        
        # Extract gold titles
        gold_titles = set()
        for fact in supporting_facts:
            title = fact[0].lower().strip()
            gold_titles.add(title)
        
        # Calculate precision
        relevant_count = sum(1 for title in retrieved_titles if title in gold_titles)
        precision = relevant_count / len(retrieved_chunks)
        
        return precision
    
    def compute_multi_hop_coverage(self, 
                                   retrieved_chunks: List[Dict], 
                                   supporting_facts: List[List[Any]]) -> float:
        """
        Multi-Hop Coverage: Binary metric - did we retrieve facts from BOTH hops?
        Critical for multi-hop reasoning!
        
        Args:
            retrieved_chunks: List of retrieved chunks
            supporting_facts: Gold supporting facts (usually from 2 different pages)
            
        Returns:
            1.0 if all unique gold titles are covered, 0.0 otherwise
        """
        if not supporting_facts:
            return 0.0
        
        # Get unique gold titles (usually 2 for multi-hop)
        gold_titles = set()
        for fact in supporting_facts:
            title = fact[0].lower().strip()
            gold_titles.add(title)
        
        # Get retrieved titles
        retrieved_titles = set()
        for chunk in retrieved_chunks:
            chunk_id = chunk.get('id')
            title = chunk.get('title') or self.chunk_id_to_title.get(chunk_id, '')
            if title:
                retrieved_titles.add(title.lower().strip())
        
        # Check if ALL gold titles are present
        coverage = 1.0 if gold_titles.issubset(retrieved_titles) else 0.0
        
        return coverage
    
    def compute_mrr(self, 
                   retrieved_chunks: List[Dict], 
                   supporting_facts: List[List[Any]]) -> float:
        """
        Mean Reciprocal Rank: How highly ranked was the first relevant chunk?
        
        Args:
            retrieved_chunks: List of retrieved chunks (ordered by relevance)
            supporting_facts: Gold supporting facts
            
        Returns:
            Reciprocal rank of first relevant chunk (0 if none found)
        """
        if not retrieved_chunks or not supporting_facts:
            return 0.0
        
        # Get gold titles
        gold_titles = set()
        for fact in supporting_facts:
            title = fact[0].lower().strip()
            gold_titles.add(title)
        
        # Find first relevant chunk
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            chunk_id = chunk.get('id')
            title = chunk.get('title') or self.chunk_id_to_title.get(chunk_id, '')
            if title and title.lower().strip() in gold_titles:
                return 1.0 / rank
        
        return 0.0
    
    # ==================== ANSWER QUALITY METRICS ====================
    
    def normalize_answer(self, text: str) -> str:
        """Normalize answer text for comparison (lowercase, remove articles, etc.)"""
        import re
        
        # Lowercase
        text = text.lower()
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def compute_exact_match(self, predicted: str, gold: str) -> float:
        """
        Exact Match: Does the predicted answer exactly match the gold answer?
        
        Returns:
            1.0 if match, 0.0 otherwise
        """
        pred_normalized = self.normalize_answer(predicted)
        gold_normalized = self.normalize_answer(gold)
        
        return 1.0 if pred_normalized == gold_normalized else 0.0
    
    def compute_f1(self, predicted: str, gold: str) -> float:
        """
        F1 Score: Token-level overlap between predicted and gold answers.
        
        Returns:
            F1 score between 0 and 1
        """
        pred_tokens = self.normalize_answer(predicted).split()
        gold_tokens = self.normalize_answer(gold).split()
        
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        # Count common tokens
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def compute_semantic_similarity(self, predicted: str, gold: str) -> float:
        """
        Semantic Similarity: Embedding-based similarity between answers.
        Captures semantic equivalence better than token overlap.
        
        Returns:
            Cosine similarity between 0 and 1
        """
        pred_embedding = self.embedding_model.encode([predicted])
        gold_embedding = self.embedding_model.encode([gold])
        
        similarity = cosine_similarity(pred_embedding, gold_embedding)[0][0]
        
        # Normalize to [0, 1] range
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    # ==================== EVALUATION PIPELINE ====================
    
    def evaluate_retriever(self, 
                          retriever: Any, 
                          dataset: Any,
                          k: int = 5,
                          limit: int = None,
                          verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate a retriever on HotpotQA dataset.
        
        Args:
            retriever: Retriever object with retrieve() method
            dataset: HotpotQA dataset (from datasets library)
            k: Number of chunks to retrieve
            limit: Limit number of examples (for testing)
            verbose: Print progress
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            'recall@k': [],
            'precision@k': [],
            'multi_hop_coverage': [],
            'mrr': [],
            'latencies': []
        }
        
        # Limit dataset if specified
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        if verbose:
            print(f"\nEvaluating on {len(dataset)} examples...")
        
        for i, example in enumerate(dataset):
            if verbose and (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} examples...")
            
            question = example['question']
            supporting_facts = example['supporting_facts']
            
            # Retrieve chunks and measure latency
            start_time = time.time()
            retrieved = retriever.retrieve(question, k=k)
            latency = time.time() - start_time
            
            # Compute metrics
            recall = self.compute_recall_at_k(retrieved, supporting_facts)
            precision = self.compute_precision_at_k(retrieved, supporting_facts)
            multi_hop = self.compute_multi_hop_coverage(retrieved, supporting_facts)
            mrr = self.compute_mrr(retrieved, supporting_facts)
            
            # Store results
            results['recall@k'].append(recall)
            results['precision@k'].append(precision)
            results['multi_hop_coverage'].append(multi_hop)
            results['mrr'].append(mrr)
            results['latencies'].append(latency)
        
        # Aggregate results
        aggregated = {
            f'Recall@{k}': np.mean(results['recall@k']),
            f'Precision@{k}': np.mean(results['precision@k']),
            'Multi-Hop Coverage': np.mean(results['multi_hop_coverage']),
            'MRR': np.mean(results['mrr']),
            'Avg Latency (ms)': np.mean(results['latencies']) * 1000
        }
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            for metric, value in aggregated.items():
                print(f"{metric:25s}: {value:.4f}")
            print("="*60)
        
        return aggregated
    
    def compare_retrievers(self, 
                          retrievers: Dict[str, Any],
                          dataset: Any,
                          k: int = 5,
                          limit: int = None) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple retrievers side-by-side.
        
        Args:
            retrievers: Dict mapping retriever names to retriever objects
            dataset: HotpotQA dataset
            k: Number of chunks to retrieve
            limit: Limit number of examples
            
        Returns:
            Dictionary mapping retriever names to their metrics
        """
        all_results = {}
        
        for name, retriever in retrievers.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print(f"{'='*60}")
            
            results = self.evaluate_retriever(
                retriever=retriever,
                dataset=dataset,
                k=k,
                limit=limit,
                verbose=True
            )
            
            all_results[name] = results
        
        # Print comparison table
        self._print_comparison_table(all_results, k)
        
        return all_results
    
    def _print_comparison_table(self, results: Dict[str, Dict[str, float]], k: int):
        """Print a comparison table of results"""
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        
        # Get all metric names
        metric_names = list(next(iter(results.values())).keys())
        
        # Print header
        print(f"{'Metric':<30s} ", end='')
        for retriever_name in results.keys():
            print(f"{retriever_name:>15s} ", end='')
        print()
        print("-" * 80)
        
        # Print each metric row
        for metric in metric_names:
            print(f"{metric:<30s} ", end='')
            for retriever_name in results.keys():
                value = results[retriever_name][metric]
                print(f"{value:>15.4f} ", end='')
            print()
        
        print("="*80)
        
        # Highlight best performer for key metrics
        key_metrics = [f'Recall@{k}', 'Multi-Hop Coverage']
        print("\n🏆 BEST PERFORMERS:")
        for metric in key_metrics:
            best_retriever = max(results.items(), key=lambda x: x[1][metric])
            print(f"  {metric}: {best_retriever[0]} ({best_retriever[1][metric]:.4f})")
        print()


# ==================== MAIN EXECUTION ====================

def main():
    """Main evaluation script"""
    
    # Load HotpotQA dev set (distractor setting)
    print("Loading HotpotQA dataset...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    # Initialize evaluator
    evaluator = HotpotQAEvaluator()
    
    # Initialize retrievers
    print("\nInitializing retrievers...")
    
    # Baseline: Top-K
    baseline_retriever = TopKRetriever()
    baseline_retriever.build_index()
    
    # TODO: Initialize other retrievers when implemented
    # graph_retriever = GraphRetriever()
    # graph_retriever.build_index()
    
    # triple_retriever = TripleHRRRetriever()
    # triple_retriever.build_index()
    
    retrievers = {
        'Baseline (Top-K)': baseline_retriever,
        # 'Method 1 (Graph)': graph_retriever,
        # 'Method 2 (Triple-HRR)': triple_retriever,
    }
    
    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.compare_retrievers(
        retrievers=retrievers,
        dataset=dataset,
        k=5,
        limit=100  # Start with 100 examples for testing, remove limit for full eval
    )
    
    # Save results
    output_file = 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    main()