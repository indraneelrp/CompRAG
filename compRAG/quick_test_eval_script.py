"""
Quick Test Evaluation Script
Uses hardcoded text samples from make_triplets.py for rapid testing.
Perfect for development before running on full HotpotQA dataset.
"""

from make_triplets import get_hardcoded_texts, main_generate_triplets
from encode import embedding_model, doc_triplets2embeddings, doc_embeddings2hrr
from retrieve import initialise_hnsw, add_items
import numpy as np
import time
import spacy

# Load spacy model
nlp = spacy.load("en_core_web_sm")


class QuickTester:
    """Quick tester for retrieval methods using hardcoded samples"""
    
    def __init__(self):
        self.texts = get_hardcoded_texts()
        self.queries = self._create_test_queries()
        
    def _create_test_queries(self):
        """Create test queries for the hardcoded texts"""
        return [
            {
                'query': 'Where was Mother Love Bone formed?',
                'expected_doc_index': 0,
                'answer': 'Seattle, Washington'
            },
            {
                'query': 'How many authors wrote Attention Is All You Need?',
                'expected_doc_index': 1,
                'answer': 'Eight'
            },
            {
                'query': 'What did researchers do to sabotage the convention?',
                'expected_doc_index': 2,
                'answer': 'They added an asterisk to each name and made listing order random'
            },
            {
                'query': 'What was the limitation of recurrent neural networks?',
                'expected_doc_index': 3,
                'answer': 'They struggled to parse longer chunks of text'
            },
            {
                'query': 'What approach did Uszkoreit develop?',
                'expected_doc_index': 4,
                'answer': 'self-attention'
            },
            {
                'query': 'Why was self-attention considered heresy?',
                'expected_doc_index': 5,
                'answer': 'It dumped out all existing neural architectures'
            }
        ]
    
    def test_vanilla_topk(self, k=3):
        """Test vanilla top-k retrieval (baseline)"""
        print("\n" + "="*70)
        print("TEST 1: Vanilla Top-K Retrieval (Baseline)")
        print("="*70)
        
        # Index all texts with vanilla embeddings
        vanilla_vecs = []
        vanilla_ids = []
        
        for i, text in enumerate(self.texts):
            vec = embedding_model.encode([text])[0]
            vanilla_vecs.append(vec)
            vanilla_ids.append(i)
        
        # Build HNSW index
        index = initialise_hnsw(384, 1000)
        add_items(index, vanilla_vecs, vanilla_ids)
        
        # Test each query
        correct = 0
        total = len(self.queries)
        
        for query_data in self.queries:
            query = query_data['query']
            expected_idx = query_data['expected_doc_index']
            
            # Encode query
            query_vec = embedding_model.encode([query])[0]
            query_vec = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
            
            # Search
            start_time = time.time()
            labels, distances = index.knn_query(query_vec, k=k)
            latency = (time.time() - start_time) * 1000
            
            retrieved_indices = labels[0]
            top_match = retrieved_indices[0]
            
            # Check if correct doc is in top-k
            hit = expected_idx in retrieved_indices
            if hit:
                correct += 1
            
            # Print results
            status = "✓" if hit else "✗"
            print(f"\n{status} Query: '{query}'")
            print(f"  Expected doc: {expected_idx}")
            print(f"  Retrieved (top-{k}): {list(retrieved_indices)}")
            print(f"  Top match: {top_match} (distance: {distances[0][0]:.4f})")
            print(f"  Latency: {latency:.2f}ms")
        
        accuracy = correct / total
        print(f"\n{'='*70}")
        print(f"Accuracy (Hit@{k}): {correct}/{total} = {accuracy:.2%}")
        print(f"{'='*70}")
        
        return accuracy
    
    def test_triple_hrr(self, k=3):
        """Test triple-based HRR retrieval (Method 2)"""
        print("\n" + "="*70)
        print("TEST 2: Triple-based HRR Retrieval (Method 2)")
        print("="*70)
        
        # Extract triplets from all texts
        all_triplets = []
        for text in self.texts:
            triplets = main_generate_triplets(nlp, text)
            all_triplets.append(triplets)
        
        print(f"\nExtracted triplets from {len(all_triplets)} documents")
        for i, triplets in enumerate(all_triplets):
            print(f"  Doc {i}: {len(triplets)} triplets")
        
        # Convert triplets to embeddings
        print("\nConverting triplets to embeddings...")
        triplet_embeddings = doc_triplets2embeddings(all_triplets)
        
        # Convert to HRR vectors
        print("Creating HRR vectors...")
        hrr_vectors = doc_embeddings2hrr(triplet_embeddings)
        
        # Build HNSW index with all HRR vectors
        index = initialise_hnsw(384, 10000)
        
        vecs = []
        ids = []
        id_to_doc = {}  # Map HRR vector ID to document index
        
        for doc_idx, doc_hrr_vecs in enumerate(hrr_vectors):
            for triplet_idx, hrr_vec in enumerate(doc_hrr_vecs):
                # Convert tensor to numpy
                hrr_np = hrr_vec.cpu().detach().numpy()
                vecs.append(hrr_np)
                
                # Create unique ID: doc_idx * 10000 + triplet_idx
                unique_id = doc_idx * 10000 + triplet_idx
                ids.append(unique_id)
                id_to_doc[unique_id] = doc_idx
        
        add_items(index, vecs, ids)
        print(f"Indexed {len(vecs)} HRR vectors from {len(all_triplets)} documents")
        
        # Test each query
        correct = 0
        total = len(self.queries)
        
        for query_data in self.queries:
            query = query_data['query']
            expected_idx = query_data['expected_doc_index']
            
            # Extract triplets from query
            query_triplets = main_generate_triplets(nlp, query)
            
            if not query_triplets:
                print(f"\n✗ Query: '{query}'")
                print(f"  No triplets extracted from query!")
                continue
            
            # Convert query triplets to HRR
            query_triplet_embeddings = doc_triplets2embeddings([query_triplets])
            query_hrr = doc_embeddings2hrr(query_triplet_embeddings)
            
            # Search with first query HRR vector
            if len(query_hrr[0]) == 0:
                print(f"\n✗ Query: '{query}'")
                print(f"  No HRR vectors created from query!")
                continue
            
            query_vec = query_hrr[0][0].cpu().detach().numpy()
            query_vec = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
            
            start_time = time.time()
            labels, distances = index.knn_query(query_vec, k=k*10)  # Get more results
            latency = (time.time() - start_time) * 1000
            
            # Map back to document indices
            retrieved_hrr_ids = labels[0]
            retrieved_doc_indices = [id_to_doc[hrr_id] for hrr_id in retrieved_hrr_ids]
            
            # Get unique top-k documents (since multiple HRR vectors per doc)
            unique_docs = []
            seen = set()
            for doc_idx in retrieved_doc_indices:
                if doc_idx not in seen:
                    unique_docs.append(doc_idx)
                    seen.add(doc_idx)
                if len(unique_docs) == k:
                    break
            
            # Check if correct doc is in top-k
            hit = expected_idx in unique_docs
            if hit:
                correct += 1
            
            # Print results
            status = "✓" if hit else "✗"
            print(f"\n{status} Query: '{query}'")
            print(f"  Query triplets: {query_triplets}")
            print(f"  Expected doc: {expected_idx}")
            print(f"  Retrieved (top-{k} docs): {unique_docs}")
            print(f"  Top match: {unique_docs[0] if unique_docs else 'None'}")
            print(f"  Latency: {latency:.2f}ms")
        
        accuracy = correct / total
        print(f"\n{'='*70}")
        print(f"Accuracy (Hit@{k}): {correct}/{total} = {accuracy:.2%}")
        print(f"{'='*70}")
        
        return accuracy
    
    def compare_methods(self, k=3):
        """Compare both methods side-by-side"""
        print("\n" + "="*70)
        print("COMPARISON: Vanilla Top-K vs Triple-HRR")
        print("="*70)
        
        vanilla_acc = self.test_vanilla_topk(k=k)
        triple_acc = self.test_triple_hrr(k=k)
        
        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(f"Vanilla Top-K Accuracy:  {vanilla_acc:.2%}")
        print(f"Triple-HRR Accuracy:     {triple_acc:.2%}")
        
        if triple_acc > vanilla_acc:
            improvement = ((triple_acc - vanilla_acc) / vanilla_acc) * 100
            print(f"\n🎉 Triple-HRR is {improvement:.1f}% better than baseline!")
        elif vanilla_acc > triple_acc:
            degradation = ((vanilla_acc - triple_acc) / vanilla_acc) * 100
            print(f"\n⚠️  Triple-HRR is {degradation:.1f}% worse than baseline")
        else:
            print(f"\n🤝 Both methods perform equally")
        print("="*70)


def main():
    """Run quick tests"""
    tester = QuickTester()
    
    print("="*70)
    print("QUICK RETRIEVAL TEST - Using Hardcoded Samples")
    print("="*70)
    print(f"Number of documents: {len(tester.texts)}")
    print(f"Number of test queries: {len(tester.queries)}")
    
    # Test both methods
    tester.compare_methods(k=3)


if __name__ == "__main__":
    main()