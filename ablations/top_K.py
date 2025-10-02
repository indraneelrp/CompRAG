"""
Top-K Baseline Retriever for RAG
This retrieves the k most similar chunks to a query using cosine similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import torch


class TopKRetriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        """
        Initialize the top-k retriever.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = SentenceTransformer(model_name, device=device)
        self.chunks = []
        self.chunk_embeddings = None
        
    def index_chunks(self, chunks: List[str]):
        """
        Create embeddings for all chunks (do this once during setup).
        
        Args:
            chunks: List of text chunks to index
        """
        print(f"Indexing {len(chunks)} chunks...")
        self.chunks = chunks
        
        # Encode all chunks in batches for efficiency
        self.chunk_embeddings = self.model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        self.chunk_embeddings = self.chunk_embeddings / np.linalg.norm(
            self.chunk_embeddings, axis=1, keepdims=True
        )
        
        print(f"Indexing complete. Shape: {self.chunk_embeddings.shape}")
    
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most similar chunks to the query.
        
        Args:
            query: The query string
            k: Number of chunks to retrieve
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by similarity
        """
        if self.chunk_embeddings is None:
            raise ValueError("Must call index_chunks() before retrieval")
        
        # Encode the query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarity with all chunks
        similarities = np.dot(self.chunk_embeddings, query_embedding)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return chunks with their similarity scores
        results = [
            (self.chunks[idx], float(similarities[idx]))
            for idx in top_k_indices
        ]
        
        return results
    
    def retrieve_with_indices(self, query: str, k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Retrieve top-k chunks with their original indices.
        Useful for evaluation where you need to map back to source documents.
        
        Args:
            query: The query string
            k: Number of chunks to retrieve
            
        Returns:
            List of (index, chunk, similarity_score) tuples
        """
        if self.chunk_embeddings is None:
            raise ValueError("Must call index_chunks() before retrieval")
        
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        similarities = np.dot(self.chunk_embeddings, query_embedding)
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = [
            (int(idx), self.chunks[idx], float(similarities[idx]))
            for idx in top_k_indices
        ]
        
        return results


# Example usage
if __name__ == "__main__":
    # Sample chunks (in practice, these come from your document chunking)
    sample_chunks = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "Ludwig van Beethoven was born in Bonn, Germany in 1770.",
        "The Moonlight Sonata was composed by Beethoven in 1801.",
        "Paris is the capital and most populous city of France.",
        "Bonn is a city on the banks of the Rhine in Germany.",
        "The Eiffel Tower was constructed in 1889 for the World's Fair.",
        "Beethoven is considered one of the greatest composers in history.",
    ]
    
    # Initialize retriever
    retriever = TopKRetriever()
    
    # Index the chunks
    retriever.index_chunks(sample_chunks)
    
    # Example query requiring multi-hop reasoning
    query = "Where was the composer of Moonlight Sonata born?"
    
    print(f"\nQuery: {query}")
    print("\nTop-3 Retrieved Chunks:")
    print("-" * 80)
    
    results = retriever.retrieve(query, k=3)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"{i}. [Score: {score:.4f}] {chunk}")
    
    print("\n" + "="*80)
    print("Note: Top-k retrieval may miss the connection between")
    print("'Moonlight Sonata' → 'Beethoven' → 'Bonn'")
    print("This is what your graph-based methods aim to improve!")