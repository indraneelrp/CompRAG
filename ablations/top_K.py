"""
Top-K Baseline Retriever for RAG
Properly integrated with existing compRAG modules
"""

import sqlite3
import numpy as np
from typing import List, Tuple
import os
from dotenv import load_dotenv

# Import from existing compRAG modules
from compRAG.retrieve import initialise_hnsw, add_items
from compRAG.encode import EMBEDDING_MODEL  # Use shared embedding model instance

load_dotenv()

class TopKRetriever:
    """
    Baseline top-k retriever using vanilla text embeddings.
    Uses existing HNSW utilities from retrieve.py and shared encoder from encode.py.
    """
    
    def __init__(self, db_path: str = ""):
        """
        Initialize the top-k retriever.
        
        Args:
            db_path: Path to SQLite database (defaults to HOTPOT_DB env var)
        """
        resolved_env = os.getenv("HOTPOT_DB")
        self.db_path: str = (db_path or (resolved_env if resolved_env else "hotpot.db"))
        if not self.db_path:
            raise ValueError("Set HOTPOT_DB in your .env file or pass db_path")
        
        # Use shared embedding model from encode.py (dim 384)
        self.embedding_model = EMBEDDING_MODEL
        self.embed_dim = 384
        
        self.index = None
        self.id_to_chunk = {}  # Map HNSW IDs to chunk data
        
    def _load_chunks_from_db(self) -> List[dict]:
        """Load all chunks from database (using schema from db_functions.py)"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, title, text FROM chunks")
        chunks = []
        for row in c.fetchall():
            chunks.append({
                'id': row[0],
                'title': row[1],
                'text': row[2]
            })
        conn.close()
        return chunks
    
    def build_index(self, max_elements: int = 100000):
        """
        Build HNSW index from chunks in database.
        Uses existing initialise_hnsw and add_items from retrieve.py.
        
        Args:
            max_elements: Maximum number of elements in index
        """
        print("Loading chunks from database...")
        chunks = self._load_chunks_from_db()
        
        print(f"Encoding {len(chunks)} chunks...")
        
        # Use existing HNSW initialization from retrieve.py
        self.index = initialise_hnsw(dim=self.embed_dim, max_elems=max_elements)
        
        # Encode all chunks
        vecs = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            # Use shared embedding model from encode.py
            vec = self.embedding_model.encode([chunk['text']])[0]
            vecs.append(vec)
            
            # Use sequential IDs for HNSW
            hnsw_id = i
            ids.append(hnsw_id)
            
            # Map back to original chunk data
            self.id_to_chunk[hnsw_id] = chunk
        
        # Use existing add_items function from retrieve.py
        add_items(self.index, vecs, ids)
        
        print(f"Index built with {len(chunks)} chunks")
    
    def retrieve(self, query: str, k: int = 5) -> List[dict]:
        """
        Retrieve top-k most similar chunks to the query.
        
        Args:
            query: The query string
            k: Number of chunks to retrieve
            
        Returns:
            List of chunk dictionaries with 'id', 'title', 'text', 'distance', 'similarity'
        """
        if self.index is None:
            raise ValueError("Must call build_index() before retrieval")
        
        # Encode query using shared embedding model
        query_vec = self.embedding_model.encode([query])[0]
        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        
        # Search using HNSW (same pattern as retrieve.py)
        labels, distances = self.index.knn_query(query_vec, k=k)
        
        # Map HNSW IDs back to chunk data
        results = []
        for hnsw_id, distance in zip(labels[0], distances[0]):
            chunk = self.id_to_chunk[hnsw_id].copy()
            chunk['distance'] = float(distance)
            chunk['similarity'] = 1 - float(distance)  # cosine similarity
            results.append(chunk)
        
        return results
    
    def retrieve_texts(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve top-k chunks as text strings only.
        Convenient for passing to LLM.
        
        Args:
            query: The query string
            k: Number of chunks to retrieve
            
        Returns:
            List of chunk texts
        """
        results = self.retrieve(query, k)
        return [chunk['text'] for chunk in results]
    
    def retrieve_with_metadata(self, query: str, k: int = 5) -> List[Tuple[str, str, str, float]]:
        """
        Retrieve top-k chunks with metadata.
        Returns (id, title, text, similarity) tuples.
        Useful for evaluation.
        
        Args:
            query: The query string
            k: Number of chunks to retrieve
            
        Returns:
            List of (chunk_id, title, text, similarity) tuples
        """
        results = self.retrieve(query, k)
        return [
            (chunk['id'], chunk['title'], chunk['text'], chunk['similarity'])
            for chunk in results
        ]


# Example usage and testing
if __name__ == "__main__":
    retriever = TopKRetriever()
    
    print("Building index from database...")
    retriever.build_index()
    
    # Test query
    query = "Where was the composer of Moonlight Sonata born?"
    print(f"\nQuery: {query}")
    print("\nTop-5 Retrieved Chunks:")
    print("-" * 80)
    
    results = retriever.retrieve(query, k=5)
    for i, chunk in enumerate(results, 1):
        print(f"\n{i}. [Similarity: {chunk['similarity']:.4f}]")
        print(f"   ID: {chunk['id']}")
        print(f"   Title: {chunk['title']}")
        print(f"   Text: {chunk['text'][:200]}...")
    
    print("\n" + "="*80)
    print("Top-K Baseline Complete!")
    print("\nThis uses:")
    print("  ✓ initialise_hnsw() and add_items() from retrieve.py")
    print("  ✓ embedding_model from encode.py")
    print("  ✓ Database schema from db_functions.py")