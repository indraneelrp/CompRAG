
"""
# Grab queries
# Make triples from queries
# Do HRR for triples of the query
# Search k most similar triples for each of the query's triples
# Grab the associated chunks (do not duplicate grab)
# Pass it to the Ollama LM and let it produce its output
# save output for later eval of Hotpot QA

CompRAG Main Loop - Implementation Skeleton
===========================================

Phase 1: Database Integration
Phase 2: Search Index Building  
Phase 3: Query Processing Pipeline
Phase 4: Retrieval Logic
Phase 5: Ollama Integration
Phase 6: Main Execution Flow
"""

import os
import sqlite3
import numpy as np
import hnswlib
import requests
import json
from typing import List, Tuple, Dict, Set, Optional
from dotenv import load_dotenv
import spacy

# Import your CompRAG modules
from compRAG.db_functions import init_db, process_dataset, get_all_hrr_vectors
from compRAG.make_triplets import main_generate_triplets  
from compRAG.encode import chunk_triplets2embeddings, chunk_embeddings2hrr
from compRAG.retrieve import initialise_hnsw, add_items

load_dotenv()


class CompRAGSystem:
    """Main CompRAG system orchestrator"""
    
    def __init__(self, db_path: str = None, ollama_url: str = "http://localhost:11434/api/generate"):
        """
        TODO Phase 1: Initialize system
        - Set db_path and ollama_url
        - Initialize database connection
        - Load SpaCy model
        - Set up instance variables (index, mappings, etc.)
        """
        self.db_path = db_path or os.getenv("HOTPOT_DB")
        self.ollama_url = ollama_url
        self.conn = sqlite3.connect(self.db_path)
        self.index = None
        self.id_to_chunk = {}

        self.dim = 384  
        self.max_elements = 100000

        if not self.db_path:
            raise ValueError("Set HOTPOT_DB in your .env file")
        
        self.conn = init_db(self.db_path)  

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            raise RuntimeError("Failed to load SpaCy model. Ensure 'en_core_web_sm' is installed.") from e
        
        self.index = None  # HNSW search index
        self.vector_to_chunk = {}  # Maps vector IDs to chunk IDs
        self.is_index_built = False

    def setup_database(self, dataset_name: str = "BeIR/hotpotqa", limit: int = None):
        """
        TODO Phase 1: Process dataset and populate database
        - Call your process_dataset function from db_functions
        - Start with small limit for testing (e.g. 50 documents)
        """
        print(f"📊 Setting up database with dataset: {dataset_name}")
        if limit:
            print(f"   Processing limit: {limit} documents")
        else:
            print("   Processing: ALL documents (this may take a while)")
        
        try:
            process_dataset(
                dataset_name=dataset_name,
                subset="corpus", 
                limit=limit,
                batch_size=100,  
                num_workers=2    
            )
            print("✅ Database setup completed successfully")
            
        except Exception as e:
            print(f"❌ Error during database setup: {str(e)}")
            raise


    def build_search_index(self, dim: int = 384, max_elements: int = 100000):
        """
        TODO Phase 2: Build HNSW index from stored HRR vectors
        - Retrieve all HRR vectors from database
        - Initialize HNSW index
        - Add vectors to index
        - Create mapping from vector IDs to chunk IDs
        """
        pass
    
    def process_query(self, query: str) -> List[Tuple[str, str, str]]:
        """
        TODO Phase 3: Extract triplets from user query
        - Use main_generate_triplets with SpaCy model
        - Handle case where no triplets found
        - Return list of (subject, relation, object) tuples
        """
        pass
    
    def query_to_hrr_vectors(self, query_triplets: List[Tuple[str, str, str]]) -> List[np.ndarray]:
        """
        TODO Phase 3: Convert query triplets to HRR vectors
        - Use chunk_triplets2embeddings function
        - Use chunk_embeddings2hrr function  
        - Convert tensors to numpy arrays for HNSW search
        """
        pass
    
    def retrieve_similar_chunks(self, query_hrr_vectors: List[np.ndarray], k: int = 5) -> Set[str]:
        """
        TODO Phase 4: Find similar chunks using HNSW search
        - For each query HRR vector, search HNSW index
        - Map vector IDs back to chunk IDs
        - Return set of unique chunk IDs
        """
        pass
    
    def get_chunk_contexts(self, chunk_ids: Set[str]) -> List[Dict]:
        """
        TODO Phase 4: Retrieve text content for chunks
        - Query database for chunk texts by ID
        - Format as list of dicts with id, title, text
        - Handle missing chunks
        """
        pass
    
    def generate_response(self, query: str, contexts: List[Dict], model: str = "gemma3") -> str:
        """
        TODO Phase 5: Generate response using Ollama
        - Format contexts into prompt template
        - Send POST request to Ollama API
        - Handle streaming response
        - Return final response text
        """
        pass
    
    def answer_query(self, query: str, k: int = 5) -> Dict:
        """
        TODO Phase 6: Main pipeline orchestration
        - Call process_query() to get triplets
        - Call query_to_hrr_vectors() to get HRR vectors
        - Call retrieve_similar_chunks() to find relevant chunks
        - Call get_chunk_contexts() to get text
        - Call generate_response() to get final answer
        - Return dict with query, results, and response
        """
        pass
    
    def interactive_mode(self):
        """
        TODO Phase 6: Interactive question-answer loop
        - Continuous input loop
        - Call answer_query() for each question
        - Handle quit commands
        """
        pass
    
    def close(self):
        """Clean up database connection"""
        pass


def main():
    """
    TODO Phase 6: Main execution
    - Initialize CompRAGSystem
    - Run setup_database() (test with limit=50 first)
    - Run build_search_index()
    - Test with sample queries
    - Run interactive_mode()
    """
    pass


if __name__ == "__main__":
    main()