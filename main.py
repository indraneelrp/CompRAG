
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
import time 

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
        print("🔍 Building search index...")
        if not self.conn:
            raise RuntimeError("Database connection not available")

        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT hrr_id, chunk_id, hrr_vector 
            FROM hrr_vectors 
            WHERE hrr_vector IS NOT NULL
        """)
        hrr_data = cursor.fetchall()
        
        if not hrr_data:
            raise RuntimeError("No HRR vectors found in database. Run setup_database first.")
        
        print(f"Found {len(hrr_data)} HRR vectors")
            
        vectors = []
        vector_ids = []
        
        for hrr_id, chunk_id, hrr_blob in hrr_data:
            hrr_vector = np.frombuffer(hrr_blob, dtype=np.float32)
            vectors.append(hrr_vector)
            vector_ids.append(hrr_id)
            
            self.vector_to_chunk[hrr_id] = chunk_id
        
        print("Initializing HNSW index...")
        self.index = initialise_hnsw(dim, max_elements)
        
        print("Adding vectors to index...")
        add_items(self.index, vectors, vector_ids)
        
        self.is_index_built = True
        
        print(f"✅ Search index built successfully with {len(vectors)} vectors")

    
    def process_query(self, query: str) -> List[Tuple[str, str, str]]:
        """
        TODO Phase 3: Extract triplets from user query
        - Use main_generate_triplets with SpaCy model
        - Handle case where no triplets found
        - Return list of (subject, relation, object) tuples
        """
        print(f"❓ Processing query: '{query}'")
    
        if not query or not query.strip():
            return []
        
        try:
            triplets = main_generate_triplets(self.nlp, query.strip())
            
            if triplets:
                print(f"✅ Extracted {len(triplets)} triplets")
                for i, (s, r, o) in enumerate(triplets):
                    print(f"      {i+1}. ({s}, {r}, {o})")
                return triplets
            else: 
                print("⚠️  Could not create any triplets from query")
                return []
            
        except Exception as e:
            print(f"   ❌ Error processing query: {str(e)}")
            return []
    
    def query_to_hrr_vectors(self, query_triplets: List[Tuple[str, str, str]]) -> List[np.ndarray]:
        """
        TODO Phase 3: Convert query triplets to HRR vectors
        - Use chunk_triplets2embeddings function
        - Use chunk_embeddings2hrr function  
        - Convert tensors to numpy arrays for HNSW search
        """
        print("🔄 Converting query triplets to HRR vectors...")
    
        if not query_triplets:
            print("   ⚠️  No triplets to convert")
            return []
        
        try:
            print(f"   📊 Processing {len(query_triplets)} triplets...")
            
            valid_triplets = []
            for s, r, o in query_triplets:
                if s and r and o: 
                    valid_triplets.append((str(s), str(r), str(o)))
                else:
                    print(f"   ⚠️  Skipping invalid triplet: ({s}, {r}, {o})")
            
            if not valid_triplets:
                print("   ❌ No valid triplets found")
                return []
            
            # Step 2: Convert to embeddings
            embeddings = chunk_triplets2embeddings(valid_triplets)
            
            if not embeddings:
                print("   ❌ Failed to generate embeddings")
                return []
            
            # Step 3: Convert to HRR vectors
            hrr_vectors_tensors = chunk_embeddings2hrr(embeddings)
            
            # Step 4: Convert to numpy and validate
            hrr_vectors = []
            for i, hrr_tensor in enumerate(hrr_vectors_tensors):
                try:
                    # Convert to numpy
                    if hasattr(hrr_tensor, 'cpu'):
                        hrr_numpy = hrr_tensor.cpu().detach().numpy()
                    elif hasattr(hrr_tensor, 'numpy'):
                        hrr_numpy = hrr_tensor.numpy()
                    else:
                        hrr_numpy = np.array(hrr_tensor)
                    
                    # Validate dimensions
                    hrr_numpy = hrr_numpy.astype(np.float32)
                    if hrr_numpy.shape[-1] != 384:  # Expected dimension
                        print(f"   ⚠️  Vector {i} has wrong dimension: {hrr_numpy.shape}")
                        continue
                    
                    # Flatten if needed
                    if hrr_numpy.ndim > 1:
                        hrr_numpy = hrr_numpy.flatten()
                    
                    hrr_vectors.append(hrr_numpy)
                    
                except Exception as e:
                    print(f"   ⚠️  Error processing vector {i}: {e}")
                    continue
            
            print(f"   ✅ Generated {len(hrr_vectors)} valid HRR vectors")
            return hrr_vectors
            
        except Exception as e:
            print(f"   ❌ Error in HRR conversion: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def retrieve_similar_chunks(self, query_hrr_vectors: List[np.ndarray], k: int = 5) -> Set[str]:
        """
        TODO Phase 4: Find similar chunks using HNSW search
        - For each query HRR vector, search HNSW index
        - Map vector IDs back to chunk IDs
        - Return set of unique chunk IDs
        """
        print(f"🎯 Retrieving top-{k} similar chunks...")
        
        # Step 1: Validate inputs
        if not self.index or not self.is_index_built:
            print("   ❌ Search index not built. Run build_search_index() first.")
            return set()
        
        if not query_hrr_vectors:
            print("   ⚠️  No query vectors provided")
            return set()
        
        retrieved_chunk_ids = set()
        
        # Step 2: Search for each query vector
        for i, query_vector in enumerate(query_hrr_vectors):
            try:
                # Reshape for HNSW search
                query_vec_reshaped = query_vector.reshape(1, -1)
                
                # Search HNSW index
                labels, distances = self.index.knn_query(query_vec_reshaped, k=k)
                
                print(f"   🔍 Query vector {i+1}: found {len(labels[0])} similar vectors")
                
                # Step 3: Map vector IDs to chunk IDs
                for vector_id in labels[0]:
                    if vector_id in self.vector_to_chunk:
                        chunk_id = self.vector_to_chunk[vector_id]
                        retrieved_chunk_ids.add(chunk_id)
                    else:
                        print(f"   ⚠️  Vector ID {vector_id} not found in mapping")
                
            except Exception as e:
                print(f"   ⚠️  Error searching with vector {i}: {e}")
                continue
        
        print(f"   ✅ Retrieved {len(retrieved_chunk_ids)} unique chunks")
        return retrieved_chunk_ids
    
    def get_chunk_contexts(self, chunk_ids: Set[str]) -> List[Dict]:
        """
        TODO Phase 4: Retrieve text content for chunks
        - Query database for chunk texts by ID
        - Format as list of dicts with id, title, text
        - Handle missing chunks
        """
        print(f"📖 Retrieving context for {len(chunk_ids)} chunks...")
        
        if not chunk_ids:
            return []
        
        if not self.conn:
            print("   ❌ Database connection not available")
            return []
        
        contexts = []
        
        try:
            cursor = self.conn.cursor()
            chunk_id_list = list(chunk_ids)
            placeholders = ','.join('?' * len(chunk_id_list))
            
            query = f"""
                SELECT id, title, text 
                FROM chunks 
                WHERE id IN ({placeholders})
                ORDER BY id
            """
            
            cursor.execute(query, chunk_id_list)
            results = cursor.fetchall()
            
            print(f"   📊 Found {len(results)} chunks in database")
            
            for chunk_id, title, text in results:
                # Clean and format text
                clean_text = text.strip() if text else ""
                clean_title = title.strip() if title else f"Document {chunk_id}"
                
                # Skip empty chunks
                if not clean_text:
                    print(f"   ⚠️  Skipping empty chunk: {chunk_id}")
                    continue
                
                # Truncate very long texts (optional)
                max_text_length = 2000  # Adjust based on your LLM context limits
                if len(clean_text) > max_text_length:
                    clean_text = clean_text[:max_text_length] + "..."
                    print(f"   ✂️  Truncated chunk {chunk_id} to {max_text_length} chars")
                
                context = {
                    'id': chunk_id,
                    'title': clean_title,
                    'text': clean_text,
                    'length': len(clean_text)
                }
                contexts.append(context)
            
            # Report missing chunks
            found_ids = {result[0] for result in results}
            missing_ids = chunk_ids - found_ids
            if missing_ids:
                print(f"   ⚠️  {len(missing_ids)} chunks not found in database")
            
            # Sort by text length (optional - put longer contexts first)
            contexts.sort(key=lambda x: x['length'], reverse=True)
            
            print(f"   ✅ Retrieved {len(contexts)} valid contexts")
            return contexts
            
        except Exception as e:
            print(f"   ❌ Error retrieving contexts: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def generate_response(self, query: str, contexts: List[Dict], model: str = "gemma3") -> str:
        """
        TODO Phase 5: Generate response using Ollama
        - Format contexts into prompt template
        - Send POST request to Ollama API
        - Handle streaming response
        - Return final response text
        """
        if not contexts:
            return "❌ No relevant context found to answer your question."
        
        if not query.strip():
            return "❌ No query provided."
        
        try:
            # Format context from retrieved chunks
            context_text = ""
            for i, ctx in enumerate(contexts, 1):
                title = ctx.get('title', f"Document {i}")
                text = ctx.get('text', '')
                context_text += f"\n--- Document {i}: {title} ---\n{text}\n"
            
            # Create prompt template
            prompt = f"""
                You are an AI assistant that answers questions based on provided documents. Use only the information from the supplied documents to answer the query. If the information is not available in the documents, state that clearly.

                Documents:
                {context_text}

                Question: {query}

                Please provide a comprehensive answer based on the information in the documents above. Be specific and cite which documents you're referencing when possible.
            """

            # Prepare request data
            data = {
                "model": model,
                "prompt": prompt,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            headers = {'Content-Type': 'application/json'}
            
            print(f"   📤 Sending request to Ollama (model: {model})")
            print(f"   📊 Context: {len(contexts)} documents")
            
            # Send request to Ollama API
            response = requests.post(
                self.ollama_url, 
                data=json.dumps(data), 
                headers=headers, 
                stream=True,
                timeout=120
            )
            
            if response.status_code != 200:
                error_msg = f"Ollama API error (status {response.status_code})"
                print(f"   ❌ {error_msg}")
                return f"❌ {error_msg}"
            
            # Handle streaming response
            full_response = []
            print("   🔄 Receiving response...")
            
            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = json.loads(line.decode('utf-8'))
                        if 'response' in decoded_line:
                            full_response.append(decoded_line['response'])
                        
                        if decoded_line.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            final_response = ''.join(full_response)
            
            if not final_response.strip():
                return "❌ Empty response received from Ollama"
            
            print(f"   ✅ Response generated ({len(final_response)} characters)")
            return final_response.strip()
            
        except requests.exceptions.Timeout:
            return "⏱️  Request to Ollama timed out. Check if Ollama is running."
            
        except requests.exceptions.ConnectionError:
            return "🔌 Could not connect to Ollama. Make sure Ollama is running (try: ollama serve)"
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def answer_query(self, query: str, k: int = 5) -> Dict:
        """Main pipeline orchestration - processes query and returns complete response"""
        print(f"\n{'='*60}")
        print(f"🎯 PROCESSING QUERY: {query}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Step 1: Extract triplets from query
            print("\n📝 Step 1: Extracting triplets from query...")
            triplets = self.process_query(query)
            
            if not triplets:
                return {
                    'query': query,
                    'triplets': [],
                    'retrieved_chunks': 0,
                    'response': "Could not extract meaningful triplets from your query. Try rephrasing your question.",
                    'success': False,
                    'processing_time': time.time() - start_time
                }
            
            # Step 2: Convert triplets to HRR vectors
            print("\n🧮 Step 2: Converting triplets to HRR vectors...")
            hrr_vectors = self.query_to_hrr_vectors(triplets)
            
            if not hrr_vectors:
                return {
                    'query': query,
                    'triplets': triplets,
                    'retrieved_chunks': 0,
                    'response': "Failed to convert query triplets to searchable vectors.",
                    'success': False,
                    'processing_time': time.time() - start_time
                }
            
            # Step 3: Search for similar chunks
            print("\n🔍 Step 3: Searching for similar chunks...")
            chunk_ids = self.retrieve_similar_chunks(hrr_vectors, k=k)
            
            if not chunk_ids:
                return {
                    'query': query,
                    'triplets': triplets,
                    'retrieved_chunks': 0,
                    'response': "No relevant documents found for your query. Try different keywords or check if the database contains relevant information.",
                    'success': False,
                    'processing_time': time.time() - start_time
                }
            
            # Step 4: Get context text
            print("\n📖 Step 4: Retrieving context text...")
            contexts = self.get_chunk_contexts(chunk_ids)
            
            if not contexts:
                return {
                    'query': query,
                    'triplets': triplets,
                    'retrieved_chunks': 0,
                    'response': "Found relevant document IDs but could not retrieve their content from database.",
                    'success': False,
                    'processing_time': time.time() - start_time
                }
            
            # Step 5: Generate response
            print("\n🤖 Step 5: Generating response...")
            response = self.generate_response(query, contexts)
            
            # Step 6: Compile results
            processing_time = time.time() - start_time
            
            result = {
                'query': query,
                'triplets': triplets,
                'retrieved_chunks': len(contexts),
                'response': response,
                'success': True,
                'processing_time': processing_time,
                'chunk_ids': list(chunk_ids),
                'context_titles': [ctx.get('title', 'Unknown') for ctx in contexts]
            }
            
            print(f"\n✅ QUERY PROCESSING COMPLETE")
            print(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"   🔗 Triplets extracted: {len(triplets)}")
            print(f"   📄 Documents retrieved: {len(contexts)}")
            print(f"   📝 Response length: {len(response)} characters")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"\n❌ PIPELINE ERROR: {error_msg}")
            
            return {
                'query': query,
                'error': error_msg,
                'success': False,
                'processing_time': time.time() - start_time
            }
    
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
