"""
FAISS Retriever - Reliable vector search without database corruption issues.
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from fastembed import TextEmbedding


class FAISSRetriever:
    """
    Reliable FAISS-based retriever for document search.
    Uses FAISS for similarity search and fastembed for embeddings.
    """
    
    def __init__(self, vector_index_path: str = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize FAISS retriever.
        
        Args:
            vector_index_path: Path to FAISS index directory. If None, uses default.
            model_name: Embedding model name for query encoding.
        """
        if vector_index_path is None:
            # Calculate default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
            vector_index_path = os.path.join(project_root, "04_models", "vector_index")
        
        self.vector_index_path = vector_index_path
        self.model_name = model_name
        self.index = None
        self.texts = []
        self.metadatas = []
        self.embedding_model = None
        
        self._initialize()
    
    def _initialize(self):
        """Load FAISS index and associated data from disk."""
        try:
            print(f"🔍 Loading FAISS index from {self.vector_index_path}")
            
            # Validate path
            if not os.path.exists(self.vector_index_path):
                raise FileNotFoundError(f"Index directory not found: {self.vector_index_path}")
            
            # 1. Load FAISS index
            index_path = os.path.join(self.vector_index_path, "faiss_index.bin")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index not found: {index_path}")
            
            self.index = faiss.read_index(index_path)
            print(f"✓ FAISS index loaded: {self.index.ntotal} vectors")
            
            # 2. Load texts
            texts_path = os.path.join(self.vector_index_path, "texts.pkl")
            with open(texts_path, "rb") as f:
                self.texts = pickle.load(f)
            print(f"✓ Texts loaded: {len(self.texts)} chunks")
            
            # 3. Load metadata
            metadata_path = os.path.join(self.vector_index_path, "metadata.pkl")
            with open(metadata_path, "rb") as f:
                self.metadatas = pickle.load(f)
            print(f"✓ Metadata loaded: {len(self.metadatas)} entries")
            
            # 4. Initialize embedding model for queries
            self.embedding_model = TextEmbedding(model_name=self.model_name)
            print(f"✓ Embedding model loaded: {self.model_name}")
            
            # Verify consistency
            if len(self.texts) != self.index.ntotal:
                print(f"⚠️ Warning: Mismatch - Texts ({len(self.texts)}) != Vectors ({self.index.ntotal})")
            
        except Exception as e:
            print(f"❌ FAISS initialization failed: {e}")
            self.index = None
            raise
    
    def retrieve(self, query: str, top_k: int = 5, similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of dictionaries with keys: 'text', 'similarity', 'metadata', 'index'
        """
        if self.index is None or self.embedding_model is None:
            raise RuntimeError("Retriever not properly initialized")
        
        try:
            # Create query embedding
            query_embedding = np.array(
                list(self.embedding_model.embed([query]))[0].tolist()
            ).astype('float32').reshape(1, -1)
            
            # Search FAISS index
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.texts):  # Valid result
                    # Convert L2 distance to similarity score
                    similarity = 1.0 / (1.0 + distance)
                    
                    if similarity >= similarity_threshold:
                        results.append({
                            'text': self.texts[idx],
                            'similarity': float(similarity),
                            'distance': float(distance),
                            'metadata': self.metadatas[idx] if idx < len(self.metadatas) else {},
                            'index': int(idx)
                        })
            
            print(f"🔍 Retrieved {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            print(f"❌ Retrieval failed for query '{query}': {e}")
            return []
    
    def retrieve_with_sources(self, query: str, k: int = 5, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Retrieve results with source information (compatibility method).
        
        Args:
            query: Search query
            k: Number of results (alias for top_k)
            threshold: Similarity threshold
            
        Returns:
            List of results with source information
        """
        results = self.retrieve(query, top_k=k, similarity_threshold=threshold)
        
        # Format for compatibility with existing code
        formatted_results = []
        for result in results:
            formatted_results.append({
                'text': result['text'],
                'similarity_score': result['similarity'],
                'source_file': result['metadata'].get('source', 'unknown'),
                'doc_type': result['metadata'].get('doc_type', 'document'),
                'metadata': result['metadata']
            })
        
        return formatted_results
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks in the index."""
        return len(self.texts)
    
    def get_doc_type_counts(self) -> Dict[str, int]:
        """Count documents by type."""
        counts = {}
        for metadata in self.metadatas:
            doc_type = metadata.get('doc_type', 'unknown')
            counts[doc_type] = counts.get(doc_type, 0) + 1
        return counts
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the loaded collection."""
        return {
            'total_chunks': len(self.texts),
            'faiss_vectors': self.index.ntotal if self.index else 0,
            'embedding_dim': self.index.d if self.index else 0,
            'doc_types': self.get_doc_type_counts()
        }