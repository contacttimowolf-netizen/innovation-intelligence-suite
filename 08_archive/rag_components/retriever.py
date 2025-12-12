import chromadb
from chromadb.config import Settings
import os
from typing import List, Tuple, Optional

class Retriever:
    def __init__(self, vector_index_path: str = "../../04_models/vector_index"):
        """
        Initialize the retriever with ChromaDB vector index.
        
        Args:
            vector_index_path: Path to the ChromaDB persistent storage
        """
        self.vector_index_path = vector_index_path
        self.client = None
        self.collection = None
        self.chunks = []
        
        self._initialize_chromadb()
        self._load_chunks()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and load the collection."""
        try:
            # Check if the vector index exists
            if not os.path.exists(self.vector_index_path):
                raise FileNotFoundError(f"Vector index not found at {self.vector_index_path}")
            
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.vector_index_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection (should already exist from notebook 2)
            self.collection = self.client.get_collection(name="documents")
            print(f"✅ ChromaDB collection loaded: {self.collection.name}")
            
        except Exception as e:
            print(f"❌ ChromaDB initialization failed: {e}")
            self.collection = None
    
    def _load_chunks(self):
        """Load chunks metadata if available."""
        chunks_metadata_path = os.path.join(self.vector_index_path, "chunks_metadata.pkl")
        try:
            import pickle
            with open(chunks_metadata_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"✅ Loaded {len(self.chunks)} chunks from metadata")
        except Exception as e:
            print(f"⚠️ Could not load chunks metadata: {e}")
            self.chunks = []
    
    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        if self.collection is None:
            print("❌ ChromaDB collection not initialized")
            return []
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "distances", "metadatas"]
            )
            
            # Format results
            retrieved_chunks = []
            if results['documents'] and results['distances']:
                for doc, distance in zip(results['documents'][0], results['distances'][0]):
                    # ChromaDB returns distances (smaller is better), convert to similarity score
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    
                    if similarity >= threshold:
                        retrieved_chunks.append((doc, similarity))
            
            print(f"🔍 Retrieved {len(retrieved_chunks)} chunks for query: '{query}'")
            return retrieved_chunks
            
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            return []
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks in the index."""
        if self.collection is None:
            return 0
        return self.collection.count()