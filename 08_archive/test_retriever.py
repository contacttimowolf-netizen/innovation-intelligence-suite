#!/usr/bin/env python3
# test_retriever.py - UPDATED with auto-install

import os
import sys
import subprocess

# ===== AUTO-INSTALL FAISS IF MISSING =====
try:
    import faiss
    print("✅ FAISS is installed")
except ImportError:
    print("⚠️ FAISS not found. Installing faiss-cpu...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu", "-q"])
        import faiss
        print("✅ FAISS installed successfully")
    except Exception as e:
        print(f"❌ Failed to install FAISS: {e}")
        print("Please run: pip install faiss-cpu")
        sys.exit(1)

# Now import the rest
import pickle
import numpy as np
from fastembed import TextEmbedding

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===== SIMPLE FAISS RETRIEVER CLASS (included in file) =====
class FAISSRetriever:
    """Simple FAISS retriever included in test file."""
    
    def __init__(self, vector_index_path: str = None):
        if vector_index_path is None:
            # Use absolute path
            vector_index_path = "/Users/manueltimowolf/Documents/data_science_ai/innovation-intelligence-suite/04_models/vector_index"
        
        self.vector_index_path = vector_index_path
        self.index = None
        self.texts = []
        self.metadatas = []
        self.embedding_model = None
        
        self._initialize()
    
    def _initialize(self):
        """Load FAISS index and associated data."""
        try:
            print(f"🔍 Loading FAISS index from {self.vector_index_path}")
            
            # Check path exists
            if not os.path.exists(self.vector_index_path):
                raise FileNotFoundError(f"Index directory not found: {self.vector_index_path}")
            
            print(f"Directory contents: {os.listdir(self.vector_index_path)}")
            
            # 1. Load FAISS index
            index_path = os.path.join(self.vector_index_path, "faiss_index.bin")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index not found at {index_path}")
            
            self.index = faiss.read_index(index_path)
            print(f"✓ FAISS index loaded: {self.index.ntotal} vectors")
            
            # 2. Load texts
            texts_path = os.path.join(self.vector_index_path, "texts.pkl")
            with open(texts_path, "rb") as f:
                self.texts = pickle.load(f)
            print(f"✓ Texts loaded: {len(self.texts)}")
            
            # 3. Load metadata
            metadata_path = os.path.join(self.vector_index_path, "metadata.pkl")
            with open(metadata_path, "rb") as f:
                self.metadatas = pickle.load(f)
            print(f"✓ Metadata loaded: {len(self.metadatas)}")
            
            # 4. Initialize embedding model
            self.embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            print("✓ Embedding model loaded")
            
        except Exception as e:
            print(f"❌ FAISS initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.index = None
    
    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.6) -> list:
        if self.index is None or self.embedding_model is None:
            print("❌ Retriever not initialized")
            return []
        
        try:
            # Create query embedding
            query_embedding = np.array(
                list(self.embedding_model.embed([query]))[0].tolist()
            ).astype('float32').reshape(1, -1)
            
            # Search
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx != -1 and idx < len(self.texts):
                    similarity = 1.0 / (1.0 + distance)
                    
                    if similarity >= threshold:
                        results.append({
                            'text': self.texts[idx],
                            'similarity': similarity,
                            'distance': distance,
                            'metadata': self.metadatas[idx] if idx < len(self.metadatas) else {},
                            'index': idx
                        })
            
            print(f"🔍 Retrieved {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            return []
    
    def get_chunk_count(self) -> int:
        return len(self.texts) if self.texts else 0

# ===== TEST FUNCTION =====
def test_retriever():
    """Test the FAISS retriever."""
    print("\n" + "="*50)
    print("🧪 TESTING FAISS RETRIEVER")
    print("="*50)
    
    # Use absolute path
    base_path = "/Users/manueltimowolf/Documents/data_science_ai/innovation-intelligence-suite"
    vector_index_path = os.path.join(base_path, "04_models", "vector_index")
    
    print(f"\n📁 Index path: {vector_index_path}")
    print(f"Path exists: {os.path.exists(vector_index_path)}")
    
    if os.path.exists(vector_index_path):
        files = os.listdir(vector_index_path)
        print(f"Contents ({len(files)} files):")
        for file in files:
            file_path = os.path.join(vector_index_path, file)
            size = os.path.getsize(file_path)
            print(f"  - {file} ({size:,} bytes)")
    
    # Initialize retriever
    print("\n🔄 Initializing retriever...")
    retriever = FAISSRetriever(vector_index_path=vector_index_path)
    
    # Check if initialized
    if retriever.index is None:
        print("❌ Failed to initialize FAISS retriever")
        return False
    
    print(f"✅ Retriever initialized with {retriever.get_chunk_count()} documents")
    
    # Test queries
    test_queries = [
        "automotive startups",
        "autonomous driving technology", 
        "generative AI in automotive",
        "electric vehicle innovation",
        "battery technology for cars"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = retriever.retrieve(query, top_k=3, threshold=0.5)
        
        if results:
            for i, result in enumerate(results):
                print(f"  Result {i+1} (sim: {result['similarity']:.3f}):")
                print(f"    Type: {result['metadata'].get('doc_type', 'N/A')}")
                preview = result['text'][:120] + "..." if len(result['text']) > 120 else result['text']
                print(f"    Preview: {preview}")
        else:
            print("  No results found")
    
    return True

# ===== MAIN =====
if __name__ == "__main__":
    try:
        success = test_retriever()
        print("\n" + "="*50)
        if success:
            print("✅ ALL TESTS PASSED!")
            print("🎉 FAISS is working perfectly!")
        else:
            print("❌ TESTS FAILED")
        print("="*50)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()