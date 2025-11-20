import os
import joblib
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class DocumentAwareRetriever:
    def __init__(self, vector_store_path):
        self.vector_store_path = vector_store_path
        self.retrieval_method = "tfidf"
        
        try:
            # Load TF-IDF data
            tfidf_data = joblib.load(os.path.join(vector_store_path, 'tfidf_embeddings.pkl'))
            self.tfidf_matrix = tfidf_data['matrix']
            self.vectorizer = tfidf_data['vectorizer']
            self.tfidf_chunks = tfidf_data['chunks']
            print("✓ TF-IDF retriever loaded successfully")
            
        except Exception as e:
            print(f"❌ TF-IDF loading failed: {e}")
            self.retrieval_method = "none"
        
        # Load chunks metadata
        try:
            with open(os.path.join(vector_store_path, "chunks_metadata.pkl"), "rb") as f:
                self.chunks_metadata = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load chunks metadata: {e}")
            self.chunks_metadata = []
    
    def retrieve_with_sources(self, query, k=5, doc_types=None):
        """Retrieve documents with full source information"""
        if self.retrieval_method != "tfidf":
            return []
        
        try:
            # Transform query to TF-IDF vector
            query_vec = self.vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top k results
            top_indices = similarities.argsort()[-k:][::-1]
            
            results = []
            for idx in top_indices:
                if idx < len(self.tfidf_chunks):
                    chunk_data = self.tfidf_chunks[idx]
                    
                    # Apply document type filter if specified
                    if doc_types is None or chunk_data['metadata'].get('doc_type') in doc_types:
                        source_info = {
                            'content': chunk_data['page_content'],
                            'source_file': chunk_data['metadata'].get('source', 'Unknown'),
                            'doc_type': chunk_data['metadata'].get('doc_type', 'Unknown'),
                            'similarity_score': float(similarities[idx]),
                            'full_metadata': chunk_data['metadata']
                        }
                        results.append(source_info)
            
            return results
            
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")
            return []
    
    def get_document_chunks(self, filename):
        """Get all chunks from a specific document"""
        return [chunk for chunk in self.chunks_metadata 
                if chunk['metadata']['source'] == filename]
    
    def get_doc_type_counts(self):
        """Get counts of chunks by document type"""
        doc_types = {}
        for chunk in self.chunks_metadata:
            doc_type = chunk['metadata'].get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        return doc_types
