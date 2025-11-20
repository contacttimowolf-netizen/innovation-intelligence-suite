import json
from typing import List, Dict

class SimpleQueryInterface:
    def __init__(self, retriever):
        self.retriever = retriever
    
    def query(self, question: str, top_k: int = 5, doc_types: List[str] = None) -> Dict:
        """Simple query interface that returns structured results"""
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve_with_sources(question, k=top_k, doc_types=doc_types)
        
        # Prepare response
        response = {
            "question": question,
            "top_k": top_k,
            "doc_types_filter": doc_types,
            "retrieved_documents": retrieved_docs,
            "summary": {
                "total_documents_found": len(retrieved_docs),
                "unique_sources": list(set(doc['source_file'] for doc in retrieved_docs)),
                "document_types": list(set(doc['doc_type'] for doc in retrieved_docs)),
                "doc_type_counts": self.retriever.get_doc_type_counts()
            }
        }
        
        return response
    
    def query_by_doc_type(self, question: str, doc_type: str, top_k: int = 5) -> Dict:
        """Query specific document type only"""
        return self.query(question, top_k, doc_types=[doc_type])
    
    def save_query_results(self, query: str, results: Dict, filename: str = None):
        """Save query results to JSON file"""
        if filename is None:
            filename = f"query_results_{len(query)}_{hash(query) % 10000}.json"
        
        save_path = f"../01_data/rag_automotive_tech/processed/{filename}"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Query results saved to: {save_path}")