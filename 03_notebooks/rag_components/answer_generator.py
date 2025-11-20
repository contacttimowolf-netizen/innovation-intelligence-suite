import json
from typing import List, Dict

class RAGAnswerGenerator:
    def __init__(self, query_interface):
        self.query_interface = query_interface
        print("✓ Template-based RAG answer generator initialized")
    
    def generate_answer(self, question: str, top_k: int = 5, doc_types: List[str] = None) -> Dict:
        """Generate comprehensive answers using template-based approach"""
        
        # Retrieve relevant documents
        retrieval_results = self.query_interface.query(question, top_k=top_k, doc_types=doc_types)
        retrieved_docs = retrieval_results["retrieved_documents"]
        
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "❌ No relevant information found in the documents for this query.",
                "sources": [],
                "context_used": {"total_chunks_used": 0}
            }
        
        # Generate comprehensive template-based answer
        answer = self._generate_template_answer(question, retrieved_docs)
        
        # Prepare final response with source attribution
        response = {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "source_file": doc["source_file"],
                    "doc_type": doc["doc_type"],
                    "relevance_score": doc["similarity_score"],
                    "content_snippet": doc["content"][:150] + "..." if len(doc["content"]) > 150 else doc["content"]
                }
                for doc in retrieved_docs
            ],
            "context_used": {
                "total_chunks_used": len(retrieved_docs),
                "unique_documents": list(set(doc["source_file"] for doc in retrieved_docs)),
                "document_types_used": list(set(doc["doc_type"] for doc in retrieved_docs))
            }
        }
        
        return response
    
    def _generate_template_answer(self, question: str, retrieved_docs: List[Dict]) -> str:
        """Generate a comprehensive template-based answer"""
        doc_types = list(set(doc['doc_type'] for doc in retrieved_docs))
        
        answer_parts = []
        answer_parts.append(f"Based on automotive {', '.join(doc_types)} data, here's information about '{question}':")
        
        # Group by document type and summarize key points
        for doc_type in doc_types:
            type_docs = [doc for doc in retrieved_docs if doc['doc_type'] == doc_type]
            answer_parts.append(f"\nFrom {doc_type.replace('_', ' ')}:")
            
            # Take key snippets from top documents of this type
            for i, doc in enumerate(type_docs[:2]):
                snippet = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
                answer_parts.append(f"  • {snippet}")
        
        answer_parts.append(f"\nTotal relevant chunks found: {len(retrieved_docs)}")
        
        return "\n".join(answer_parts)
    
    def _get_doc_type_emoji(self, doc_type: str) -> str:
        """Get appropriate emoji for document type"""
        emoji_map = {
            'startups_data': '🚀',
            'research_paper': '📄', 
            'tech_report': '🔍',
            'unknown': '📁'
        }
        return emoji_map.get(doc_type, '📁')