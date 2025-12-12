# test_thresholds.py
from faiss_retriever import FAISSRetriever

retriever = FAISSRetriever()

print("🧪 Testing 'automotive startups' with different thresholds:")

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    print(f"\nThreshold: {threshold}")
    results = retriever.retrieve("automotive startups", top_k=3, similarity_threshold=threshold)
    
    print(f"  Found {len(results)} results")
    
    for i, result in enumerate(results):
        metadata = result['metadata']
        source = metadata.get('source', 'unknown')
        doc_type = metadata.get('doc_type', 'unknown')
        similarity = result['similarity']
        
        print(f"  Result {i+1}:")
        print(f"    Source: {source}")
        print(f"    Type: {doc_type}")
        print(f"    Similarity: {similarity:.4f}")
        
        # Check if it's a startup file
        if 'startup' in source.lower():
            print(f"    🚀 STARTUP FILE!")