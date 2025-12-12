# debug_faiss.py - Run this from rag_components directory
import sys
import os

print("🔍 DIAGNOSING FAISS INDEX")

from faiss_retriever import FAISSRetriever

retriever = FAISSRetriever()

# 1. Check document types
print("\n📊 Document type distribution:")
doc_counts = retriever.get_doc_type_counts()
for doc_type, count in doc_counts.items():
    print(f"  {doc_type}: {count}")

# 2. Check what startup-related documents exist
print("\n🔎 Searching for startup-related documents in metadata...")
startup_docs = []
for i, metadata in enumerate(retriever.metadatas[:1000]):  # Check first 1000
    if 'startup' in str(metadata).lower():
        startup_docs.append((i, metadata))

print(f"Found {len(startup_docs)} documents with 'startup' in metadata")
if startup_docs:
    for idx, metadata in startup_docs[:5]:  # Show first 5
        print(f"  Index {idx}: {metadata}")

# 3. Check source files
print("\n📁 Checking source files...")
source_files = set()
for metadata in retriever.metadatas[:1000]:
    source = metadata.get('source', metadata.get('source_file', ''))
    if source:
        source_files.add(source)

print(f"Unique source files (first 1000 docs): {len(source_files)}")
for file in sorted(list(source_files))[:10]:  # Show first 10
    print(f"  {file}")

# 4. Try different queries
print("\n🧪 Testing different queries:")
test_queries = [
    "startup",
    "company",
    "automotive company",
    "AI automotive",
    "electric vehicle",
    "technology",
    "research"
]

for query in test_queries:
    results = retriever.retrieve(query, top_k=2, similarity_threshold=0.3)
    print(f"  '{query}': {len(results)} results")
    if results:
        # Show source of first result
        source = results[0]['metadata'].get('source', results[0]['metadata'].get('source_file', 'unknown'))
        doc_type = results[0]['metadata'].get('doc_type', results[0]['metadata'].get('type', 'unknown'))
        print(f"    First result: {source} ({doc_type})")

# 5. Check metadata structure
print("\n🔧 Checking metadata structure of first few documents...")
for i in range(3):
    print(f"\nDocument {i} metadata:")
    for key, value in retriever.metadatas[i].items():
        print(f"  {key}: {value}")
    
    # Also show text preview
    preview = retriever.texts[i][:100] + "..." if len(retriever.texts[i]) > 100 else retriever.texts[i]
    print(f"  Preview: {preview}")