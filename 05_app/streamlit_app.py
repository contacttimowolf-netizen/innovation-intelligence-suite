import streamlit as st
import sys
import os
sys.path.append('../03_notebooks')

# Use the same nuclear import for Streamlit
import importlib.util

def import_rag_components():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, '..', '03_notebooks')
    
    # Import retriever
    retriever_path = os.path.join(parent_dir, 'rag_components', 'retriever.py')
    spec = importlib.util.spec_from_file_location("retriever", retriever_path)
    retriever_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retriever_module)
    
    # Import query_interface  
    query_interface_path = os.path.join(parent_dir, 'rag_components', 'query_interface.py')
    spec = importlib.util.spec_from_file_location("query_interface", query_interface_path)
    query_interface_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(query_interface_module)
    
    # Import answer_generator
    answer_generator_path = os.path.join(parent_dir, 'rag_components', 'answer_generator.py')
    spec = importlib.util.spec_from_file_location("answer_generator", answer_generator_path)
    answer_generator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(answer_generator_module)
    
    return (retriever_module.DocumentAwareRetriever, 
            query_interface_module.SimpleQueryInterface,
            answer_generator_module.RAGAnswerGenerator)

@st.cache_resource
def load_pipeline():
    DocumentAwareRetriever, SimpleQueryInterface, RAGAnswerGenerator = import_rag_components()
    retriever = DocumentAwareRetriever("../04_models/vector_index")
    query_interface = SimpleQueryInterface(retriever)
    return RAGAnswerGenerator(query_interface)

st.set_page_config(page_title="Automotive Tech RAG", page_icon="🚗")
st.title("🚗 Automotive Tech RAG System")
st.markdown("Ask questions about automotive technology, startups, and research")

question = st.text_input("Enter your question:")

if st.button("Search") and question:
    answer_generator = load_pipeline()
    result = answer_generator.generate_answer(question, top_k=5)
    
    st.subheader("Answer")
    st.write(result["answer"])
    
    st.subheader(f"Sources ({len(result['sources'])} found)")
    for source in result["sources"]:
        with st.expander(f"{source['source_file']} (Score: {source['relevance_score']:.3f})"):
            st.write(f"**Type:** {source['doc_type']}")
            st.write(f"**Content:** {source['content_snippet']}")
