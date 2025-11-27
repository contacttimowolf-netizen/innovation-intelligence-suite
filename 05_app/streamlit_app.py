# AUTOMOTIVE TECH INTELLIGENCE - STREAMLIT APP
# Complete RAG interface with patent definitions and updated source mapping

import streamlit as st
import sys
import os
import importlib.util

def get_correct_paths():
    """Get absolute paths based on your exact folder structure"""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 05_app folder
    project_root = os.path.dirname(current_dir)  # innovation-intelligence-suite
    
    rag_components_path = os.path.join(project_root, '03_notebooks', 'rag_notebooks', 'rag_components')
    vector_index_path = os.path.join(project_root, '04_models', 'vector_index')
    
    return rag_components_path, vector_index_path, project_root

def import_your_components():
    """Import your actual tested components with exact paths"""
    rag_components_path, _, _ = get_correct_paths()
    retriever_path = os.path.join(rag_components_path, 'retriever.py')
    
    if not os.path.exists(retriever_path):
        return None, f"Retriever not found at: {retriever_path}"
        
    try:
        if rag_components_path not in sys.path:
            sys.path.insert(0, rag_components_path)
        
        spec = importlib.util.spec_from_file_location("retriever", retriever_path)
        retriever_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(retriever_module)
        return retriever_module, None
    except Exception as e:
        return None, f"Error importing retriever: {str(e)}"

def setup_groq_client():
    """Your exact client setup from notebook 03"""
    try:
        from groq import Groq
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            return None, "GROQ_API_KEY not found in environment variables"
        
        client = Groq(api_key=api_key)
        return client, None
    except ImportError:
        return None, "Groq package not installed. Run: pip install groq"
    except Exception as e:
        return None, f"Error setting up Groq client: {str(e)}"

def build_smart_prompt(question, context):
    """UPDATED universal prompt template with patent definitions"""
    # Detect if this is a technology maturity question
    maturity_keywords = ['trl', 'mature', 'transition', 'academy to application', 
                        'commercial', 'moving from academy', 'readiness', 'development stage']
    
    # Detect if this is a patent-related question
    patent_keywords = ['patent', 'intellectual property', 'ip', 'jurisdiction', 'ep', 'us', 'wo',
                      'kind', 'a1', 'b2', 'filing', 'protection', 'patent office', 'lens']
    
    question_lower = question.lower()
    is_maturity_question = any(keyword in question_lower for keyword in maturity_keywords)
    is_patent_question = any(keyword in question_lower for keyword in patent_keywords)
    
    # Include TRL section only for maturity questions
    if is_maturity_question:
        trl_section = """
TECHNOLOGY MATURITY ASSESSMENT:
- When discussing technology readiness, reference these stages:
  * Research Phase (TRL 1-4): Basic research, lab validation
  * Development Phase (TRL 5-6): Prototyping, testing  
  * Commercialization Phase (TRL 7-9): Deployment, scaling
- Assess current stage based on evidence in context
- Identify transition indicators and timelines
"""
    else:
        trl_section = ""
    
    # Include patent definitions only for patent questions
    if is_patent_question:
        patent_section = """
PATENT DOCUMENT INTERPRETATION:
- JURISDICTION indicates geographic protection scope:
  * EP: European Patent Office (multiple European countries)
  * US: United States Patent and Trademark Office
  * WO: World Intellectual Property Organization (PCT international applications)
  
- KIND CODES indicate document type and status:
  * A1: Patent application with search report
  * A2: Patent application without search report  
  * A3: Search report published separately
  * B1: Granted patent (examined and approved)
  * B2: Amended/revised granted patent
  
- Consider jurisdiction for market focus and protection scope
- Use kind codes to distinguish between applications (A) and granted patents (B)
"""
    else:
        patent_section = ""
    
    prompt = f"""
CONTEXT:
{context}

USER QUESTION:
{question}

ANALYSIS INSTRUCTIONS:
1. Provide a comprehensive answer based strictly on the context provided
2. Cite specific sources for each key point
3. If the context is insufficient, acknowledge what cannot be answered

{trl_section}
{patent_section}

ADDITIONAL GUIDELINES:
- For technology maturity questions: assess development stage and transition evidence
- For patent questions: consider jurisdiction and document type implications
- For trend questions: identify velocity, drivers, and key players  
- For descriptive questions: provide specific examples and entities

ANSWER:
"""
    return prompt

def determine_source_count(question):
    """YOUR dynamic source counting from notebook 03"""
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in ['summarize', 'trends', 'overview', 'comprehensive']):
        return 5
    elif any(keyword in question_lower for keyword in ['which', 'list', 'show me']):
        return 4
    elif any(keyword in question_lower for keyword in ['specific', 'exact', 'precise']):
        return 2
    else:
        return 3

def format_source_name(source_file):
    """UPDATED file name formatting with new data sources"""
    name_mapping = {
        # Automotive Papers
        'a_benchmark_framework_for_AL_models_in_automotive_aerodynamics.txt': 'Benchmark Framework for AI Models in Automotive Aerodynamics',
        'AL_agents_in_engineering_design_a_multiagent_framework_for_aesthetic_and_aerodynamic_car_design.txt': 'AI Agents in Engineering Design',
        'automating_automotive_software_development_a_synergy_of_generative_AL_and_formal_methods.txt': 'Automating Automotive Software Development',
        'automotive-software-and-electronics-2030-full-report.txt': 'Automotive Software and Electronics 2030 Report',
        'drive_disfluency-rich_synthetic_dialog_data_generation_framework_for_intelligent_vehicle_environments.txt': 'DRIVE Framework for Intelligent Vehicles',
        'Embedded_acoustic_intelligence_for_automotive_systems.txt': 'Embedded Acoustic Intelligence',
        'enhanced_drift_aware_computer_vision_achitecture_for_autonomous_driving.txt': 'Enhanced Computer Vision for Autonomous Driving',
        'Gen_AL_in_automotive_applications_challenges_and_opportunities_with_a_case_study_on_in-vehicle_experience.txt': 'Generative AI in Automotive',
        'generative_AL_for_autonomous_driving_a_review.txt': 'Generative AI for Autonomous Driving',
        'leveraging_vision_language_models_for_visual_grounding_and_analysis_of_automative_UI.txt': 'Vision-Language Models for Automotive UI',
        
        # Tech Reports
        'bog_ai_value_2025.txt': 'BCG: AI Value Creation 2025',
        'mckinsey_tech_trends_2025.txt': 'McKinsey Technology Trends 2025',
        'wef_emerging_tech_2025.txt': 'WEF: Emerging Technologies 2025',
        
        # New Processed Files
        'startups_processed.txt': 'Startup Innovation Database',
        'automotive_papers_processed.txt': 'Automotive Research Papers Database',
        'automotive_patents_processed.txt': 'Automotive Technology Patents Database',
    }
    return name_mapping.get(source_file, source_file.replace('.txt', '').replace('_', ' ').title())

# Initialize components with lazy loading
@st.cache_resource
def initialize_rag_system():
    """Initialize all RAG components using exact paths"""
    rag_components_path, vector_index_path, project_root = get_correct_paths()
    
    # Check if vector index exists
    if not os.path.exists(vector_index_path):
        return None, None, f"Vector index not found at: {vector_index_path}"
    
    # Import retriever
    retriever_module, retriever_error = import_your_components()
    if retriever_error:
        return None, None, retriever_error
    
    # Setup Groq client
    groq_client, groq_error = setup_groq_client()
    if groq_error:
        return None, None, groq_error
    
    # Initialize retriever
    try:
        retriever = retriever_module.DocumentAwareRetriever(vector_index_path)
        return retriever, groq_client, None
    except Exception as e:
        return None, None, f"Error initializing retriever: {str(e)}"

def ask_question(question, retriever, groq_client):
    """YOUR complete RAG pipeline from notebook 03"""
    try:
        k = determine_source_count(question)
        retrieved_data = retriever.retrieve_with_sources(question, k=k)
        
        if not retrieved_data:
            return {
                'answer': "I couldn't find relevant information in our knowledge base for this specific question. Try asking about automotive AI, tech trends, startup innovations, or patents.",
                'sources': [],
                'success': True,
                'source_count': k
            }
        
        context = "\n\n".join([
            f"Source: {format_source_name(item['source_file'])} | Type: {item['doc_type']}\nContent: {item['content']}"
            for item in retrieved_data
        ])
        
        prompt = build_smart_prompt(question, context)
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        return {
            'answer': answer,
            'sources': retrieved_data,
            'success': True,
            'source_count': k
        }
        
    except Exception as e:
        return {
            'answer': f"I encountered an error while processing your question: {str(e)}",
            'sources': [],
            'success': False
        }

# STREAMLIT UI
def main():
    st.set_page_config(
        page_title="INNOVATION INTELLIGENCE SUITE", 
        page_icon="🚗", 
        layout="wide"
    )
    
    st.title("INNOVATION INTELLIGENCE SUITE")
    st.markdown("Ask questions about latest tech trends in the automotive industry, including patents and startups.")
    
    # Initialize system with session state for persistence
    if 'rag_initialized' not in st.session_state:
        with st.spinner("Loading your RAG system..."):
            st.session_state.retriever, st.session_state.groq_client, error = initialize_rag_system()
            
            if error:
                st.session_state.rag_initialized = False
                st.error(f"❌ System initialization failed")
                
                # User-friendly error messages with solutions
                if "Groq package not installed" in error:
                    st.error("**Missing Dependency**")
                    st.info("""
                    **Solution:** Install the Groq package:
                    ```bash
                    pip install groq
                    ```
                    Then restart the Streamlit app.
                    """)
                elif "GROQ_API_KEY" in error:
                    st.error("**API Key Missing**")
                    st.info("""
                    **Solution:** Add your Groq API key to the `.env` file in the project root:
                    ```
                    GROQ_API_KEY=your_actual_api_key_here
                    ```
                    """)
                elif "Vector index not found" in error:
                    st.error("**Knowledge Base Missing**")
                    st.info("""
                    **Solution:** Generate the vector index first by running your RAG notebooks.
                    The vector index should be in `04_models/vector_index/`
                    """)
                else:
                    st.error(f"**Error:** {error}")
                    
            elif st.session_state.retriever and st.session_state.groq_client:
                st.session_state.rag_initialized = True
                st.success("✅ RAG system ready!")
            else:
                st.session_state.rag_initialized = False
                st.error("❌ Failed to initialize RAG system")
    
    # Only show query interface if system is initialized
    if not st.session_state.get('rag_initialized', False):
        st.warning("Please fix the initialization issues above to use the query system.")
        
        # Show debug info in expander
        with st.expander("🔧 Technical Details"):
            rag_path, vector_path, project_root = get_correct_paths()
            st.write(f"**Project Root:** `{project_root}`")
            st.write(f"**RAG Components Path:** `{rag_path}`")
            st.write(f"**Vector Index Path:** `{vector_path}`")
            st.write(f"**retriever.py exists:** {os.path.exists(os.path.join(rag_path, 'retriever.py'))}")
            st.write(f"**vector_index exists:** {os.path.exists(vector_path)}")
            
            # Check .env file
            env_path = os.path.join(project_root, '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    env_content = f.read()
                st.write(f"**.env exists:** Yes")
                st.write(f"**GROQ_API_KEY in .env:** {'GROQ_API_KEY' in env_content}")
            else:
                st.write(f"**.env exists:** No")
        
        return
    
    # Query interface (only shown when system is ready)
    st.success("🎉 System ready! Ask your question below.")
    
    # Initialize question input in session state if not exists
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ""
    
    # Query input
    question = st.text_input(
        "💬 Your question:",
        value=st.session_state.question_input,
        placeholder="e.g., Which startups work on AI for automotive?",
        key="question_input"
    )
    
    # Pre-defined query buttons
    st.subheader("📋 Example Questions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔬 Latest AI Research", use_container_width=True, key="research_btn"):
            st.session_state.question_input = "Summarize the latest research on AI and autonomous driving."
            st.rerun()
        if st.button("📜 Automotive Patents", use_container_width=True, key="patents_btn"):
            st.session_state.question_input = "What are the key patents in automotive AI with US jurisdiction?"
            st.rerun()
        if st.button("🚀 Startups in AI Automotive", use_container_width=True, key="startups_btn"):
            st.session_state.question_input = "Which startups work on AI for automotive?"
            st.rerun()
    
    with col2:
        if st.button("📈 Tech Trends", use_container_width=True, key="trends_btn"):
            st.session_state.question_input = "Show me recent reports on technology trends."
            st.rerun()
        if st.button("🤖 AI Agents Development", use_container_width=True, key="agents_btn"):
            st.session_state.question_input = "Summarize latest tech trends in development of AI agents"
            st.rerun()
        if st.button("🎯 Tech Maturity", use_container_width=True, key="maturity_btn"):
            st.session_state.question_input = "Which automotive technologies are moving from academy to application?"
            st.rerun()
    
    # Process question
    if question:
        with st.spinner("🔍 Searching documents and generating answer..."):
            result = ask_question(question, st.session_state.retriever, st.session_state.groq_client)
        
        # Display results
        st.subheader("📝 Answer")
        st.write(result['answer'])
            
        # Display sources if available
        if result['sources']:
            st.subheader(f"📚 Sources ({len(result['sources'])} documents)")
            for i, source in enumerate(result['sources']):
                readable_name = format_source_name(source['source_file'])
                with st.expander(f"📄 {readable_name} (Relevance: {source['similarity_score']:.3f})"):
                    st.write(source['content'])
        
        st.markdown("---")
        st.caption("Powered by RAG + Groq/Llama | Innovation Intelligence Suite")

if __name__ == "__main__":
    main()