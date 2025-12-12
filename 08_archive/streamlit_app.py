# AUTOMOTIVE TECH INTELLIGENCE - STREAMLIT APP
# Complete RAG interface with Query Expansion

import streamlit as st
import sys
import os
import importlib.util

import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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

def import_query_expander():
    """Import the query expander module"""
    rag_components_path, _, _ = get_correct_paths()
    expander_path = os.path.join(rag_components_path, 'query_expander.py')
    
    if not os.path.exists(expander_path):
        # Create the file if it doesn't exist
        with open(expander_path, 'w') as f:
            f.write('''# Query expander placeholder - will be created automatically''')
    
    try:
        if rag_components_path not in sys.path:
            sys.path.insert(0, rag_components_path)
        
        spec = importlib.util.spec_from_file_location("query_expander", expander_path)
        expander_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(expander_module)
        return expander_module, None
    except Exception as e:
        return None, f"Error importing query expander: {str(e)}"

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
    """UPDATED universal prompt template with patent AND startup definitions"""
    # Detect if this is a technology maturity question
    maturity_keywords = ['trl', 'mature', 'transition', 'academy to application', 
                        'commercial', 'moving from academy', 'readiness', 'development stage']
    
    # Detect if this is a patent-related question
    patent_keywords = ['patent', 'intellectual property', 'ip', 'jurisdiction', 'ep', 'us', 'wo',
                      'kind', 'a1', 'b2', 'filing', 'protection', 'patent office', 'lens']
    
    # Detect if this is a startup-related question
    startup_keywords = ['startup', 'startups', 'company', 'companies', 'venture', 'business', 
                       'funding', 'investment', 'series a', 'series b', 'series c', 'backed']
    
    question_lower = question.lower()
    is_maturity_question = any(keyword in question_lower for keyword in maturity_keywords)
    is_patent_question = any(keyword in question_lower for keyword in patent_keywords)
    is_startup_question = any(keyword in question_lower for keyword in startup_keywords)
    
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
- Include a definition of TRL stages in the answer
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
    
    # Include startup guidance only for startup questions
    if is_startup_question:
        startup_section = """
CRITICAL INSTRUCTIONS FOR STARTUP QUERIES:
1. **EXTRACT ALL SPECIFIC STARTUP/COMPANY NAMES** mentioned in the context
2. **FOCUS ON STARTUP DATABASES**: Pay special attention to sections from "Automotive Startup Profiles & Tracker" and "Automotive Industry Startups to Watch in 2025"
3. **FOR EACH STARTUP FOUND**:
   * State the company name clearly and prominently
   * Describe their primary technology or business focus
   * Include location information if available
   * Mention any funding details (rounds raised, investors)
   * Note their automotive/AI specialization
4. **REQUIRED ANSWER STRUCTURE**:
   - Start with a summary of findings
   - Then provide a CLEAR, NUMBERED LIST of startups
   - Format: "1. **Company Name**: [description] [Source: filename]"
   - Cite the specific source file for each piece of information
5. **IF STARTUPS EXIST IN CONTEXT BUT AREN'T EXPLICITLY MENTIONED**, still extract them
6. **IF NO STARTUPS ARE FOUND**, clearly state: "No specific startup companies were found in the available documents."
7. **PRIORITIZE INFORMATION FROM STARTUP DATABASES** over general reports when answering startup questions

EXAMPLE FORMAT:
"Based on the startup databases, I found these automotive AI companies:

1. **Company X**: Develops AI perception systems for autonomous vehicles. Based in Berlin. [Source: Automotive Startup Profiles & Tracker]
2. **Company Y**: Specializes in battery management AI for electric vehicles. Raised $20M Series A. [Source: Automotive Industry Startups to Watch in 2025]"
"""
    else:
        startup_section = ""
    
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
{startup_section}

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
        
        # New Processed Files (UPDATED)
        'autotechinsight_startups_processed.txt': 'Automotive Startup Profiles & Tracker',
        'seedtable_startups_processed.txt': 'Automotive Industry Startups to Watch in 2025',
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
        return None, None, None, f"Vector index not found at: {vector_index_path}"
    
    # Import retriever
    retriever_module, retriever_error = import_your_components()
    if retriever_error:
        return None, None, None, retriever_error
    
    # Import query expander
    expander_module, expander_error = import_query_expander()
    if expander_error and "placeholder" not in expander_error:
        print(f"Note: Query expander not fully available: {expander_error}")
    
    # Setup Groq client
    groq_client, groq_error = setup_groq_client()
    if groq_error:
        return None, None, None, groq_error
    
    # Initialize retriever
    try:
        retriever = retriever_module.DocumentAwareRetriever(vector_index_path)
        
        # Initialize query expander if module loaded
        query_expander = None
        if expander_module and not expander_error:
            try:
                query_expander = expander_module.QueryExpander()
                print("✅ Query expander initialized")
            except Exception as e:
                print(f"⚠️ Query expander init warning: {e}")
                query_expander = None
        
        return retriever, groq_client, query_expander, None
        
    except Exception as e:
        return None, None, None, f"Error initializing retriever: {str(e)}"

def retrieve_with_expansion(question, retriever, query_expander=None, k=3):
    """Enhanced retrieval with query expansion (NO VISUAL DISPLAY)"""
    all_results = []
    
    # Generate query variations if expander is available
    if query_expander:
        try:
            expanded_queries = query_expander.expand_query(question, use_llm=True)
            
            # ⚠️ REMOVED: Do NOT display search variations to user
            # Just use them silently in the background
            
            # Retrieve for each query variation
            for query_variant in expanded_queries:
                try:
                    results = retriever.retrieve_with_sources(query_variant, k=2)
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error retrieving for variant '{query_variant}': {e}")
                    continue
        except Exception as e:
            print(f"Query expansion failed: {e}")
            # Fallback to regular retrieval
            results = retriever.retrieve_with_sources(question, k=k)
            all_results.extend(results)
    else:
        # No expansion available - use regular retrieval
        results = retriever.retrieve_with_sources(question, k=k)
        all_results.extend(results)
    
    if not all_results:
        return []
    
    # Remove duplicates while preserving order
    unique_results = []
    seen_content = set()
    
    for result in all_results:
        # Create a signature based on content and source
        content_start = result['content'][:150]  # First 150 chars
        source = result['source_file']
        signature = f"{source}:{content_start}"
        
        if signature not in seen_content:
            seen_content.add(signature)
            unique_results.append(result)
    
    # Sort by similarity score and return top k
    unique_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
    return unique_results[:k]

def retrieve_with_forced_startups(question, retriever, query_expander=None, k=3):
    """🚀 FIXED: FORCE-INCLUDE startup files with query expansion"""
    question_lower = question.lower()
    startup_keywords = ['startup', 'company', 'venture', 'business', 'firm', 'enterprise', 'companies', 'startups']
    
    is_startup_query = any(keyword in question_lower for keyword in startup_keywords)
    
    # Get initial results with query expansion
    initial_results = retrieve_with_expansion(question, retriever, query_expander, k=k)
    
    # If it's a startup query, check and force-add startup files
    if is_startup_query:
        # Check if we already have startup files in results
        has_startup_files = False
        for item in initial_results:
            if any(startup_file in item['source_file'] 
                   for startup_file in ['autotechinsight_startups_processed.txt', 'seedtable_startups_processed.txt']):
                has_startup_files = True
                break
        
        # If no startup files found, force add them
        if not has_startup_files:
            # Try specific startup-related queries
            startup_specific_queries = [
                "automotive startup AI technology",
                "AI automotive companies startups",
                "autonomous vehicle startup companies",
                "automotive tech ventures",
                "electric vehicle AI startups"
            ]
            
            startup_items = []
            for startup_query in startup_specific_queries:
                try:
                    startup_results = retriever.retrieve_with_sources(startup_query, k=1)
                    for item in startup_results:
                        # Only take items from startup files
                        if any(startup_file in item['source_file'] 
                              for startup_file in ['autotechinsight_startups_processed.txt', 'seedtable_startups_processed.txt']):
                            # Check for duplicates with existing results
                            is_duplicate = False
                            for existing in initial_results:
                                if item['content'][:100] == existing['content'][:100]:
                                    is_duplicate = True
                                    break
                            if not is_duplicate:
                                startup_items.append(item)
                                if len(startup_items) >= 2:  # Get at most 2 startup items
                                    break
                    if len(startup_items) >= 2:
                        break
                except Exception as e:
                    continue
            
            # Add startup items to results
            if startup_items:
                # Combine and re-sort
                combined_results = startup_items + initial_results
                combined_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                return combined_results[:k]
        
        # If we already have startup files or couldn't find any, return initial results
        return initial_results[:k] if initial_results else []
    else:
        # Non-startup query, just return initial results
        return initial_results[:k] if initial_results else []

def ask_question(question, retriever, groq_client, query_expander=None):
    """UPDATED RAG pipeline with query expansion"""
    try:
        k = determine_source_count(question)
        
        # Use enhanced retrieval with query expansion
        retrieved_data = retrieve_with_forced_startups(
            question, 
            retriever, 
            query_expander=query_expander, 
            k=k
        )
        
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
            st.session_state.retriever, st.session_state.groq_client, st.session_state.query_expander, error = initialize_rag_system()
            
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
                if st.session_state.query_expander:
                    st.success("✅ RAG system ready with query expansion!")
                else:
                    st.success("✅ RAG system ready! (Query expansion not available)")
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
            st.write(f"**query_expander.py exists:** {os.path.exists(os.path.join(rag_path, 'query_expander.py'))}")
            
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
    
    # Initialize button flags if not exists
    if 'research_clicked' not in st.session_state:
        st.session_state.research_clicked = False
    if 'patents_clicked' not in st.session_state:
        st.session_state.patents_clicked = False
    if 'startups_clicked' not in st.session_state:
        st.session_state.startups_clicked = False
    if 'trends_clicked' not in st.session_state:
        st.session_state.trends_clicked = False
    if 'agents_clicked' not in st.session_state:
        st.session_state.agents_clicked = False
    if 'maturity_clicked' not in st.session_state:
        st.session_state.maturity_clicked = False
    
    # Check for button clicks BEFORE creating the text input
    # This must happen BEFORE the text_input widget is created
    
    if st.session_state.research_clicked:
        st.session_state.question_input = "Summarize the latest research on AI and autonomous driving."
        st.session_state.research_clicked = False
    elif st.session_state.patents_clicked:
        st.session_state.question_input = "What are the key patents in automotive AI with US jurisdiction?"
        st.session_state.patents_clicked = False
    elif st.session_state.startups_clicked:
        st.session_state.question_input = "Which startups work on AI for automotive?"
        st.session_state.startups_clicked = False
    elif st.session_state.trends_clicked:
        st.session_state.question_input = "Show me recent reports on technology trends."
        st.session_state.trends_clicked = False
    elif st.session_state.agents_clicked:
        st.session_state.question_input = "Summarize latest tech trends in development of AI agents"
        st.session_state.agents_clicked = False
    elif st.session_state.maturity_clicked:
        st.session_state.question_input = "Which automotive technologies are moving from academy to application?"
        st.session_state.maturity_clicked = False
    
    # Query input - NOW this comes AFTER button checks
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
            st.session_state.research_clicked = True
            st.rerun()
        if st.button("📜 Automotive Patents", use_container_width=True, key="patents_btn"):
            st.session_state.patents_clicked = True
            st.rerun()
        if st.button("🚀 Startups in AI Automotive", use_container_width=True, key="startups_btn"):
            st.session_state.startups_clicked = True
            st.rerun()
    
    with col2:
        if st.button("📈 Tech Trends", use_container_width=True, key="trends_btn"):
            st.session_state.trends_clicked = True
            st.rerun()
        if st.button("🤖 AI Agents Development", use_container_width=True, key="agents_btn"):
            st.session_state.agents_clicked = True
            st.rerun()
        if st.button("🎯 Tech Maturity", use_container_width=True, key="maturity_btn"):
            st.session_state.maturity_clicked = True
            st.rerun()
    
    # Process question
    if question:
        with st.spinner("🔍 Searching documents and generating answer..."):
            result = ask_question(
                question, 
                st.session_state.retriever, 
                st.session_state.groq_client,
                query_expander=st.session_state.query_expander
            )
        
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
        st.caption("Powered by RAG + Query Expansion + Groq/Llama | Innovation Intelligence Suite")

if __name__ == "__main__":
    main()