# AUTOMOTIVE TECH INNOVATION INTELLIGENCE SUITE - STREAMLIT APP
# Complete RAG interface with Predictive Model Integration

import streamlit as st
import sys
import os
import importlib.util
import re
import matplotlib.pyplot as plt  # For plotting graphs

# ADDED: Technology definitions dictionary - UPDATED with correct formatting
AUTO_TOP_SEEDS_1 = {
    "Sensor_Fusion": "multi sensor fusion architecture combining lidar radar and camera into unified perception outputs",
    "Occupancy_Grid": "spatial occupancy grid mapping for free space and obstacle representation around the vehicle",
    "SLAM": "simultaneous localization and mapping using onboard sensors for ego pose and map building in dynamic traffic",
    "Trajectory_Prediction": "future motion and path prediction of vehicles and vulnerable road users in traffic scenes",
    "Environment_Modeling": "semantic scene and object relationship modeling for a structured driving environment representation"
}
AUTO_TOP_SEEDS_2 = {
    "4G": "4g lte cellular communication for connected vehicles",
    "5G": "5g cellular communication for ultra low latency vehicle connectivity",
    "Wireless_Communication": "wireless communication protocols for vehicle data transmission",
    "V2V": "vehicle to vehicle direct communication",
    "V2I": "vehicle to infrastructure communication",
    "V2X": "v2x communication framework including v2v and v2i",
    "Edge_Computing": "edge computing for real time vehicle data processing",
    "Fog_Computing": "fog computing layer between edge and cloud",
    "Cloud_Computing": "cloud computing backend for vehicle data storage and processing"
}
AUTO_TOP_SEEDS_3 = {
    "Renewable_Energy": "renewable energy systems",
    "Solar_Cell": "photovoltaic solar cells (solar cell, perovskite solar, sensitized solar, organic photovoltaics, quantum dots, silicon solar)",
    "Electrochemical_Energy": "electrochemical energy systems (electrochemical energy, hydrogen evolution)",
    "Nano_Energy": "nanoscale energy systems (nano energy, quantum dots)",
    "Natural_Gas_Energy": "natural gas based energy systems (natural gas)",
}
AUTO_TOP_SEEDS_4 = {
    "Battery_Management_System": "battery management and control (battery management bms, battery management system, state charge soc, state health soh, battery state health)",
    "High_Energy_Batteries": "lithium ion battery technology (li ion batteries, ion batteries electric, performance lithium ion, high energy density)",
    "Battery_Thermal_Management": "battery thermal and cooling systems (battery thermal management, thermal management systems)",
    "Battery_Diagnostics_EIS": "electrochemical battery diagnostics (electrochemical impedance spectroscopy)",
    "Battery_Performance_Prediction": "lithium ion aging and performance prediction (prediction lithium ion, RUL)"
}
AUTO_TOP_SEEDS_5 = {
    "Smart_Grid": "smart grid control and monitoring (smart grid technologies, power grid)",
    "Distributed_Energy_Resources": "distributed energy generation and control (distributed energy resources)",
    "V2G_G2V_Technologies": "bidirectional vehicle grid interaction (grid v2g technology, vehicle g2v, bidirectional energy)",
    "Charging_Infrastructure": "ev charging systems and network (charging infrastructure)",
    "Reactive_Power_Management": "reactive power control in grids (reactive power)",
    "DC_Microgrid": "dc based local energy networks (dc microgrid)"
}
AUTO_TOP_SEEDS_6 = {
    "Traffic_Planning": "urban traffic congestion dynamics (traffic congestion)",
    "Transport_Infrastructure": "urban transport network and infrastructure (transport infrastructure)",
    "Shared_Mobility": "ride sharing and shared transport systems (shared mobility)",
    "Mobility_Demand_Forecasting": "urban travel demand prediction (demand forecasting)",
    "Micro_Mobility": "e scooters bikes and small personal transport (micro mobility)",
    "Node_Activity": "traffic node and intersection activity dynamics (node activity)"
}
AUTO_TOP_SEEDS_7 = {
    "InLine_Quality_Inspection": "inline defect and weld quality inspection (defect detection, weld quality)",
    "Error_Proofing_PokaYoke": "mistake proofing and error prevention (poka yoke)",
    "Predictive_Maintenance": "vehicle and equipment predictive maintenance (vehicle maintenance, predictive maintenance pd)",
    "Process_Monitoring_Optimization": "real time monitoring and waste minimizing stable processes (real time monitoring, proactively finding deviations, minimizing waste optimizing, quality constant process)"
}
AUTO_TOP_SEEDS_8 = {
    "Autonomous_Delivery_Robots": "autonomous robotic delivery and last mile transport (automated delivery, robotic delivery shipping)",
    "Warehouse_Intelligence_Robots": "autonomous warehouse robots for inventory tracking mapping inspection and stock counting (counting stock warehouse)",
    "AGV_Systems": "automated guided vehicles for structured factory and warehouse transport (automated guided vehicle)",
    "Hybrid_Modular_Robotics": "hybrid and modular robotic system architectures combining multiple robot types into reconfigurable platforms (hybrid modular)"
}
AUTO_TOP_SEEDS_9 = {
    "Intrusion_Detection": "network intrusion detection (intrusion detection)",
    "Cyber_Physical_Security": "security of cyber physical automotive systems (cyber physical)",
    "InVehicle_Network_Protocols": "automotive communication and bus protocols (controller area network, protocols)",
    "Cryptography_Key_Management": "encryption ciphering and cryptographic key management (ciphering key)",
    "Integrity_Protection": "data and message integrity protection mechanisms (integrity protection)",
    "Functional_Safety": "automotive functional safety and fail safe systems (functional safety)"
}
AUTO_TOP_SEEDS = {
    "Perception": AUTO_TOP_SEEDS_1,
    "Communication_Technologies": AUTO_TOP_SEEDS_2,
    "Energy_Source": AUTO_TOP_SEEDS_3,
    "Energy_Storage": AUTO_TOP_SEEDS_4,
    "Energy_Management": AUTO_TOP_SEEDS_5,
    "Urban_Mobility": AUTO_TOP_SEEDS_6,
    "Manufacturing": AUTO_TOP_SEEDS_7,
    "Robotics": AUTO_TOP_SEEDS_8,
    "Cybersecurity": AUTO_TOP_SEEDS_9,
}

def get_correct_paths():
    """Get absolute paths based on your exact folder structure"""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 05_app folder
    project_root = os.path.dirname(current_dir)  # innovation-intelligence-suite
    
    rag_components_path = os.path.join(project_root, '03_notebooks', 'rag_notebooks', 'rag_components')
    vector_index_path = os.path.join(project_root, '04_models', 'vector_index')
    
    return rag_components_path, vector_index_path, project_root

def import_your_components():
    """Import FAISS retriever with exact paths"""
    rag_components_path, _, _ = get_correct_paths()
    
    # FAISS retriever
    faiss_retriever_path = os.path.join(rag_components_path, 'faiss_retriever.py')
    
    if not os.path.exists(faiss_retriever_path):
        return None, f"FAISS Retriever not found at: {faiss_retriever_path}"
        
    try:
        if rag_components_path not in sys.path:
            sys.path.insert(0, rag_components_path)
        
        spec = importlib.util.spec_from_file_location("faiss_retriever", faiss_retriever_path)
        retriever_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(retriever_module)
        return retriever_module, None
    except Exception as e:
        return None, f"Error importing FAISS retriever: {str(e)}"

def import_predictive_components():
    """Import predictive model components"""
    _, _, project_root = get_correct_paths()

    from pathlib import Path
    notebooks_root = Path(project_root) / "03_notebooks"
    predictive_components_path = notebooks_root / "predictive_notebooks" / "predictive_components"

    if not predictive_components_path.exists():
        return None, "Predictive components directory not found"

    if str(notebooks_root) not in sys.path:
        sys.path.insert(0, str(notebooks_root))

    try:
        from predictive_notebooks.predictive_components import analytics as predictive_analytics

        predictive_functions = {
            "load_area_tech_ts": predictive_analytics.load_area_tech_ts,
            "get_fastest_growing_topics": predictive_analytics.get_fastest_growing_topics,
            "get_likely_to_mature_next_year": predictive_analytics.get_likely_to_mature_next_year,
            "plot_simple_timeseries": predictive_analytics.plot_simple_timeseries,  # ADDED: New function
        }

        return predictive_functions, None

    except ImportError as e:
        return None, f"Error importing analytics module: {e}"
    except Exception as e:
        return None, f"Error in import_predictive_components: {e}"

def import_query_expander():
    """Import the existing query expander module (if it exists)"""
    rag_components_path, _, _ = get_correct_paths()
    expander_path = os.path.join(rag_components_path, 'query_expander.py')
    
    # Only import if file exists
    if not os.path.exists(expander_path):
        return None, None  # Returns None if no expander available
    
    try:
        if rag_components_path not in sys.path:
            sys.path.insert(0, rag_components_path)
        
        spec = importlib.util.spec_from_file_location("query_expander", expander_path)
        expander_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(expander_module)
        return expander_module, None
    except Exception as e:
        # Silent fail - system works without expander
        return None, None

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

def analyze_question_type(question):
    """Enhanced question type analysis with predictive model detection"""
    question_lower = question.lower()
    
    # Define question type characteristics
    question_type = {
        'is_startup_query': any(keyword in question_lower for keyword in 
                               ['startup', 'company', 'companies', 'venture', 'business', 'funding']),
        'is_patent_query': any(keyword in question_lower for keyword in 
                              ['patent', 'intellectual property', 'ip', 'jurisdiction', 'ep', 'us', 'wo']),
        'is_research_query': any(keyword in question_lower for keyword in 
                                ['research', 'study', 'paper', 'academic', 'method', 'experiment']),
        'is_trend_query': any(keyword in question_lower for keyword in 
                             ['trend', 'forecast', 'future', 'emerging', 'development', 'innovation', 
                              'pain point', 'challenge', 'barrier', 'obstacle']),
        
        # PREDICTIVE MODEL QUERIES - Simplified detection
        'is_growth_query': any(keyword in question_lower for keyword in 
                              ['fastest growing', 'growth rate', 'increasing', 'growing technology',
                               'accelerating', 'expanding', 'growth trend', 'rapid growth']),
        
        'is_maturity_query': any(keyword in question_lower for keyword in 
                                ['maturity', 'readiness', 'trl', 'commercial', 'stage', 'transition',
                                 'hype cycle', 'adoption', 'development stage', 'technology readiness',
                                 'commercialization', 'scaling', 'deployment', 'market readiness',
                                 'innovation trigger', 'peak of expectations', 'trough of disillusionment',
                                 'slope of enlightenment', 'plateau of productivity',
                                 'likely to mature', 'mature next year', 'reaching maturity',
                                 'next 12 months', 'coming year', 'near future', 'next year']),
        
        'is_technology_query': any(keyword in question_lower for keyword in 
                                  ['technology', 'tech', 'system', 'solution', 'application', 
                                   'algorithm', 'agent', 'agents', 'model', 'framework']),
        
        'is_list_query': any(keyword in question_lower for keyword in 
                            ['list', 'which', 'what are', 'name', 'show me', 'examples']),
        
        'is_summary_query': any(keyword in question_lower for keyword in 
                               ['summarize', 'overview', 'explain', 'describe', 'what is']),
    }
    
    # Determine optimal thresholds based on question type
    if question_type['is_startup_query'] or question_type['is_list_query']:
        semantic_threshold = 0.15  # Lower for broad coverage
        keyword_threshold = 0.1    # Very low for keyword matching
        k_semantic = 4             # More semantic results
        k_keyword = 2              # Fewer keyword results
    elif question_type['is_patent_query']:
        semantic_threshold = 0.25  # Moderate for technical queries
        keyword_threshold = 0.15   # Low for technical terms
        k_semantic = 4
        k_keyword = 2
    elif question_type['is_research_query']:
        semantic_threshold = 0.3   # Standard for academic content
        keyword_threshold = 0.2    # Standard keyword matching
        k_semantic = 3
        k_keyword = 2
    else:
        semantic_threshold = 0.25  # Balanced for general queries
        keyword_threshold = 0.15   # Standard keyword matching
        k_semantic = 3
        k_keyword = 2
    
    return question_type, semantic_threshold, keyword_threshold, k_semantic, k_keyword

def determine_question_category(question):
    """Determine which system should handle the question - SIMPLIFIED"""
    question_type, _, _, _, _ = analyze_question_type(question)
    
    # Priority: Predictive Model Questions
    if question_type['is_growth_query']:
        return 'predictive_growth'
    elif question_type['is_maturity_query'] and any(word in question.lower() for word in ['next', 'future', 'coming', 'likely', 'forecast', 'prediction', '12 months']):
        return 'predictive_maturity'
    
    # Default: RAG-Only Questions
    else:
        return 'rag_only'

def format_predictive_results(results_df, category):
    """Format predictive model results for display with technology definitions - ROBUST VERSION"""
    # Check if results_df is None or empty
    if results_df is None:
        return "Error: Predictive model returned None results."
    
    if hasattr(results_df, 'empty') and results_df.empty:
        return "No predictive insights available for this query (empty results)."
    
    try:
        # GROWTH QUERY FORMAT
        if category == 'predictive_growth':
            formatted = "**Fastest Growing Automotive Technologies**\n\n"
            
            # Check if we have the required columns
            required_cols = ['auto_tech_cluster', 'auto_focus_area', 'growth_slope_n_total']
            missing_cols = [col for col in required_cols if col not in results_df.columns]
            
            if missing_cols:
                return f"Error: Missing required columns in growth data: {missing_cols}\nAvailable columns: {list(results_df.columns)}"
            
            # Create formatted output for each technology
            for idx, row in enumerate(results_df.head(10).itertuples(), 1):
                tech = row.auto_tech_cluster
                tech_display = tech.replace('_', ' ') if isinstance(tech, str) else str(tech)
                
                area = row.auto_focus_area
                area_display = area.replace('_', ' ') if isinstance(area, str) else str(area)
                
                growth = getattr(row, 'growth_slope_n_total', 0)
                
                # Get definition
                definition = "N/A"
                area_key = area_display.replace(' ', '_')
                if area_key in AUTO_TOP_SEEDS and tech in AUTO_TOP_SEEDS[area_key]:
                    definition = AUTO_TOP_SEEDS[area_key][tech]
                    # Capitalize first letter if definition exists
                    if definition and len(definition) > 0:
                        definition = definition[0].upper() + definition[1:]
                
                # Format as requested - each item on separate line with explicit line breaks
                formatted += f"**{idx}. {tech_display}**\n\n"
                formatted += f"**Definition:** {definition}\n\n"
                formatted += f"**Area:** {area_display}\n\n"
                
                if isinstance(growth, (int, float)):
                    # Convert growth rate to percentage
                    growth_pct = growth * 100
                    formatted += f"**Growth Rate:** {growth_pct:.1f}%\n\n"
                else:
                    formatted += f"**Growth Rate:** {growth}\n\n"
                
                recent_activity = getattr(row, 'n_total_last', getattr(row, 'recent_activity', 0))
                formatted += f"**Recent Activity:** {int(recent_activity)} documents\n\n"
        
        # MATURITY QUERY FORMAT  
        elif category == 'predictive_maturity':
            formatted = "**Technologies Likely to Mature in Coming Year**\n\n"
            
            # Check if we have the required columns
            required_cols = ['auto_tech_cluster', 'auto_focus_area', 'last_share_patent', 'forecast_share_patent_mean', 'delta_share_patent']
            missing_cols = [col for col in required_cols if col not in results_df.columns]
            
            if missing_cols:
                return f"Error: Missing required columns in maturity data: {missing_cols}\nAvailable columns: {list(results_df.columns)}"
            
            # Create formatted output for each technology
            for idx, row in enumerate(results_df.head(15).itertuples(), 1):
                tech = row.auto_tech_cluster
                tech_display = tech.replace('_', ' ') if isinstance(tech, str) else str(tech)
                
                area = row.auto_focus_area
                area_display = area.replace('_', ' ') if isinstance(area, str) else str(area)
                
                # Get patent percentages
                current_pct = getattr(row, 'last_share_patent', 0) * 100
                forecast_pct = getattr(row, 'forecast_share_patent_mean', 0) * 100
                growth_pct = getattr(row, 'delta_share_patent', 0) * 100
                
                # Get definition
                definition = "N/A"
                area_key = area_display.replace(' ', '_')
                if area_key in AUTO_TOP_SEEDS and tech in AUTO_TOP_SEEDS[area_key]:
                    definition = AUTO_TOP_SEEDS[area_key][tech]
                    # Capitalize first letter if definition exists
                    if definition and len(definition) > 0:
                        definition = definition[0].upper() + definition[1:]
                
                # Format exactly as requested - each item on separate line with explicit line breaks
                formatted += f"**{idx}. {tech_display}**\n\n"
                formatted += f"**Definition:** {definition}\n\n"
                formatted += f"**Area:** {area_display}\n\n"
                formatted += f"**Current:** {current_pct:.1f}% patents\n\n"
                formatted += f"**Forecast:** {forecast_pct:.1f}% patents\n\n"
                formatted += f"**Growth:** +{growth_pct:.1f}%\n\n"
        
        else:
            formatted = f"**Predictive Insights**\n\n"
            if hasattr(results_df, 'head'):
                formatted += results_df.head(10).to_markdown()
            else:
                formatted += str(results_df)
        
        return formatted
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error formatting predictive results: {str(e)}\n\nDebug info:\n{error_details}"
    

def build_smart_prompt(question, context, predictive_insights=None):
    """Your existing prompt template - UPDATED with consistent requirements"""
    question_lower = question.lower()
    
    # Detect query type for targeted guidance
    is_startup_question = any(keyword in question_lower for keyword in 
                            ['startup', 'company', 'companies', 'venture', 'business', 'funding'])
    
    is_patent_question = any(keyword in question_lower for keyword in 
                           ['patent', 'intellectual property', 'ip', 'jurisdiction', 'ep', 'us', 'wo'])
    
    is_research_question = any(keyword in question_lower for keyword in 
                             ['research', 'study', 'paper', 'academic', 'scientific', 'methodology',
                              'journal', 'article', 'publication'])
    
    is_trend_question = any(keyword in question_lower for keyword in 
                          ['trend', 'forecast', 'future', 'emerging', 'development', 'innovation', 'pain point', 'challenge'])
    
    is_maturity_question = any(keyword in question_lower for keyword in 
                             ['trl', 'maturity', 'readiness', 'commercial', 'transition', 'stage'])
    
    is_technology_question = any(keyword in question_lower for keyword in 
                               ['technology', 'tech', 'system', 'solution', 'application', 'deployment', 'agent', 'agents'])
    
    # Build targeted guidance sections - UPDATED
    guidance_sections = []
    
    # Patents - UPDATED with strict requirements
    if is_patent_question:
        guidance_sections.append("""
🔍 **PATENT QUERY GUIDANCE - STRICT REQUIREMENTS:**

**MANDATORY PATENT FORMAT (For each patent mentioned):**
1. **PATENT DETAILS**: Always include: Patent number
2. **ABSTRACT SUMMARY**: 
   - ALWAYS include a summary of the patent abstract
   - If no abstract is available in context, DO NOT MENTION THE PATENT AT ALL
   - Skip patents entirely if abstract information is missing
3. **JURISDICTION ANALYSIS**: Clearly indicate jurisdiction type:
   - EP: European Patent Office (covers multiple European countries)
   - US: United States Patent and Trademark Office
   - WO: World Intellectual Property Organization (international/PCT applications)
4. **TECHNOLOGY FOCUS**: Specific automotive/AI technologies protected
5. **KEY DATES**: Include filing date, publication date, grant date when available
6. **SOURCE FORMAT**: [Source: Lens.org Automotive Technology Patents 2025] - USE THIS EXACT FORMAT
""")
    
    # Startups
    if is_startup_question:
        guidance_sections.append("""
🚀 **STARTUP QUERY GUIDANCE:**
1. **EXTRACT COMPANY NAMES**: All startup/company names mentioned
2. **INCLUDE DETAILS**: Location, founding year, funding stage, key technologies
3. **FOCUS ON DATABASES**: Prioritize information from startup-specific sources
4. **ORGANIZE CLEARLY**: Create numbered lists with consistent formatting
5. **HIGHLIGHT AI FOCUS**: Note AI applications in automotive context
6. **SOURCE FORMAT**: [Source: Seedtable Best Automotive Industry Startups to Watch in 2025] OR [Source: AutoTechInsight Automotive Startup Profiles & Tracker]
""")
    
    # Research - UPDATED with strict requirements
    if is_research_question:
        guidance_sections.append("""
📚 **RESEARCH QUERY GUIDANCE - STRICT REQUIREMENTS:**

**MANDATORY RESEARCH PAPER FORMAT (For each paper mentioned):**
1. **PAPER DETAILS**: Always include: Title, authors, institution/affiliation, publication year
2. **ABSTRACT SUMMARY**: 
   - ALWAYS include a short summary of the research (2-3 sentences)
   - Summarize key findings, methodology, and contributions
   - Focus on automotive/AI applications and relevance
3. **TECHNICAL DETAILS**: Specific algorithms, models, datasets, methodologies used
4. **KEY FINDINGS**: Main conclusions and results
5. **AUTOMOTIVE RELEVANCE**: Practical applications in automotive context
6. **SOURCE FORMAT**: [Source: Lens.org Automotive Research Papers Abstracts 2025] - USE THIS EXACT FORMAT
""")
    
    # Trends
    if is_trend_question:
        guidance_sections.append("""
📈 **TREND/CHALLENGE GUIDANCE:**
1. **IDENTIFY KEY TRENDS/PAIN POINTS**: Major developments, challenges, or patterns
2. **EXTRACT VELOCITY INDICATORS**: Growth rates, adoption curves, investment trends
3. **NOTE DRIVERS & BARRIERS**: Factors enabling or hindering adoption
4. **HIGHLIGHT KEY PLAYERS**: Companies, institutions mentioned
5. **PROVIDE EXAMPLES**: Specific technologies or cases mentioned
6. **COMPARE SOURCES**: Note consistency or variations across different reports
7. **SOURCE FORMAT**: Use appropriate source format based on document type
""")
    
    # Maturity
    if is_maturity_question:
        guidance_sections.append("""        
🎯 **TECHNOLOGY MATURITY & HYPE CYCLE GUIDANCE:**

**FRAMEWORK FOR ASSESSMENT:**
1. **HYPE CYCLE PHASES** (Use when mentioned or implied):
   - 📈 **Innovation Trigger**: Early research, proof-of-concept, initial patents
   - 🚀 **Peak of Inflated Expectations**: Media hype, startup boom, high funding
   - 🛑 **Trough of Disillusionment**: Implementation failures, skepticism, consolidation
   - 📚 **Slope of Enlightenment**: Practical applications, standards, pilot projects
   - 🏭 **Plateau of Productivity**: Mainstream adoption, price competition, services market

2. **TECHNOLOGY READINESS LEVELS (TRL)** (When specifically mentioned):
   - TRL 1-4: Basic research, lab validation (Academic focus)
   - TRL 5-6: Prototyping, testing (University-industry collaboration)
   - TRL 7-9: Deployment, scaling (Industry dominant)

3. **ACADEMIC TO INDUSTRY TRANSFER INDICATORS**:
   - Academic papers → Industry patents
   - Research grants → Venture funding
   - University labs → Startup formations
   - Conference talks → Product demonstrations

**ANALYSIS REQUIREMENTS:**
1. **IDENTIFY CURRENT STAGE**: Based on evidence in context
2. **EXTRACT TRANSITION EVIDENCE**: Patents, funding, partnerships, deployments
3. **ASSESS TIMELINES**: When technology moved/might move between stages
4. **PROVIDE SPECIFIC EXAMPLES**: Companies, products, projects mentioned
5. **SOURCE FORMAT**: Use appropriate source format based on document type
""")
    
    # Technology
    if is_technology_question:
        guidance_sections.append("""
⚙️ **TECHNOLOGY QUERY GUIDANCE:**
1. **EXTRACT SPECIFICS**: Technology names, versions, capabilities
2. **IDENTIFY APPLICATIONS**: How technologies are used in automotive context
3. **NOTE PERFORMANCE METRICS**: Speed, accuracy, efficiency improvements
4. **ASSESS INTEGRATION**: How technologies work together or integrate
5. **HIGHLIGHT INNOVATIONS**: Novel approaches or breakthroughs
6. **COMPARE ALTERNATIVES**: Different technology options mentioned
7. **SOURCE FORMAT**: Use appropriate source format based on document type
""")
    
    # UNIFIED REFERENCING REQUIREMENTS - NEW SECTION
    unified_referencing = """
🔗 **UNIFIED REFERENCING REQUIREMENTS - STRICTLY ENFORCED:**

**MANDATORY SOURCE FORMATTING:**
ALWAYS use these EXACT source formats - no variations allowed:

1. **PATENTS**: [Source: Lens.org Automotive Technology Patents 2025]
2. **RESEARCH PAPERS**: [Source: Lens.org Automotive Research Papers Abstracts 2025]
3. **STARTUPS**: [Source: Seedtable Best Automotive Industry Startups to Watch in 2025] 
   OR [Source: AutoTechInsight Automotive Startup Profiles & Tracker]
4. **INDUSTRY REPORTS**: 
   - [Source: BCG: AI Value Creation 2025]
   - [Source: McKinsey Technology Trends 2025]
   - [Source: WEF: Emerging Technologies 2025]
   - [Source: Automotive Software 2030 Report]
5. **TECHNICAL PAPERS**: Use specific paper titles with icons as shown in context

**REFERENCING RULES:**
- Each key fact must have a source reference
- Use the EXACT format shown above - no deviations
- Place reference at the end of the sentence or paragraph
- If multiple facts from same source, reference once at end of paragraph
"""
    
    # General guidance - UPDATED
    general_guidance = """
📋 **GENERAL ANSWER GUIDELINES:**
1. **BE SPECIFIC**: Use exact names, numbers, dates from context
2. **BE COMPREHENSIVE**: Cover all relevant aspects of the question
3. **BE STRUCTURED**: Use clear organization (numbered lists, sections)
4. **BE ACCURATE**: Only use information from the provided context
5. **CITE SOURCES**: For each key point, include source in EXACT format specified above
6. **ACKNOWLEDGE LIMITATIONS**: If information is incomplete, state what's missing

**SPECIAL REQUIREMENTS:**
- For patents: Only include if abstract is available, otherwise skip entirely
- For research papers: Always include a short summary (2-3 sentences)
- Use consistent source formatting as specified in Unified Referencing section
"""
    
    # Combine all guidance with unified referencing
    targeted_guidance = "\n\n".join(guidance_sections)
    
    prompt = f"""
CONTEXT:
{context}

USER QUESTION:
{question}

ANALYSIS INSTRUCTIONS:
You are an automotive technology intelligence analyst. Your task is to provide detailed, accurate answers based strictly on the context provided.

{targeted_guidance}

{unified_referencing}

{general_guidance}

FORMAT REQUIREMENTS:
- Use **bold** for company names, technology names, patent numbers
- Use numbered lists for multiple items (e.g., 1., 2., 3.)
- Use bullet points for sub-items within descriptions
- Include specific metrics (percentages, amounts, dates) when available
- Group related information together (e.g., by technology, by company, by region)

ANSWER STRUCTURE:
1. Direct answer to the main question
2. Supporting details with specific examples
3. Source citations in EXACT format specified above
4. Summary or implications if relevant

**CRITICAL REMINDERS:**
1. **PATENTS**: Only mention patents that have abstracts in the context. Skip patents without abstracts.
2. **RESEARCH PAPERS**: Always include a short summary (2-3 sentences) for each paper mentioned.
3. **SOURCES**: Always use the EXACT source formats specified above - no variations.

ANSWER:
"""
    return prompt


def determine_source_count(question):
    """Dynamic source counting based on question type - UNCHANGED"""
    question_lower = question.lower()
    
    # Complex questions need more sources
    if any(keyword in question_lower for keyword in ['summarize', 'comprehensive', 'overall', 'complete', 'latest']):
        return 5
    # List questions need more sources for coverage
    elif any(keyword in question_lower for keyword in ['list', 'which', 'what are', 'show all', 'show me']):
        return 5
    # Specific questions can use fewer sources
    elif any(keyword in question_lower for keyword in ['specific', 'exact', 'precise', 'detailed']):
        return 3
    # Default for most questions
    else:
        return 4

def format_source_name(source_file):
    """Enhanced file name formatting - WITHOUT EMOJIS for consistency"""
    name_mapping = {
        # Automotive Papers
        'a_benchmark_framework_for_AL_models_in_automotive_aerodynamics.txt': 'AI in Automotive Aerodynamics Research',
        'AL_agents_in_engineering_design_a_multiagent_framework_for_aesthetic_and_aerodynamic_car_design.txt': 'AI Agents in Car Design Research',
        'automating_automotive_software_development_a_synergy_of_generative_AL_and_formal_methods.txt': 'AI for Automotive Software Development',
        'automotive-software-and-electronics-2030-full-report.txt': 'Automotive Software 2030 Report',
        'drive_disfluency-rich_synthetic_dialog_data_generation_framework_for_intelligent_vehicle_environments.txt': 'AI Dialogue Systems for Vehicles',
        'Embedded_acoustic_intelligence_for_automotive_systems.txt': 'Acoustic AI for Automotive Systems',
        'enhanced_drift_aware_computer_vision_achitecture_for_autonomous_driving.txt': 'Computer Vision for Autonomous Driving',
        'Gen_AL_in_automotive_applications_challenges_and_opportunities_with_a_case_study_on_in-vehicle_experience.txt': 'Generative AI in Automotive Applications',
        'generative_AL_for_autonomous_driving_a_review.txt': 'Generative AI for Autonomous Driving Review',
        'leveraging_vision_language_models_for_visual_grounding_and_analysis_of_automative_UI.txt': 'Vision-Language Models for Automotive UI',
        
        # Tech Reports
        'bog_ai_value_2025.txt': 'BCG: AI Value Creation 2025',
        'mckinsey_tech_trends_2025.txt': 'McKinsey Technology Trends 2025',
        'wef_emerging_tech_2025.txt': 'WEF: Emerging Technologies 2025',
        
        # Processed Files - CRITICAL: These must match what the LLM will output
        'autotechinsight_startups_processed.txt': 'AutoTechInsight Automotive Startup Profiles & Tracker',
        'seedtable_startups_processed.txt': 'Seedtable Best Automotive Industry Startups to Watch in 2025',
        'automotive_papers_processed.txt': 'Lens.org Automotive Research Papers Abstracts 2025',
        'automotive_patents_processed.txt': 'Lens.org Automotive Technology Patents 2025',
        
        # Generic fallbacks
        'startup': 'Startup Database',
        'patent': 'Patent Database',
        'paper': 'Research Database',
        'report': 'Industry Report',
    }
    
    # Try exact match first
    if source_file in name_mapping:
        return name_mapping[source_file]
    
    # Try partial matching
    source_lower = source_file.lower()
    for key, value in name_mapping.items():
        if key in source_lower:
            return value
    
    # Default formatting
    return source_file.replace('.txt', '').replace('_', ' ').title()

# Initialize components with lazy loading
@st.cache_resource
def initialize_rag_system():
    """Initialize all RAG components using exact paths - UNCHANGED"""
    rag_components_path, vector_index_path, project_root = get_correct_paths()
    
    # Check if FAISS vector index exists - updated check for FAISS files
    if not os.path.exists(vector_index_path):
        return None, None, None, f"Vector index not found at: {vector_index_path}"
    
    # Check for FAISS files specifically
    faiss_files = ['faiss_index.bin', 'texts.pkl', 'metadata.pkl']
    missing_files = []
    for file in faiss_files:
        if not os.path.exists(os.path.join(vector_index_path, file)):
            missing_files.append(file)
    
    if missing_files:
        return None, None, None, f"FAISS files missing: {', '.join(missing_files)}. Did you run the FAISS embedding creation?"
    
    # Import FAISS retriever
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
    
    # Initialize FAISS retriever
    try:
        retriever = retriever_module.FAISSRetriever(vector_index_path)
        
        # Initialize query expander if module loaded
        query_expander = None
        if expander_module and not expander_error:
            try:
                query_expander = expander_module.QueryExpander()
                print("✅ Enhanced query expander initialized")
            except Exception as e:
                print(f"⚠️ Query expander init warning: {e}")
                query_expander = None
        
        return retriever, groq_client, query_expander, None
        
    except Exception as e:
        return None, None, None, f"Error initializing FAISS retriever: {str(e)}"

@st.cache_resource
def initialize_predictive_system():
    """Initialize predictive model components"""
    try:
        predictive_functions, error = import_predictive_components()
        if error:
            print(f"⚠️ Predictive system warning: {error}")
            return None
        return predictive_functions
    except Exception as e:
        print(f"⚠️ Predictive system initialization error: {e}")
        return None

def get_targeted_keyword_queries(question, question_type):
    """Generate targeted keyword queries based on question type - SIMPLIFIED"""
    question_lower = question.lower()
    keyword_queries = []
    
    # Startup related keywords
    if question_type['is_startup_query']:
        keyword_queries.extend([
            "automotive startup",
            "AI company automotive",
            "autonomous vehicle company",
            "electric vehicle startup",
            "mobility tech company",
            "car technology startup",
            "vehicle AI startup",
            "automotive venture capital",
            "tech startup funding",
            "emerging automotive companies"
        ])
    
    # Patent related keywords
    if question_type['is_patent_query']:
        keyword_queries.extend([
            "automotive patent",
            "AI patent automotive",
            "vehicle technology patent",
            "intellectual property automotive",
            "patent EP automotive",
            "patent US automotive",
            "patent WO automotive",
            "automotive invention",
            "vehicle innovation patent",
            "autonomous driving patent"
        ])
    
    # Research related keywords
    if question_type['is_research_query']:
        keyword_queries.extend([
            "automotive research",
            "AI study automotive",
            "vehicle technology research",
            "autonomous driving study",
            "electric vehicle research",
            "automotive AI paper",
            "vehicle system research",
            "mobility technology study",
            "automotive engineering research",
            "intelligent vehicle study"
        ])
    
    # Trend related keywords
    if question_type['is_trend_query']:
        keyword_queries.extend([
            "automotive trend",
            "AI trend automotive",
            "vehicle technology trend",
            "emerging automotive technology",
            "future of automotive",
            "automotive innovation trend",
            "mobility future trend",
            "vehicle market trend",
            "automotive industry trend",
            "tech trend automotive",
            "automotive challenge",
            "AI challenge automotive",
            "vehicle technology barrier",
            "automotive adoption barrier"
        ])
    
    # Maturity related keywords
    if question_type['is_maturity_query']:
        keyword_queries.extend([
           # TRL & Readiness Keywords
            "technology readiness automotive",
            "TRL automotive",
            "maturity automotive technology",
            "commercialization automotive",
            "development stage automotive",
            "automotive technology adoption",
            "vehicle tech readiness",
            "automotive innovation maturity",
            "scaling automotive technology",
            "deployment automotive AI",
            
            # Hype Cycle Keywords
            "hype cycle automotive",
            "innovation trigger automotive",
            "peak of expectations automotive",
            "trough of disillusionment automotive",
            "slope of enlightenment automotive",
            "plateau of productivity automotive",
            "technology adoption curve automotive",
            "market adoption automotive",
            
            # Market Readiness Keywords
            "market readiness automotive",
            "industry adoption automotive",
            "mainstream adoption automotive",
            "commercial deployment automotive",
            "production ready automotive",
            "enterprise adoption automotive"
        ])
    
    # Technology/Agent related keywords
    if question_type['is_technology_query']:
        # Extract technology terms from question
        tech_terms = re.findall(r'\b[A-Z][a-z]+\b', question)
        for term in tech_terms[:3]:  # Use first 3 capitalized terms
            if len(term) > 3 and term.lower() not in ['ai', 'automotive', 'vehicle', 'car']:  # Avoid common words
                keyword_queries.append(f"{term} automotive")
                keyword_queries.append(f"{term} vehicle")
        
        # Special handling for AI agents
        if 'agent' in question_lower or 'agents' in question_lower:
            keyword_queries.extend([
                "AI agent automotive",
                "intelligent agent vehicle",
                "autonomous agent system",
                "multi-agent system automotive",
                "agent-based automotive"
            ])
        
        keyword_queries.extend([
            "automotive technology",
            "vehicle system",
            "car technology",
            "automotive solution",
            "vehicle application",
            "automotive AI system",
            "intelligent vehicle technology",
            "connected car technology",
            "autonomous driving system",
            "electric vehicle technology"
        ])
    
    # General automative/AI keywords (for all queries)
    keyword_queries.extend([
        "automotive AI",
        "vehicle artificial intelligence",
        "car machine learning",
        "autonomous vehicle",
        "electric vehicle",
        "connected car",
        "smart mobility",
        "intelligent transportation"
    ])
    
    # Remove duplicates and limit
    return list(set(keyword_queries))[:8]  # Max 8 keyword queries

def universal_hybrid_retrieval(question, retriever, query_expander=None, k=4):
    """Your existing retrieval function - UNCHANGED"""
    all_results = []
    
    # Identify question type
    question_type, semantic_threshold, keyword_threshold, k_semantic, k_keyword = analyze_question_type(question)
    
    # 1. Semantic search with query expansion
    if query_expander:
        try:
            expanded_queries = query_expander.expand_query(question, use_llm=False)
            if not expanded_queries:
                expanded_queries = [question]
        except:
            expanded_queries = [question]
    else:
        expanded_queries = [question]
    
    for query in expanded_queries[:3]:  # Use first 3 expanded queries
        try:
            semantic_results = retriever.retrieve_with_sources(
                query, 
                k=k_semantic, 
                threshold=semantic_threshold
            )
            all_results.extend(semantic_results)
        except Exception as e:
            continue
    
    # 2. Targeted keyword search
    keyword_queries = get_targeted_keyword_queries(question, question_type)
    
    for keyword_query in keyword_queries[:5]:  # Use first 5 keyword queries
        try:
            keyword_results = retriever.retrieve_with_sources(
                keyword_query,
                k=k_keyword,
                threshold=keyword_threshold
            )
            
            # Filter to ensure relevance
            filtered_results = []
            for result in keyword_results:
                content = result.get('text', result.get('content', '')).lower()
                # Check if result contains relevant terms
                if any(term in content for term in keyword_query.split()[:2]):
                    filtered_results.append(result)
            
            all_results.extend(filtered_results)
        except Exception as e:
            continue
    
    # Ensure relevant document types are included based on question type
    required_doc_types = []
    
    if question_type['is_startup_query']:
        required_doc_types.extend(['startup', 'seedtable', 'autotech'])
    
    if question_type['is_patent_query']:
        required_doc_types.extend(['patent', 'lens'])
    
    if question_type['is_research_query']:
        required_doc_types.extend(['paper', 'research', 'study', 'academic'])
    
    # Check if we have required document types
    for doc_type in required_doc_types[:2]:  # Check first 2 required types
        has_type = False
        for result in all_results:
            source_file = result.get('source_file', '').lower()
            if doc_type in source_file:
                has_type = True
                break
        
        # If missing, try to find documents of this type
        if not has_type:
            try:
                type_results = retriever.retrieve_with_sources(
                    doc_type,
                    k=1,
                    threshold=0.05  # Extremely low threshold
                )
                all_results.extend(type_results)
            except:
                pass
    
    # Remove duplicates
    unique_results = []
    seen_content = set()
    
    for result in all_results:
        content = result.get('text', result.get('content', ''))
        content_start = content[:250]  # Use first 250 chars for deduplication
        source = result.get('source_file', 'unknown')
        signature = f"{source}:{content_start}"
        
        if signature not in seen_content:
            seen_content.add(signature)
            unique_results.append(result)
    
    # Enhance ranking: prioritize by relevance to question type
    def calculate_relevance_score(result, question_type):
        """Calculate enhanced relevance score based on question type"""
        base_score = result.get('similarity_score', 0)
        source_file = result.get('source_file', '').lower()
        
        # Type matching bonus
        type_bonus = 0
        
        if question_type['is_startup_query'] and any(keyword in source_file for keyword in ['startup', 'seedtable', 'autotech']):
            type_bonus += 0.3
        
        if question_type['is_patent_query'] and any(keyword in source_file for keyword in ['patent', 'lens']):
            type_bonus += 0.3
        
        if question_type['is_research_query'] and any(keyword in source_file for keyword in ['paper', 'research', 'study']):
            type_bonus += 0.2
        
        return base_score + type_bonus
    
    # Sort by enhanced relevance score
    unique_results.sort(key=lambda x: calculate_relevance_score(x, question_type), reverse=True)
    
    # Return top k results
    return unique_results[:k]

def retrieve_with_expansion(question, retriever, query_expander=None, k=4):
    """Main retrieval function - UNCHANGED"""
    return universal_hybrid_retrieval(question, retriever, query_expander, k)

def process_predictive_query(question, predictive_functions):
    """Handle questions that require predictive model"""
    if not predictive_functions:
        return {
            'answer': "⚠️ Predictive model components are not available. Please ensure the predictive model data files are properly set up.",
            'sources': [],
            'success': True,
            'source_count': 0,
            'predictive_used': False,
        }
    
    try:
        # Load time series data
        ts_data = predictive_functions['load_area_tech_ts']()
        
        # Get question category for routing
        category = determine_question_category(question)
        
        # Route to appropriate function
        if category == 'predictive_growth':
            results = predictive_functions['get_fastest_growing_topics'](ts_data, top_n=15)
            
            # Generate formatted text and graphs for EACH technology
            technology_items = []
            if not results.empty:
                for idx, row in enumerate(results.head(10).itertuples(), 1):
                    tech = row.auto_tech_cluster
                    tech_display = tech.replace('_', ' ') if isinstance(tech, str) else str(tech)
                    
                    area = row.auto_focus_area
                    area_display = area.replace('_', ' ') if isinstance(area, str) else str(area)
                    
                    growth = getattr(row, 'growth_slope_n_total', 0)
                    
                    # Get definition
                    definition = "N/A"
                    area_key = area_display.replace(' ', '_')
                    if area_key in AUTO_TOP_SEEDS and tech in AUTO_TOP_SEEDS[area_key]:
                        definition = AUTO_TOP_SEEDS[area_key][tech]
                        # Capitalize first letter if definition exists
                        if definition and len(definition) > 0:
                            definition = definition[0].upper() + definition[1:]
                    
                    # Convert growth rate to percentage
                    growth_pct = growth * 100 if isinstance(growth, (int, float)) else growth
                    
                    recent_activity = getattr(row, 'n_total_last', getattr(row, 'recent_activity', 0))
                    
                    # Format technology text
                    tech_text = f"**{idx}. {tech_display}**\n\n"
                    tech_text += f"**Definition:** {definition}\n\n"
                    tech_text += f"**Area:** {area_display}\n\n"
                    tech_text += f"**Growth Rate:** {growth_pct:.1f}%\n\n"
                    tech_text += f"**Recent Activity:** {int(recent_activity)} documents\n\n"
                    
                    # Generate graph for this technology
                    graph_fig = None
                    if 'plot_simple_timeseries' in predictive_functions:
                        try:
                            fig = predictive_functions['plot_simple_timeseries'](
                                ts_data, 
                                area=area, 
                                tech=tech,
                                value_col="n_total"
                            )
                            
                            # Make graph smaller
                            if hasattr(fig, 'set_size_inches'):
                                fig.set_size_inches(6, 3)  # Smaller size
                            if hasattr(fig, 'tight_layout'):
                                fig.tight_layout()
                            
                            graph_fig = fig
                        except Exception as e:
                            print(f"Warning: Could not generate graph for {tech}: {e}")
                            graph_fig = None
                    
                    # Add to list
                    technology_items.append({
                        'text': tech_text,
                        'figure': graph_fig,
                        'tech_display': tech_display,
                        'area_display': area_display,
                        'growth_rate': growth_pct
                    })
            
            # Don't include methodology note here - will add it at the end
            insights = "##### Fastest Growing Automotive Technologies\n\n"
            
        elif category == 'predictive_maturity':
            results = predictive_functions['get_likely_to_mature_next_year'](ts_data)
            
            # Generate formatted text WITHOUT graphs for maturity, but WITH separators
            formatted_text = "##### Technologies Likely to Mature in Coming Year\n\n"
            
            if not results.empty:
                for idx, row in enumerate(results.head(15).itertuples(), 1):
                    tech = row.auto_tech_cluster
                    tech_display = tech.replace('_', ' ') if isinstance(tech, str) else str(tech)
                    
                    area = row.auto_focus_area
                    area_display = area.replace('_', ' ') if isinstance(area, str) else str(area)
                    
                    # Get patent percentages
                    current_pct = getattr(row, 'last_share_patent', 0) * 100
                    forecast_pct = getattr(row, 'forecast_share_patent_mean', 0) * 100
                    growth_pct = getattr(row, 'delta_share_patent', 0) * 100
                    
                    # Get definition
                    definition = "N/A"
                    area_key = area_display.replace(' ', '_')
                    if area_key in AUTO_TOP_SEEDS and tech in AUTO_TOP_SEEDS[area_key]:
                        definition = AUTO_TOP_SEEDS[area_key][tech]
                        # Capitalize first letter if definition exists
                        if definition and len(definition) > 0:
                            definition = definition[0].upper() + definition[1:]
                    
                    # Format technology text WITH separator
                    formatted_text += f"**{idx}. {tech_display}**\n\n"
                    formatted_text += f"**Definition:** {definition}\n\n"
                    formatted_text += f"**Area:** {area_display}\n\n"
                    formatted_text += f"**Current:** {current_pct:.1f}% patents\n\n"
                    formatted_text += f"**Forecast:** {forecast_pct:.1f}% patents\n\n"
                    formatted_text += f"**Growth:** +{growth_pct:.1f}%\n\n"
                    
                    # Add separator after each technology (except the last one)
                    if idx < min(15, len(results)):
                        formatted_text += "---\n\n"
            
            insights = formatted_text
            technology_items = []  # No graphs for maturity
            
        else:
            return {
                'answer': "This question doesn't match any predictive model categories.",
                'sources': [],
                'success': True,
                'source_count': 0,
                'predictive_used': False,
            }
        
        # Format final answer
        answer = f"""

{insights}
**Methodology Note**
- Based on time-series analysis of automotive technology publications and patents
- Forecasts derived from historical growth patterns
- Updated with the latest available data
"""
        
        return {
            'answer': answer,
            'sources': [],
            'success': True,
            'source_count': 0,
            'predictive_used': True,
            'predictive_results': results.to_dict('records') if hasattr(results, 'to_dict') else [],
            'technology_items': technology_items if category == 'predictive_growth' else []  # Store technology items for growth only
        }
        
    except Exception as e:
        return {
            'answer': f"Error processing predictive query: {str(e)}",
            'sources': [],
            'success': False,
            'predictive_used': True,
            'technology_items': []
        }
    

def process_rag_query(question, retriever, groq_client, query_expander=None):
    """Handle RAG-based questions"""
    try:
        k = determine_source_count(question)
        
        # Use universal hybrid retrieval
        retrieved_data = retrieve_with_expansion(
            question, 
            retriever, 
            query_expander=query_expander, 
            k=k
        )
        
        if not retrieved_data:
            return {
                'answer': "I couldn't find relevant information in our knowledge base for this specific question.",
                'sources': [],
                'success': True,
                'source_count': k,
                'predictive_used': False,
                'graphs': []  # ADDED: Empty graphs list
            }
        
        # Build context from retrieved data
        context_parts = []
        for i, item in enumerate(retrieved_data):
            content = item.get('text', item.get('content', ''))
            source_file = item.get('source_file', 'unknown')
            
            readable_name = format_source_name(source_file)
            similarity = item.get('similarity_score', 0)
            
            context_parts.append(f"--- DOCUMENT {i+1} ---")
            context_parts.append(f"Source: {readable_name}")
            context_parts.append(f"Relevance Score: {similarity:.3f}")
            context_parts.append(f"Content:\n{content}")
            context_parts.append("")  # Empty line for separation
        
        context = "\n".join(context_parts)
        prompt = build_smart_prompt(question, context)
        
        # Adjust tokens based on context length
        max_tokens = 800 if len(context) > 3000 else 600
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        return {
            'answer': answer,
            'sources': retrieved_data,
            'success': True,
            'source_count': k,
            'predictive_used': False,
            'graphs': []  # ADDED: Empty graphs list
        }
        
    except Exception as e:
        return {
            'answer': f"I encountered an error while processing your question: {str(e)}",
            'sources': [],
            'success': False,
            'predictive_used': False,
            'graphs': []
        }

def ask_question(question, retriever, groq_client, predictive_functions=None, query_expander=None):
    """Main question processing function with simplified routing logic"""
    
    # Determine question category
    category = determine_question_category(question)
    
    # Route to appropriate processor
    if category in ['predictive_growth', 'predictive_maturity']:
        if predictive_functions:
            return process_predictive_query(question, predictive_functions)
        else:
            # Fall back to RAG if predictive not available
            return process_rag_query(question, retriever, groq_client, query_expander)
    else:
        # Everything else goes to RAG
        return process_rag_query(question, retriever, groq_client, query_expander)


def main():
    st.set_page_config(
        page_title="INNOVATION INTELLIGENCE SUITE", 
        page_icon="🚗", 
        layout="wide"
    )
    
    st.title("Uncover What's Next in Auto Tech")
    st.markdown("Developed as a Data Science + AI Bootcamp capstone project")
    
    # Initialize RAG system
    if 'rag_initialized' not in st.session_state:
        with st.spinner("Loading RAG system..."):
            st.session_state.retriever, st.session_state.groq_client, st.session_state.query_expander, error = initialize_rag_system()
            
            if error:
                st.session_state.rag_initialized = False
                st.error(f"❌ RAG system initialization failed")
                
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
                elif "Vector index not found" in error or "FAISS files missing" in error:
                    st.error("**Knowledge Base Missing**")
                    st.info("""
                    **Solution:** Generate the FAISS vector index first by running the FAISS embedding creation code in notebook 02.
                    The FAISS index should be in `04_models/vector_index/` with these files:
                    - faiss_index.bin
                    - texts.pkl  
                    - metadata.pkl
                    - embeddings.npy
                    """)
                elif "FAISS Retriever not found" in error:
                    st.error("**FAISS Retriever Missing**")
                    st.info("""
                    **Solution:** Make sure `faiss_retriever.py` exists in `rag_components/` folder.
                    Also install required packages:
                    ```bash
                    pip install faiss-cpu fastembed
                    ```
                    """)
                elif "Error importing FAISS retriever" in error:
                    st.error("**Import Error**")
                    st.info("""
                    **Solution:** Check that `faiss_retriever.py` has correct imports:
                    ```python
                    import faiss
                    from fastembed import TextEmbedding
                    ```
                    Install missing packages with:
                    ```bash
                    pip install faiss-cpu fastembed
                    ```
                    """)
                else:
                    st.error(f"**Error:** {error}")
                    
            elif st.session_state.retriever and st.session_state.groq_client:
                st.session_state.rag_initialized = True
            else:
                st.session_state.rag_initialized = False
                st.error("❌ Failed to initialize RAG system")
    
    # Initialize predictive system
    if 'predictive_initialized' not in st.session_state:
        with st.spinner("Loading predictive model..."):
            st.session_state.predictive_functions = initialize_predictive_system()
            if st.session_state.predictive_functions:
                st.session_state.predictive_initialized = True
            else:
                st.session_state.predictive_initialized = False
                st.warning("⚠️ Predictive model not available - using RAG only")
    
    # Only show query interface if RAG system is initialized
    if not st.session_state.get('rag_initialized', False):
        st.warning("Please fix the initialization issues above to use the system.")
        # Show footer even if system not initialized
        st.markdown("---")
        st.caption(f"Powered by Innovation Intelligence Suite (2025)")
        return
    
    # Initialize question input in session state if not exists
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ""
    
    # Simplified button state management
    button_questions = {
        'research_clicked': "Summarize the latest AI research on autonomous driving vehicles.",
        'patents_clicked': "Show me recent patents on AI for automotive vehicles.",
        'startups_clicked': "Which startups work on automotive and autonomous driving?",
        'trends_clicked': "What are the key challenges and pain points in automotive AI adoption?",
        'agents_clicked': "Summarize latest tech trends in development of AI agents.",
        'growth_clicked': "What are the fastest growing automotive technologies?",
        'maturity_clicked': "Which automotive technologies are reaching commercial maturity in the next 12 months?"
    }
    
    # Initialize button flags
    for flag in button_questions.keys():
        if flag not in st.session_state:
            st.session_state[flag] = False
    
    # Check for button clicks BEFORE creating the text input
    for flag, question_text in button_questions.items():
        if st.session_state[flag]:
            st.session_state.question_input = question_text
            st.session_state[flag] = False
            st.rerun()
    
    # Query input - NOW this comes AFTER button checks
    question = st.text_input(
        "💬 Your question:",
        placeholder="e.g., Which startups work on AI for automotive?",
        key="question_input"
    )
    
    # Pre-defined query buttons
    st.subheader("Example Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🚀 Innovation Intelligence**")
        if st.button("Latest AI Research", use_container_width=True, key="research_btn"):
            st.session_state.research_clicked = True
            st.rerun()
        if st.button("Automotive Patents", use_container_width=True, key="patents_btn"):
            st.session_state.patents_clicked = True
            st.rerun()
        if st.button("Startups in Automotive", use_container_width=True, key="startups_btn"):
            st.session_state.startups_clicked = True
            st.rerun()
    
    with col2:
        st.markdown("**📈 Market Insights**")
        if st.button("Industry Pain Points", use_container_width=True, key="trends_btn"):
            st.session_state.trends_clicked = True
            st.rerun()
        if st.button("AI Agents Development", use_container_width=True, key="agents_btn"):
            st.session_state.agents_clicked = True
            st.rerun()
    
    with col3:
        st.markdown("**🔮 Predictive Analytics**")
        if st.button("Fastest Growing Tech", use_container_width=True, key="growth_btn"):
            st.session_state.growth_clicked = True
            st.rerun()
        if st.button("Tech Maturity", use_container_width=True, key="transition_btn"):
            st.session_state.maturity_clicked = True
            st.rerun()
    
    # Process question
    if question:
        # Determine processing type for status message
        category = determine_question_category(question)
        
        if category.startswith('predictive_'):
            status_msg = "🔮 Running predictive analysis..."
        else:
            status_msg = "🔍 Searching documents and generating answer..."
        
        with st.spinner(status_msg):
            result = ask_question(
                question, 
                st.session_state.retriever, 
                st.session_state.groq_client,
                predictive_functions=st.session_state.predictive_functions,
                query_expander=st.session_state.query_expander
            )
        
        # Display results with appropriate header
        st.subheader("📝 **Answer**")
        
        if result.get('predictive_used', False) and 'technology_items' in result and result['technology_items']:
            # For growth queries: Show each technology with its graph
            st.markdown("##### Fastest Growing Automotive Technologies")
            
            # Display each technology with its graph
            for item in result['technology_items']:
                # Create container for this technology
                tech_container = st.container()
                
                with tech_container:
                    # Create columns for layout: left for text, right for graph
                    col1, col2 = st.columns([2, 1])  # 2:1 ratio for text:graph
                    
                    with col1:
                        # Display technology information
                        st.markdown(item['text'])
                    
                    with col2:
                        # Display graph if available
                        if item['figure']:
                            # Add centered heading above the graph
                            st.markdown(f"<div style='text-align: center;'><b>{item['tech_display']} - Growth Trend</b></div>", unsafe_allow_html=True)
                            st.pyplot(item['figure'], use_container_width=True)
                
                st.markdown("---")  # Separator between technologies
            
            # Add methodology note at the bottom
            st.markdown("**Methodology Note**")
            st.markdown("- Based on time-series analysis of automotive technology publications and patents")
            st.markdown("- Forecasts derived from historical growth patterns")
            st.markdown("- Updated with the latest available data")
            
            st.caption("*Based on Time-Series Predictive Modelling*")
            
        elif result.get('predictive_used', False):
            # For maturity queries: Show as text only (no graphs)
            # The answer already includes separators from the formatting function
            st.markdown(result['answer'])
            st.caption("*Based on Time-Series Predictive Modelling*")
            
        else:
            # For RAG queries
            # Remove colon from the first line if present and format as smaller heading
            answer_lines = result['answer'].split('\n')
            if len(answer_lines) > 0:
                # Process first line
                first_line = answer_lines[0].strip()
                # Remove colon if present
                first_line = first_line.replace(':', '')
                # Remove bold markers if present
                first_line = first_line.replace('**', '')
                # Format as smaller heading
                answer_lines[0] = f"##### {first_line}"
                result['answer'] = '\n'.join(answer_lines)
            
            st.markdown(result['answer'])
            
            if result['sources']:
                st.caption(f"*Based on {len(result['sources'])} documents*")
        
        # Display sources if available (for RAG queries)
        if result['sources']:
            with st.expander(f"📚 Source Documents ({len(result['sources'])})"):
                for i, source in enumerate(result['sources']):
                    readable_name = format_source_name(source['source_file'])
                    similarity = source.get('similarity_score', 0)
                    
                    st.markdown(f"**{readable_name}** (Relevance: {similarity:.3f})")
                    content = source.get('text', source.get('content', ''))
                    st.text(content[:500] + "..." if len(content) > 500 else content)
                    st.markdown("---")
    
    # Footer - ALWAYS SHOWN (moved outside the if question: block)
    st.markdown("---")
    st.caption(f"Powered by Innovation Intelligence Suite (2025)")

if __name__ == "__main__":
    main()