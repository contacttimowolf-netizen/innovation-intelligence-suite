# innovation-intelligence-suite

## Overview 
This capstone project explores how AI/ML can accelerate insight generation for R&D teams:
- A **Retrieval-Augmented Generation (RAG)** interface synthesizing information and answering domain-scoped innovation queries. 
- A **Technology Adoption Readiness Predictor** that classifies emerging technologies by maturity adding an intelligence layer to RAG by feeding tags into a prediction model. 
- **Automotive AI technology focus** with research papers, tech reports, startups and patent data (2025).

Developed as a Data Science + AI Bootcamp capstone project (2025).

## Features 
- Query and summarize automotive technology documents (30,000+ chunks)
- Retrieve insights on specific innovation topics from research papers, tech reports, startups and patent data (2025) 
- Source attribution with relevance scoring
- Template-based answer generation with full transparency
- Semi-supervised TRL classifier (Fraunhofer ISI-inspired).

## Quick Start

### 1. Clone Repository
git clone https://github.com/<yourname>/innovation-intelligence-suite.git
cd innovation-intelligence-suite

### 2. Set Up Environment
python -m venv .venv

# Mac/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Configure API Keys
Copy .env.example to .env
Add your Groq API key: GROQ_API_KEY=your_key_here
Get free API key at: https://console.groq.com

### 5. Run the System

#### CLI Interface (Testing & Development)
jupyter notebook
# Open and run 03_notebooks/

#### Web Interface (Demo)
cd 05_app
streamlit run streamlit_app.py

## Project Structure
innovation-intelligence-suite/
├── 01_data/                 # Data directories
│   └── rag_automotive_tech/
│       ├── raw_sources/     # Source documents
│       └── processed/       # Processed chunks
├── 03_notebooks/            # Development & analysis
│   ├── 01_processing_rag.ipynb    # Document processing
│   ├── 02_retrieval_rag.ipynb     # Retrieval development  
│   ├── 03_generator_rag.ipynb     # Answer generation
│   ├── 04_interface_rag.ipynb     # CLI & interfaces
│   └── rag_components/      # Reusable code modules
├── 04_models/               # Vector store & embeddings
├── 05_app/                  # Streamlit web application
├── 06_doc/                  # Documentation & architecture
├── requirements.txt         # Python dependencies
└── README.md               # This file

## Demo Queries
Try these in the CLI or web app:
- "Tell me about automotive startups in AI" 🚀
- "What are the challenges in autonomous driving?" 📄  
- "How is AI transforming the automotive industry?" 🔍

## Stack 
- **RAG Pipeline**: LangChain - FAISS - TF-IDF Vectorization
- **Answer Generation**: Template-based with source attribution
- **Interface**: Streamlit & Jupyter notebooks
- **Language**: Python

## What's NOT to push
- Raw PDFs if under copyright (only metadata or short excerpts)
- API keys (use environment variables only)  
- Large model/vector files (>100 MB)
- Processed chunks (can be regenerated)

## Optional/Additional Enhancements
- **MIT License** file added
- Include a **demo video link** or **GIF** in the README
- Add a **Streamlit Cloud or HuggingFace Space** link for live demo
- **TRL Classifier** - Technology Readiness Level prediction (future enhancement)

---
## Development
For development and customization:
- Use 02_retrieval_rag.ipynb for retrieval system improvements
- Use 03_generator_rag.ipynb for answer generation enhancements  
- All notebooks use modular components from rag_components/