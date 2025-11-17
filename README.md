# innovation-intelligence-suite

## Overview 
This capstone project explores how AI/ML can accelerate insight generation for R&D teams:
- A **Retrieval-Augmented Generation (RAG)** interface synthesizing information and answering domain-scoped innovation queries. 
- A **Technology Adoption Readiness Predictor** that classifies emerging technologies by maturity adding an intelligence layer to RAG by feeding tags into a prediction model. 

Developed as a Data Science + AI Bootcamp capstone project (2025).

## Features 
- Query and summarize open datasets, reports, and research papers.
- Retrieve pain points or challenges for specific innovation topics.
- Semi-supervised TRL classifier (Fraunhofer ISI-inspired).

## Stack 
LangChain - FAISS - Sentence Transformers - Llama 3 (via Groq API) - Python - Streamlit

## Setup 
1. Clone repo 
   ```bash 
   git clone https://github.com/<yourname>/innovation-intelligence-suite.git

2. Install dependencies 
   ```bash 
   pip install -r requirements.txt

3. Add your Groq API key as environment variable 
   ```bash
   export GROQ_API_KEY="your_key"

4. Run prototype 
   ```bash 
   python src/03_query_app.py

---

## **What's NOT to push**
- Raw PDFs if under copyright (only metadata or short excerpts).  
- API keys (`.env` or `os.environ` only).  
- Large embeddings/index files (>100 MB).  

Use `.gitignore`:
data/*
vector_index/*
*.env
*.key


---

## **Optional/Additional Enhancements**
- **MIT License** file added. 
- Include a **demo video link** or **GIF** in the README.  
- Add a **Streamlit Cloud or HuggingFace Space** link for live demo.  


