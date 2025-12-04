# rag_components/__init__.py

from .faiss_retriever import FAISSRetriever
from .query_interface import QueryInterface
from .answer_generator import AnswerGenerator
from .query_expander import QueryExpander

# For backward compatibility
try:
    from .retriever import Retriever
except ImportError:
    Retriever = None  # Old retriever not available

__all__ = [
    'FAISSRetriever',
    'Retriever',  # Keep for compatibility
    'QueryInterface',
    'AnswerGenerator',
    'QueryExpander'
]