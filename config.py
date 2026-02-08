import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration for Cybersecurity RAG System"""
    
    # Flask settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    
    # Security settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'cybersecurity-rag-secret-key-change-in-production')
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # File paths
    DATA_DIR = 'data'
    DOCUMENTS_DIR = os.path.join(DATA_DIR, 'documents')
    VECTOR_STORE_DIR = os.path.join(DATA_DIR, 'vector_store')
    LOGS_DIR = 'logs'
    
    # Document processing
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 50))
    
    # Embeddings
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDING_DIMENSION = 384  # For all-MiniLM-L6-v2
    
    # Vector search
    VECTOR_STORE_TYPE = os.getenv('VECTOR_STORE_TYPE', 'faiss')  # faiss, chroma, or qdrant
    SIMILARITY_TOP_K = int(os.getenv('SIMILARITY_TOP_K', 5))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.7))
    
    # LLM settings (for OpenAI or local)
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')  # openai, anthropic, local
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
    ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307')
    LOCAL_MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', '')
    
    # RAG settings
    MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', 4000))
    ENABLE_HYBRID_SEARCH = os.getenv('ENABLE_HYBRID_SEARCH', 'True').lower() == 'true'
    ENABLE_RERANKING = os.getenv('ENABLE_RERANKING', 'True').lower() == 'true'
    
    # Security specific
    ALLOWED_FILE_EXTENSIONS = {'.pdf', '.txt', '.md', '.json', '.csv', '.log', '.xml', '.yaml', '.yml'}
    MAX_DOCUMENTS_PER_REQUEST = int(os.getenv('MAX_DOCUMENTS_PER_REQUEST', 10))
    
    # Monitoring
    ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'True').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')