"""Configuration loader for Math Mentor"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Load local .env (ignored on Streamlit Cloud where secrets come from dashboard)
load_dotenv()

def _get(key: str, default: str = "") -> str:
    """Read a config value from Streamlit secrets (cloud) or os.getenv (local)."""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


class Config:
    """Application configuration"""
    
    # API Keys
    OPENAI_API_KEY = _get("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY", "")
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
    UPLOADS_DIR = DATA_DIR / "uploads"
    MEMORY_DB_DIR = DATA_DIR / "memory_db"
    
    # Vector Store
    CHROMA_PERSIST_DIR = _get("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db"))
    
    # Memory
    MEMORY_DB_PATH = _get("MEMORY_DB_PATH", str(DATA_DIR / "memory.db"))
    
    # OCR Settings
    OCR_CONFIDENCE_THRESHOLD = float(_get("OCR_CONFIDENCE_THRESHOLD", "0.7"))
    OCR_LANGUAGES = _get("OCR_LANGUAGES", "en").split(",")
    
    # ASR Settings
    WHISPER_MODEL = _get("WHISPER_MODEL", "base")
    
    # Verifier Settings
    VERIFIER_CONFIDENCE_THRESHOLD = float(_get("VERIFIER_CONFIDENCE_THRESHOLD", "0.8"))
    
    # Application Settings
    DEBUG = _get("DEBUG", "False").lower() == "true"
    LOG_LEVEL = _get("LOG_LEVEL", "INFO")
    MAX_UPLOAD_SIZE_MB = int(_get("MAX_UPLOAD_SIZE_MB", "10"))
    
    # Agent Settings
    MAX_RETRIES = int(_get("MAX_RETRIES", "3"))
    AGENT_TEMPERATURE = float(_get("AGENT_TEMPERATURE", "0.3"))
    
    # LLM Provider Settings
    LLM_PROVIDER = _get("LLM_PROVIDER", "groq")
    
    # Google Gemini Settings
    GEMINI_API_KEY = _get("GEMINI_API_KEY", "")
    GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-pro")
    
    # Groq Settings
    GROQ_API_KEY = _get("GROQ_API_KEY", "")
    GROQ_MODEL = _get("GROQ_MODEL", "llama-3.1-8b-instant")
    
    # Hugging Face Settings
    HUGGINGFACE_API_KEY = _get("HUGGINGFACE_API_KEY", "")
    HUGGINGFACE_MODEL = _get("HUGGINGFACE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    
    # Ollama Settings
    OLLAMA_MODEL = _get("OLLAMA_MODEL", "llama3.1")
    OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # LLM Models (use appropriate model based on provider)
    if LLM_PROVIDER == "gemini":
        _default_model = GEMINI_MODEL
        _advanced_model = GEMINI_MODEL
    elif LLM_PROVIDER == "groq":
        _default_model = GROQ_MODEL
        _advanced_model = GROQ_MODEL
    elif LLM_PROVIDER == "huggingface":
        _default_model = HUGGINGFACE_MODEL
        _advanced_model = HUGGINGFACE_MODEL
    elif LLM_PROVIDER == "ollama":
        _default_model = OLLAMA_MODEL
        _advanced_model = OLLAMA_MODEL
    else:
        _default_model = "gpt-3.5-turbo"
        _advanced_model = "gpt-4"
    
    PARSER_MODEL = _default_model
    ROUTER_MODEL = _default_model
    SOLVER_MODEL = _advanced_model
    VERIFIER_MODEL = _advanced_model
    EXPLAINER_MODEL = _default_model
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.UPLOADS_DIR.mkdir(exist_ok=True)
        cls.MEMORY_DB_DIR.mkdir(exist_ok=True)
        Path(cls.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

# Initialize directories on import
Config.ensure_directories()
