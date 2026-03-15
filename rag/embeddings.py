"""Embeddings generation for RAG pipeline"""
from typing import List, Union
import openai
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingGenerator:
    """Generates embeddings for text using OpenAI or sentence-transformers"""
    
    def __init__(self, use_openai: bool = None):
        """
        Initialize embedding generator
        
        Args:
            use_openai: If True, use OpenAI embeddings. If False, use sentence-transformers.
                       If None, auto-detect based on API key availability.
        """
        # Auto-detect if not specified
        if use_openai is None:
            use_openai = bool(Config.OPENAI_API_KEY)
        
        self.use_openai = use_openai
        
        if use_openai:
            if not Config.OPENAI_API_KEY:
                logger.warning("OpenAI API key not found, falling back to sentence-transformers")
                self.use_openai = False
            else:
                openai.api_key = Config.OPENAI_API_KEY
                self.model = "text-embedding-3-small"
                logger.info(f"EmbeddingGenerator initialized with OpenAI model: {self.model}")
        
        if not self.use_openai:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("EmbeddingGenerator initialized with sentence-transformers")
            
    def generate_embeddings(
        self,
        texts: Union[str, List[str]]
    ) -> List[List[float]]:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if self.use_openai:
            return self._generate_openai_embeddings(texts)
        else:
            return self._generate_local_embeddings(texts)
            
    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        logger.info(f"Generating OpenAI embeddings for {len(texts)} texts...")
        
        response = openai.embeddings.create(
            model=self.model,
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        return embeddings
        
    def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local sentence-transformers"""
        logger.info(f"Generating local embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts)
        embeddings_list = embeddings.tolist()
        
        logger.info(f"Generated {len(embeddings_list)} embeddings")
        
        return embeddings_list
