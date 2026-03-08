"""Text input processor"""
from typing import Dict, Any
from utils.logger import setup_logger

logger = setup_logger(__name__)

class TextProcessor:
    """Processes direct text inputs"""
    
    def __init__(self):
        """Initialize text processor"""
        logger.info("TextProcessor initialized")
        
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text input
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing:
                - text: Cleaned and normalized text
                - confidence: Always 1.0 for direct text input
                - hitl_required: Always False for text input
        """
        logger.info("Processing text input...")
        
        # Basic cleaning
        cleaned_text = text.strip()
        
        # Normalize whitespace
        cleaned_text = " ".join(cleaned_text.split())
        
        logger.info(f"Text processed. Length: {len(cleaned_text)} characters")
        
        return {
            "text": cleaned_text,
            "confidence": 1.0,
            "hitl_required": False,
            "hitl_intervention": None
        }
    
    def validate_text(self, text: str) -> bool:
        """
        Validate that text is not empty and contains meaningful content
        
        Args:
            text: Input text
            
        Returns:
            True if valid, False otherwise
        """
        if not text or not text.strip():
            return False
            
        # Check minimum length
        if len(text.strip()) < 3:
            return False
            
        return True
