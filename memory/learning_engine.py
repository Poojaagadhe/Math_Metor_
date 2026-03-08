"""Learning engine for pattern reuse and improvement"""
from typing import Dict, Any, List, Optional
from memory.memory_store import MemoryStore
from utils.logger import setup_logger

logger = setup_logger(__name__)

class LearningEngine:
    """Learns from past interactions to improve future performance"""
    
    def __init__(self, memory_store: Optional[MemoryStore] = None):
        """
        Initialize learning engine
        
        Args:
            memory_store: MemoryStore instance (creates new one if None)
        """
        self.memory_store = memory_store or MemoryStore()
        logger.info("LearningEngine initialized")
        
    def find_similar_solved_problems(
        self,
        topic: str,
        subtopic: Optional[str] = None,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find similar problems that were successfully solved
        
        Args:
            topic: Problem topic
            subtopic: Optional subtopic
            limit: Maximum number of results
            
        Returns:
            List of similar solved problems
        """
        logger.info(f"Finding similar problems for topic: {topic}")
        
        similar = self.memory_store.get_similar_problems(topic, limit=limit * 2)
        
        # Filter for successful solutions (high verifier confidence or positive feedback)
        successful = []
        for problem in similar:
            confidence = problem.get('verifier_confidence', 0)
            feedback = problem.get('user_feedback')
            
            if confidence >= 0.8 or feedback == 'correct':
                successful.append(problem)
                
            if len(successful) >= limit:
                break
                
        logger.info(f"Found {len(successful)} successful similar problems")
        
        return successful
        
    def get_solution_patterns(
        self,
        topic: str
    ) -> List[str]:
        """
        Get common solution patterns for a topic
        
        Args:
            topic: Problem topic
            
        Returns:
            List of solution pattern descriptions
        """
        similar = self.find_similar_solved_problems(topic, limit=5)
        
        patterns = []
        for problem in similar:
            # Extract solution approach from routing info
            routing = problem.get('routing_info')
            if routing:
                import json
                try:
                    routing_dict = json.loads(routing)
                    strategy = routing_dict.get('solution_strategy')
                    if strategy:
                        patterns.append(strategy)
                except:
                    pass
                    
        return list(set(patterns))  # Return unique patterns
        
    def get_ocr_corrections(self) -> Dict[str, str]:
        """
        Get common OCR correction patterns
        
        Returns:
            Dictionary of common OCR errors and corrections
        """
        # This would be populated from HITL corrections over time
        # For now, return common math OCR issues
        return {
            "x2": "x²",
            "x3": "x³",
            "sqrt": "√",
            "pi": "π",
            "theta": "θ",
            "alpha": "α",
            "beta": "β",
            "<=": "≤",
            ">=": "≥",
            "!=": "≠"
        }
        
    def apply_ocr_corrections(self, text: str) -> str:
        """
        Apply learned OCR corrections to text
        
        Args:
            text: Text to correct
            
        Returns:
            Corrected text
        """
        corrections = self.get_ocr_corrections()
        corrected = text
        
        for error, correction in corrections.items():
            corrected = corrected.replace(error, correction)
            
        return corrected
        
    def get_learning_insights(self) -> Dict[str, Any]:
        """
        Get insights from learning data
        
        Returns:
            Dictionary of insights
        """
        stats = self.memory_store.get_stats()
        
        insights = {
            "total_problems_solved": stats.get('total_problems', 0),
            "topics_covered": list(stats.get('by_topic', {}).keys()),
            "success_rate": self._calculate_success_rate(stats),
            "most_common_topic": self._get_most_common_topic(stats)
        }
        
        return insights
        
    def _calculate_success_rate(self, stats: Dict[str, Any]) -> float:
        """Calculate success rate from feedback"""
        feedback = stats.get('feedback_stats', {})
        correct = feedback.get('correct', 0)
        incorrect = feedback.get('incorrect', 0)
        total = correct + incorrect
        
        if total == 0:
            return 0.0
            
        return correct / total
        
    def _get_most_common_topic(self, stats: Dict[str, Any]) -> Optional[str]:
        """Get most common topic"""
        by_topic = stats.get('by_topic', {})
        
        if not by_topic:
            return None
            
        return max(by_topic.items(), key=lambda x: x[1])[0]
