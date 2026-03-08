"""Human-in-the-loop utilities"""
from typing import Dict, Any, Optional
from enum import Enum
from utils.logger import setup_logger

logger = setup_logger(__name__)

class HITLTrigger(Enum):
    """Reasons for triggering HITL"""
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    LOW_ASR_CONFIDENCE = "low_asr_confidence"
    PARSER_AMBIGUITY = "parser_ambiguity"
    LOW_VERIFIER_CONFIDENCE = "low_verifier_confidence"
    USER_REQUEST = "user_request"

class HITLAction(Enum):
    """Actions user can take in HITL"""
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"
    ADD_CONTEXT = "add_context"

class HITLManager:
    """Manages human-in-the-loop interventions"""
    
    def __init__(self):
        self.current_intervention: Optional[Dict[str, Any]] = None
        
    def should_trigger(
        self,
        trigger_type: HITLTrigger,
        confidence: Optional[float] = None,
        threshold: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if HITL should be triggered
        
        Args:
            trigger_type: Type of trigger
            confidence: Confidence score (if applicable)
            threshold: Threshold for confidence (if applicable)
            data: Additional data for decision
            
        Returns:
            True if HITL should be triggered
        """
        if trigger_type == HITLTrigger.USER_REQUEST:
            return True
            
        if confidence is not None and threshold is not None:
            if confidence < threshold:
                logger.info(f"HITL triggered: {trigger_type.value} (confidence: {confidence:.2f} < {threshold:.2f})")
                return True
                
        if trigger_type == HITLTrigger.PARSER_AMBIGUITY:
            if data and data.get("needs_clarification", False):
                logger.info(f"HITL triggered: {trigger_type.value}")
                return True
                
        return False
    
    def create_intervention(
        self,
        trigger_type: HITLTrigger,
        message: str,
        data: Dict[str, Any],
        suggestions: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Create an HITL intervention
        
        Args:
            trigger_type: Why HITL was triggered
            message: Message to show user
            data: Current state data
            suggestions: Optional suggestions for user
            
        Returns:
            Intervention object
        """
        intervention = {
            "trigger": trigger_type.value,
            "message": message,
            "data": data,
            "suggestions": suggestions or [],
            "resolved": False,
            "action": None,
            "user_input": None
        }
        
        self.current_intervention = intervention
        return intervention
    
    def resolve_intervention(
        self,
        action: HITLAction,
        user_input: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Resolve the current HITL intervention
        
        Args:
            action: Action taken by user
            user_input: User's input (edited text, additional context, etc.)
            
        Returns:
            Resolved intervention with user's decision
        """
        if not self.current_intervention:
            raise ValueError("No active intervention to resolve")
            
        self.current_intervention["resolved"] = True
        self.current_intervention["action"] = action.value
        self.current_intervention["user_input"] = user_input
        
        logger.info(f"HITL resolved: {action.value}")
        
        resolved = self.current_intervention.copy()
        self.current_intervention = None
        
        return resolved
    
    def get_current_intervention(self) -> Optional[Dict[str, Any]]:
        """Get the current active intervention"""
        return self.current_intervention
