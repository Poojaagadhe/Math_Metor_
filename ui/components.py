"""Reusable UI components for Streamlit app"""
import streamlit as st
from typing import Dict, Any, List

def render_agent_trace(trace_data: Dict[str, Any]):
    """
    Render agent execution trace
    
    Args:
        trace_data: Dictionary with agent execution information
    """
    st.subheader("🤖 Agent Execution Trace")
    
    agents = [
        ("Parser", trace_data.get('parser')),
        ("Router", trace_data.get('router')),
        ("Solver", trace_data.get('solver')),
        ("Verifier", trace_data.get('verifier')),
        ("Explainer", trace_data.get('explainer'))
    ]
    
    for agent_name, agent_data in agents:
        if agent_data:
            status = agent_data.get('status', 'completed')
            if status == 'completed':
                st.success(f"✓ {agent_name}: {agent_data.get('summary', 'Completed')}")
            elif status == 'running':
                st.info(f"⏳ {agent_name}: Running...")
            else:
                st.error(f"✗ {agent_name}: {agent_data.get('error', 'Failed')}")
                
            # Show details in expander
            with st.expander(f"View {agent_name} details"):
                st.json(agent_data)

def render_confidence_badge(confidence: float, label: str = "Confidence"):
    """
    Render color-coded confidence badge
    
    Args:
        confidence: Confidence score (0-1)
        label: Label for the badge
    """
    if confidence >= 0.8:
        color = "green"
        emoji = "✅"
    elif confidence >= 0.6:
        color = "orange"
        emoji = "⚠️"
    else:
        color = "red"
        emoji = "❌"
        
    st.markdown(
        f"**{label}**: :{color}[{emoji} {confidence:.1%}]"
    )

def render_retrieved_context(contexts: List[Dict[str, Any]]):
    """
    Render retrieved RAG context
    
    Args:
        contexts: List of retrieved context dictionaries
    """
    st.subheader("📚 Retrieved Knowledge")
    
    if not contexts:
        st.info("No context retrieved")
        return
        
    for i, ctx in enumerate(contexts, 1):
        with st.expander(f"Source {i}: {ctx.get('source', 'Unknown')} (Topic: {ctx.get('topic', 'Unknown')})"):
            relevance = ctx.get('relevance_score', 0)
            render_confidence_badge(relevance, "Relevance")
            st.markdown("---")
            st.markdown(ctx.get('content', ''))

def render_feedback_buttons(problem_id: str):
    """
    Render feedback collection buttons
    
    Args:
        problem_id: ID of the problem
    """
    st.subheader("💬 Feedback")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Correct", key=f"correct_{problem_id}"):
            st.session_state[f'feedback_{problem_id}'] = 'correct'
            st.success("Thanks for your feedback!")
            
    with col2:
        if st.button("❌ Incorrect", key=f"incorrect_{problem_id}"):
            st.session_state[f'feedback_{problem_id}'] = 'incorrect'
            st.warning("Thanks for letting us know. Please provide details below.")
            
    with col3:
        if st.button("🔄 Re-check", key=f"recheck_{problem_id}"):
            st.session_state[f'recheck_{problem_id}'] = True
            st.info("Re-check requested")
            
    # Comment box
    if st.session_state.get(f'feedback_{problem_id}') == 'incorrect':
        comment = st.text_area(
            "What was wrong?",
            key=f"comment_{problem_id}",
            placeholder="Please describe the issue..."
        )
        
        if st.button("Submit Feedback", key=f"submit_{problem_id}"):
            st.session_state[f'comment_submitted_{problem_id}'] = comment
            st.success("Feedback submitted! This will help improve the system.")

def render_hitl_dialog(intervention: Dict[str, Any]):
    """
    Render HITL intervention dialog
    
    Args:
        intervention: HITL intervention object
    """
    st.warning("⚠️ Human Review Required")
    
    st.info(intervention.get('message', 'Please review the following'))
    
    # Show current data
    data = intervention.get('data', {})
    
    if 'extracted_text' in data:
        st.subheader("Extracted Text")
        corrected_text = st.text_area(
            "Review and correct if needed:",
            value=data.get('extracted_text', ''),
            height=150,
            key="hitl_text_correction"
        )
        
    # Show suggestions
    suggestions = intervention.get('suggestions', [])
    if suggestions:
        st.subheader("Suggestions")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
            
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Approve", key="hitl_approve"):
            st.session_state['hitl_action'] = 'approve'
            st.session_state['hitl_input'] = None
            
    with col2:
        if st.button("✏️ Edit & Continue", key="hitl_edit"):
            st.session_state['hitl_action'] = 'edit'
            st.session_state['hitl_input'] = corrected_text if 'extracted_text' in data else None
            
    with col3:
        if st.button("🔄 Reject & Restart", key="hitl_reject"):
            st.session_state['hitl_action'] = 'reject'
            st.session_state['hitl_input'] = None

def render_similar_problems(similar_problems: List[Dict[str, Any]]):
    """
    Render similar problems from memory
    
    Args:
        similar_problems: List of similar problem dictionaries
    """
    if not similar_problems:
        return
        
    st.sidebar.subheader("🔍 Similar Problems Solved")
    
    for i, problem in enumerate(similar_problems, 1):
        with st.sidebar.expander(f"Problem {i}"):
            st.markdown(f"**Topic**: {problem.get('topic', 'Unknown')}")
            st.markdown(f"**Confidence**: {problem.get('verifier_confidence', 0):.1%}")
            st.markdown(f"**Feedback**: {problem.get('user_feedback', 'None')}")
            
            # Show brief solution
            solution = problem.get('solution', '')
            if solution:
                st.markdown(f"**Solution**: {solution[:100]}...")

def render_input_preview(
    text: str,
    confidence: float,
    source_type: str,
    editable: bool = True
):
    """
    Render input preview with edit capability
    
    Args:
        text: Extracted/transcribed text
        confidence: Confidence score
        source_type: Type of input (image/audio/text)
        editable: Whether text can be edited
        
    Returns:
        Edited text if editable, otherwise original text
    """
    st.subheader(f"📝 {source_type.title()} Preview")
    
    render_confidence_badge(confidence, "Extraction Confidence")
    
    if editable:
        edited_text = st.text_area(
            "Review and edit if needed:",
            value=text,
            height=150,
            key=f"preview_{source_type}"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            confirmed = st.button("✓ Confirm", key=f"confirm_{source_type}")
        with col2:
            reset = st.button("↺ Reset", key=f"reset_{source_type}")
            
        if reset:
            return text
            
        return edited_text
    else:
        st.text_area(
            "Extracted text:",
            value=text,
            height=150,
            disabled=True,
            key=f"preview_readonly_{source_type}"
        )
        return text
