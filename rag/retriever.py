"""Retriever for RAG pipeline"""
from typing import List, Dict, Any, Optional
from rag.vector_store import VectorStore
from utils.logger import setup_logger

logger = setup_logger(__name__)

class Retriever:
    """Retrieves relevant context for problem solving"""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize retriever
        
        Args:
            vector_store: VectorStore instance (creates new one if None)
        """
        self.vector_store = vector_store or VectorStore()
        logger.info("Retriever initialized")
        
    def retrieve(
        self,
        query: str,
        topic: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: Query text (problem description)
            topic: Optional topic filter (algebra, calculus, probability, linear_algebra)
            n_results: Number of results to retrieve
            
        Returns:
            List of retrieved contexts with metadata
        """
        logger.info(f"Retrieving context for query (topic: {topic})")
        
        # Build metadata filter
        filter_metadata = None
        if topic:
            filter_metadata = {"topic": topic}
            
        # Query vector store
        results = self.vector_store.query(
            query_text=query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )
        
        # Format results
        contexts = []
        for i in range(len(results['documents'])):
            contexts.append({
                "content": results['documents'][i],
                "metadata": results['metadatas'][i],
                "relevance_score": 1.0 - results['distances'][i],  # Convert distance to similarity
                "source": results['metadatas'][i].get('source', 'unknown'),
                "topic": results['metadatas'][i].get('topic', 'unknown')
            })
            
        logger.info(f"Retrieved {len(contexts)} contexts")
        
        return contexts
        
    def format_context_for_prompt(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Format retrieved contexts for inclusion in LLM prompt
        
        Args:
            contexts: List of retrieved contexts
            
        Returns:
            Formatted context string
        """
        if not contexts:
            return "No relevant context found."
            
        formatted = "# Retrieved Context\n\n"
        
        for i, ctx in enumerate(contexts, 1):
            formatted += f"## Source {i}: {ctx['source']} (Topic: {ctx['topic']})\n"
            formatted += f"Relevance: {ctx['relevance_score']:.2f}\n\n"
            formatted += ctx['content'] + "\n\n"
            formatted += "---\n\n"
            
        return formatted
