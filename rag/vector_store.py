"""Vector store for RAG pipeline using ChromaDB"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from rag.embeddings import EmbeddingGenerator
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorStore:
    """Manages vector storage and retrieval using ChromaDB"""
    
    def __init__(self, collection_name: str = "math_knowledge"):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Math knowledge base for RAG"}
        )
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(use_openai=True)
        
        logger.info(f"VectorStore initialized. Collection: {collection_name}")
        
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dicts for each document
            ids: Optional list of document IDs (auto-generated if not provided)
        """
        if not documents:
            logger.warning("No documents to add")
            return
            
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
            
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(documents)
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully added {len(documents)} documents")
        
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"topic": "calculus"})
            
        Returns:
            Dictionary containing:
                - documents: List of retrieved document texts
                - metadatas: List of metadata for each document
                - distances: List of similarity distances
                - ids: List of document IDs
        """
        logger.info(f"Querying vector store: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings(query_text)[0]
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        
        logger.info(f"Retrieved {len(results['documents'][0])} results")
        
        return {
            "documents": results['documents'][0],
            "metadatas": results['metadatas'][0],
            "distances": results['distances'][0],
            "ids": results['ids'][0]
        }
        
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": Config.CHROMA_PERSIST_DIR
        }
        
    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        logger.warning(f"Clearing collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Math knowledge base for RAG"}
        )
        logger.info("Collection cleared")
        
    def load_knowledge_base(self, knowledge_base_dir: Optional[Path] = None) -> None:
        """
        Load documents from knowledge base directory
        
        Args:
            knowledge_base_dir: Path to knowledge base directory (uses Config default if None)
        """
        if knowledge_base_dir is None:
            knowledge_base_dir = Config.KNOWLEDGE_BASE_DIR
            
        logger.info(f"Loading knowledge base from {knowledge_base_dir}")
        
        documents = []
        metadatas = []
        ids = []
        
        # Walk through knowledge base directory
        for topic_dir in knowledge_base_dir.iterdir():
            if not topic_dir.is_dir():
                continue
                
            topic = topic_dir.name
            logger.info(f"Processing topic: {topic}")
            
            for doc_file in topic_dir.glob("*.md"):
                # Read document
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Chunk document
                chunks = self._chunk_document(content)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({
                        "topic": topic,
                        "source": doc_file.name,
                        "chunk_id": i
                    })
                    ids.append(f"{topic}_{doc_file.stem}_{i}")
                    
        if documents:
            logger.info(f"Adding {len(documents)} chunks from knowledge base")
            self.add_documents(documents, metadatas, ids)
        else:
            logger.warning("No documents found in knowledge base")
            
    def _chunk_document(self, content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Chunk document into smaller pieces
        
        Args:
            content: Document content
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Simple chunking by paragraphs and size
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks


# CLI for initializing/managing vector store
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Math Mentor vector store")
    parser.add_argument("--init", action="store_true", help="Initialize vector store with knowledge base")
    parser.add_argument("--reindex", action="store_true", help="Clear and reindex knowledge base")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    
    args = parser.parse_args()
    
    store = VectorStore()
    
    if args.init or args.reindex:
        if args.reindex:
            store.clear_collection()
        store.load_knowledge_base()
        print("Knowledge base loaded successfully")
        
    if args.stats:
        stats = store.get_collection_stats()
        print(json.dumps(stats, indent=2))
