"""Memory store for persisting solved problems and feedback"""
import sqlite3
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MemoryStore:
    """Persistent storage for solved problems and user feedback"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize memory store
        
        Args:
            db_path: Path to SQLite database (uses Config default if None)
        """
        self.db_path = db_path or Config.MEMORY_DB_PATH
        self._init_database()
        logger.info(f"MemoryStore initialized at {self.db_path}")
        
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create problems table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS problems (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                input_type TEXT NOT NULL,
                raw_input_path TEXT,
                extracted_text TEXT,
                parsed_problem TEXT,
                topic TEXT,
                subtopic TEXT,
                routing_info TEXT,
                retrieved_context TEXT,
                solution TEXT,
                explanation TEXT,
                verifier_confidence REAL,
                user_feedback TEXT,
                user_comment TEXT,
                hitl_triggered INTEGER,
                hitl_corrections TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Create index on topic for faster retrieval
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_topic ON problems(topic)
        ''')
        
        # Create index on timestamp
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON problems(timestamp)
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Database schema initialized")
        
    def store_problem(self, problem_data: Dict[str, Any]) -> str:
        """
        Store a solved problem
        
        Args:
            problem_data: Dictionary containing all problem data
            
        Returns:
            Problem ID
        """
        import uuid
        
        problem_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO problems (
                id, timestamp, input_type, raw_input_path, extracted_text,
                parsed_problem, topic, subtopic, routing_info, retrieved_context,
                solution, explanation, verifier_confidence, user_feedback,
                user_comment, hitl_triggered, hitl_corrections, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            problem_id,
            timestamp,
            problem_data.get('input_type', 'text'),
            problem_data.get('raw_input_path'),
            problem_data.get('extracted_text'),
            json.dumps(problem_data.get('parsed_problem', {})),
            problem_data.get('topic'),
            problem_data.get('subtopic'),
            json.dumps(problem_data.get('routing_info', {})),
            json.dumps(problem_data.get('retrieved_context', [])),
            problem_data.get('solution'),
            problem_data.get('explanation'),
            problem_data.get('verifier_confidence'),
            problem_data.get('user_feedback'),
            problem_data.get('user_comment'),
            1 if problem_data.get('hitl_triggered') else 0,
            json.dumps(problem_data.get('hitl_corrections')),
            timestamp
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored problem {problem_id}")
        
        return problem_id
        
    def update_feedback(
        self,
        problem_id: str,
        feedback: str,
        comment: Optional[str] = None
    ):
        """
        Update user feedback for a problem
        
        Args:
            problem_id: Problem ID
            feedback: Feedback (correct/incorrect)
            comment: Optional comment
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE problems
            SET user_feedback = ?, user_comment = ?
            WHERE id = ?
        ''', (feedback, comment, problem_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated feedback for problem {problem_id}: {feedback}")
        
    def get_similar_problems(
        self,
        topic: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get similar problems by topic
        
        Args:
            topic: Topic to search for
            limit: Maximum number of results
            
        Returns:
            List of similar problems
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM problems
            WHERE topic = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (topic, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        problems = []
        for row in rows:
            problems.append(dict(row))
            
        logger.info(f"Found {len(problems)} similar problems for topic: {topic}")
        
        return problems
        
    def get_problem_by_id(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """Get a problem by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM problems WHERE id = ?', (problem_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
        
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total problems
        cursor.execute('SELECT COUNT(*) FROM problems')
        total = cursor.fetchone()[0]
        
        # By topic
        cursor.execute('''
            SELECT topic, COUNT(*) as count
            FROM problems
            GROUP BY topic
        ''')
        by_topic = dict(cursor.fetchall())
        
        # Feedback stats
        cursor.execute('''
            SELECT user_feedback, COUNT(*) as count
            FROM problems
            WHERE user_feedback IS NOT NULL
            GROUP BY user_feedback
        ''')
        feedback_stats = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_problems": total,
            "by_topic": by_topic,
            "feedback_stats": feedback_stats
        }
