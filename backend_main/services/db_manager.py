import logging
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Any

# Import the session maker from the extensions module
from ..extensions import SessionLocal

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Handles all database interactions.
    Uses a session-based approach for connection management.
    """
    
    @staticmethod
    def get_db():
        """Provides a database session."""
        db = SessionLocal()
        try:
            return db
        except Exception as e:
            db.close()
            raise e
    
    @staticmethod
    def close_db(db):
        """Closes the database session."""
        if db:
            db.close()
    
    @staticmethod
    def execute_query(query: str, params: Dict = None) -> List[Dict[str, Any]]:
        """
        Executes a SQL query with parameters and returns the results.
        Handles both read (SELECT) and write (INSERT, UPDATE) operations.
        """
        db = DatabaseManager.get_db()
        try:
            result_proxy = db.execute(text(query), params or {})
            
            # For SELECT statements, fetch and return rows
            if result_proxy.returns_rows:
                columns = result_proxy.keys()
                return [dict(zip(columns, row)) for row in result_proxy.fetchall()]
            # For INSERT, UPDATE, DELETE, commit the transaction
            else:
                db.commit()
                return []
        except SQLAlchemyError as e:
            # Rollback the transaction in case of an error
            db.rollback()
            logger.error(f"Database query failed: {e}")
            raise  # Re-raise the exception to be handled by the caller
        finally:
            DatabaseManager.close_db(db)
