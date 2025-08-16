import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict

# Import shared resources and helpers
from ..extensions import embeddings
from .db_manager import DatabaseManager
from ..utils.helpers import cosine_similarity

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manages the creation, storage, and searching of vector embeddings.
    """
    
    @staticmethod
    def create_embedding(text: str) -> List[float]:
        """Creates a vector embedding for a given piece of text."""
        try:
            return embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            return []
    
    @staticmethod
    def store_embedding(entity_type: str, entity_id: str, vector: List[float]):
        """Stores a vector embedding in the database, linked to an entity."""
        if not vector:
            logger.warning(f"Attempted to store an empty vector for {entity_type}:{entity_id}")
            return

        query = """
        INSERT INTO embeddings (id, entity_type, entity_id, vector, created_at)
        VALUES (:id, :entity_type, :entity_id, :vector, :created_at)
        ON DUPLICATE KEY UPDATE vector = :vector, created_at = :created_at
        """
        params = {
            'id': str(uuid.uuid4()),
            'entity_type': entity_type,
            'entity_id': entity_id,
            'vector': json.dumps(vector), # Store vector as a JSON string
            'created_at': datetime.now()
        }
        DatabaseManager.execute_query(query, params)
    
    @staticmethod
    def search_similar(query_text: str, entity_type: str, limit: int = 10) -> List[Dict]:
        """
        Finds entities similar to a query text using cosine similarity on embeddings.
        This is a brute-force search performed in application memory.
        For very large datasets, a dedicated vector database (e.g., Pinecone, Weaviate)
        would be more efficient.
        """
        query_vector = EmbeddingManager.create_embedding(query_text)
        if not query_vector:
            return []
        
        # Fetch all vectors of the specified type from the database
        all_embeddings = DatabaseManager.execute_query(
            "SELECT entity_id, vector FROM embeddings WHERE entity_type = :entity_type", 
            {'entity_type': entity_type}
        )
        
        # Calculate similarity for each item
        similarities = []
        for item in all_embeddings:
            try:
                stored_vector = json.loads(item['vector'])
                similarity = cosine_similarity(query_vector, stored_vector)
                similarities.append({
                    'entity_id': item['entity_id'],
                    'similarity': similarity
                })
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Could not process vector for entity {item['entity_id']}: {e}")
                continue
        
        # Sort by similarity in descending order and return the top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]
