"""
Flask Backend for API Discovery and Community Platform
Features: MySQL connectivity, LangChain integration, semantic search, API management
"""

import os
import uuid
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import pymysql

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Additional imports for API monitoring and web scraping
import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
from config import config

# Initialize Flask app
app = Flask(__name__)
config_name = os.getenv('ENVIRONMENT', 'development')
app.config.from_object(config[config_name])

# CORS configuration
CORS(app, origins=["http://localhost:3000", "http://localhost:5000"])

# JWT Manager
jwt = JWTManager(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database engine using config
DATABASE_URL = app.config['SQLALCHEMY_DATABASE_URI']
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# LangChain setup
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
llm = ChatOpenAI(temperature=0.7, openai_api_key=os.getenv('OPENAI_API_KEY'))

# Global vector store (in production, use Redis or persistent storage)
vector_store = None

class DatabaseManager:
    """Database connection and query management"""
    
    @staticmethod
    def get_db():
        """Get database session"""
        db = SessionLocal()
        try:
            return db
        except Exception as e:
            db.close()
            raise e
    
    @staticmethod
    def close_db(db):
        """Close database session"""
        if db:
            db.close()
    
    @staticmethod
    def execute_query(query: str, params: Dict = None) -> List[Dict]:
        """Execute a query and return results"""
        db = DatabaseManager.get_db()
        try:
            result = db.execute(text(query), params or {})
            if result.returns_rows:
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
            else:
                db.commit()
                return []
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Database error: {e}")
            raise e
        finally:
            DatabaseManager.close_db(db)

class EmbeddingManager:
    """Manage embeddings and vector search"""
    
    @staticmethod
    def create_embedding(text: str) -> List[float]:
        """Create embedding for text"""
        try:
            return embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []
    
    @staticmethod
    def store_embedding(entity_type: str, entity_id: str, vector: List[float]):
        """Store embedding in database"""
        query = """
        INSERT INTO embeddings (id, entity_type, entity_id, vector, created_at)
        VALUES (:id, :entity_type, :entity_id, :vector, :created_at)
        ON DUPLICATE KEY UPDATE vector = :vector
        """
        params = {
            'id': str(uuid.uuid4()),
            'entity_type': entity_type,
            'entity_id': entity_id,
            'vector': json.dumps(vector),
            'created_at': datetime.now()
        }
        DatabaseManager.execute_query(query, params)
    
    @staticmethod
    def search_similar(query_text: str, entity_type: str, limit: int = 10) -> List[Dict]:
        """Search for similar entities using embeddings"""
        query_vector = EmbeddingManager.create_embedding(query_text)
        if not query_vector:
            return []
        
        # This is a simplified similarity search
        # In production, use a proper vector database like Pinecone or Weaviate
        embeddings_query = """
        SELECT entity_id, vector FROM embeddings 
        WHERE entity_type = :entity_type
        """
        embeddings_data = DatabaseManager.execute_query(
            embeddings_query, 
            {'entity_type': entity_type}
        )
        
        # Calculate cosine similarity (simplified)
        similarities = []
        for item in embeddings_data:
            stored_vector = json.loads(item['vector'])
            similarity = cosine_similarity(query_vector, stored_vector)
            similarities.append({
                'entity_id': item['entity_id'],
                'similarity': similarity
            })
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    import numpy as np
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class APIMonitor:
    """Monitor API changes and versions"""
    
    @staticmethod
    def fetch_api_spec(spec_url: str) -> Dict:
        """Fetch API specification from URL"""
        try:
            response = requests.get(spec_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching API spec from {spec_url}: {e}")
            return {}
    
    @staticmethod
    def compare_api_versions(old_spec: Dict, new_spec: Dict) -> List[Dict]:
        """Compare two API specifications and return changes"""
        changes = []
        
        # Compare paths (simplified)
        old_paths = set(old_spec.get('paths', {}).keys())
        new_paths = set(new_spec.get('paths', {}).keys())
        
        # Added endpoints
        for path in new_paths - old_paths:
            changes.append({
                'type': 'added',
                'endpoint': path,
                'description': f"New endpoint added: {path}"
            })
        
        # Removed endpoints (deprecated)
        for path in old_paths - new_paths:
            changes.append({
                'type': 'deprecated',
                'endpoint': path,
                'description': f"Endpoint deprecated: {path}"
            })
        
        return changes
    
    @staticmethod
    async def monitor_apis():
        """Monitor all APIs for changes (background task)"""
        query = "SELECT id, spec_url, last_known_version FROM apis WHERE spec_url IS NOT NULL"
        apis = DatabaseManager.execute_query(query)
        
        for api in apis:
            try:
                new_spec = APIMonitor.fetch_api_spec(api['spec_url'])
                if new_spec:
                    # Store new version and detect changes
                    # Implementation would go here
                    pass
            except Exception as e:
                logger.error(f"Error monitoring API {api['id']}: {e}")

class LLMService:
    """LangChain LLM service for various AI tasks"""
    
    @staticmethod
    def generate_code(api_description: str, language: str = 'python') -> str:
        """Generate code snippet for API integration"""
        prompt = f"""
        Generate a {language} code snippet to integrate with the following API:
        {api_description}
        
        Include error handling and best practices.
        """
        
        try:
            response = llm.predict(prompt)
            return response
        except Exception as e:
            logger.error(f"Code generation error: {e}")
            return "# Error generating code"
    
    @staticmethod
    def analyze_sentiment(text: str) -> float:
        """Analyze sentiment of review text"""
        prompt = f"""
        Analyze the sentiment of this API review and return a score between -1 (very negative) and 1 (very positive):
        
        {text}
        
        Return only a decimal number between -1 and 1.
        """
        
        try:
            response = llm.predict(prompt)
            # Extract numeric value from response
            import re
            match = re.search(r'-?\d+\.?\d*', response)
            if match:
                return float(match.group())
            return 0.0
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return 0.0
    
    @staticmethod
    def recommend_apis(problem_description: str) -> List[str]:
        """Recommend APIs based on problem description"""
        prompt = f"""
        Based on this problem description, recommend relevant API categories or types:
        {problem_description}
        
        Return a comma-separated list of API categories.
        """
        
        try:
            response = llm.predict(prompt)
            return [cat.strip() for cat in response.split(',')]
        except Exception as e:
            logger.error(f"API recommendation error: {e}")
            return []

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration"""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not all([username, email, password]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if user exists
        existing_user = DatabaseManager.execute_query(
            "SELECT id FROM users WHERE username = :username OR email = :email",
            {'username': username, 'email': email}
        )
        
        if existing_user:
            return jsonify({'error': 'User already exists'}), 409
        
        # Create new user
        user_id = str(uuid.uuid4())
        password_hash = generate_password_hash(password)
        
        DatabaseManager.execute_query(
            """INSERT INTO users (id, username, email, password_hash, role, created_at)
               VALUES (:id, :username, :email, :password_hash, :role, :created_at)""",
            {
                'id': user_id,
                'username': username,
                'email': email,
                'password_hash': password_hash,
                'role': 'user',
                'created_at': datetime.now()
            }
        )
        
        access_token = create_access_token(identity=user_id)
        return jsonify({
            'access_token': access_token,
            'user': {'id': user_id, 'username': username, 'email': email}
        })
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not all([username, password]):
            return jsonify({'error': 'Username and password required'}), 400
        
        # Find user
        user = DatabaseManager.execute_query(
            "SELECT id, username, email, password_hash FROM users WHERE username = :username OR email = :username",
            {'username': username}
        )
        
        if not user or not check_password_hash(user[0]['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user_data = user[0]
        access_token = create_access_token(identity=user_data['id'])
        
        return jsonify({
            'access_token': access_token,
            'user': {
                'id': user_data['id'],
                'username': user_data['username'],
                'email': user_data['email']
            }
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

# API Management Routes
@app.route('/api/apis', methods=['GET'])
def get_apis():
    """Get list of APIs with optional search"""
    try:
        search_query = request.args.get('search', '')
        category = request.args.get('category', '')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        # Base query
        query = """
        SELECT a.*, 
               AVG(ar.latency_score) as avg_latency,
               AVG(ar.ease_of_use) as avg_ease_of_use,
               AVG(ar.docs_quality) as avg_docs_quality,
               AVG(ar.cost_efficiency) as avg_cost_efficiency,
               COUNT(ar.id) as rating_count
        FROM apis a
        LEFT JOIN api_ratings ar ON a.id = ar.api_id
        WHERE 1=1
        """
        params = {}
        
        if search_query:
            if len(search_query) > 2:  # Use semantic search for longer queries
                similar_apis = EmbeddingManager.search_similar(search_query, 'api_doc')
                if similar_apis:
                    api_ids = [item['entity_id'] for item in similar_apis[:20]]
                    query += " AND a.id IN :api_ids"
                    params['api_ids'] = tuple(api_ids)
            else:
                query += " AND (a.name LIKE :search OR a.description LIKE :search)"
                params['search'] = f"%{search_query}%"
        
        if category:
            query += " AND a.category = :category"
            params['category'] = category
        
        query += " GROUP BY a.id ORDER BY rating_count DESC, a.name"
        query += f" LIMIT {per_page} OFFSET {(page - 1) * per_page}"
        
        apis = DatabaseManager.execute_query(query, params)
        
        return jsonify({
            'apis': apis,
            'page': page,
            'per_page': per_page
        })
        
    except Exception as e:
        logger.error(f"Get APIs error: {e}")
        return jsonify({'error': 'Failed to fetch APIs'}), 500

@app.route('/api/apis', methods=['POST'])
@jwt_required()
def create_api():
    """Create a new API entry"""
    try:
        data = request.get_json()
        user_id = get_jwt_identity()
        
        # Check if user has permission (admin/moderator)
        user = DatabaseManager.execute_query(
            "SELECT role FROM users WHERE id = :user_id",
            {'user_id': user_id}
        )
        
        if not user or user[0]['role'] not in ['admin', 'moderator']:
            return jsonify({'error': 'Permission denied'}), 403
        
        api_id = str(uuid.uuid4())
        api_data = {
            'id': api_id,
            'name': data.get('name'),
            'base_url': data.get('base_url'),
            'category': data.get('category'),
            'description': data.get('description'),
            'homepage_url': data.get('homepage_url'),
            'docs_url': data.get('docs_url')
        }
        
        DatabaseManager.execute_query(
            """INSERT INTO apis (id, name, base_url, category, description, homepage_url, docs_url)
               VALUES (:id, :name, :base_url, :category, :description, :homepage_url, :docs_url)""",
            api_data
        )
        
        # Create embedding for the API description
        if data.get('description'):
            embedding_vector = EmbeddingManager.create_embedding(data['description'])
            EmbeddingManager.store_embedding('api_doc', api_id, embedding_vector)
        
        return jsonify({'message': 'API created successfully', 'api_id': api_id})
        
    except Exception as e:
        logger.error(f"Create API error: {e}")
        return jsonify({'error': 'Failed to create API'}), 500

# Community Features - Questions & Answers
@app.route('/api/questions', methods=['GET'])
def get_questions():
    """Get questions with optional filtering"""
    try:
        api_id = request.args.get('api_id')
        search = request.args.get('search', '')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        query = """
        SELECT q.*, u.username,
               COUNT(a.id) as answer_count,
               SUM(CASE WHEN v.vote_type = 'up' THEN 1 ELSE 0 END) as upvotes,
               SUM(CASE WHEN v.vote_type = 'down' THEN 1 ELSE 0 END) as downvotes
        FROM questions q
        JOIN users u ON q.user_id = u.id
        LEFT JOIN answers a ON q.id = a.question_id
        LEFT JOIN votes v ON q.id = v.entity_id AND v.entity_type = 'question'
        WHERE 1=1
        """
        params = {}
        
        if api_id:
            query += " AND q.api_id = :api_id"
            params['api_id'] = api_id
        
        if search:
            query += " AND (q.title LIKE :search OR q.body_md LIKE :search)"
            params['search'] = f"%{search}%"
        
        query += " GROUP BY q.id ORDER BY q.created_at DESC"
        query += f" LIMIT {per_page} OFFSET {(page - 1) * per_page}"
        
        questions = DatabaseManager.execute_query(query, params)
        
        return jsonify({
            'questions': questions,
            'page': page,
            'per_page': per_page
        })
        
    except Exception as e:
        logger.error(f"Get questions error: {e}")
        return jsonify({'error': 'Failed to fetch questions'}), 500

@app.route('/api/questions', methods=['POST'])
@jwt_required()
def create_question():
    """Create a new question"""
    try:
        data = request.get_json()
        user_id = get_jwt_identity()
        question_id = str(uuid.uuid4())
        
        question_data = {
            'id': question_id,
            'user_id': user_id,
            'api_id': data.get('api_id'),
            'title': data.get('title'),
            'body_md': data.get('body_md'),
            'created_at': datetime.now()
        }
        
        DatabaseManager.execute_query(
            """INSERT INTO questions (id, user_id, api_id, title, body_md, created_at)
               VALUES (:id, :user_id, :api_id, :title, :body_md, :created_at)""",
            question_data
        )
        
        # Create embedding for semantic search
        question_text = f"{data.get('title')} {data.get('body_md')}"
        embedding_vector = EmbeddingManager.create_embedding(question_text)
        EmbeddingManager.store_embedding('question', question_id, embedding_vector)
        
        return jsonify({'message': 'Question created successfully', 'question_id': question_id})
        
    except Exception as e:
        logger.error(f"Create question error: {e}")
        return jsonify({'error': 'Failed to create question'}), 500

# AI-Powered Features
@app.route('/api/ai/generate-code', methods=['POST'])
@jwt_required()
def generate_code():
    """Generate code snippet using AI"""
    try:
        data = request.get_json()
        api_id = data.get('api_id')
        language = data.get('language', 'python')
        description = data.get('description', '')
        
        # Get API information
        api_info = DatabaseManager.execute_query(
            "SELECT name, description, docs_url FROM apis WHERE id = :api_id",
            {'api_id': api_id}
        )
        
        if not api_info:
            return jsonify({'error': 'API not found'}), 404
        
        api_desc = f"API: {api_info[0]['name']}\nDescription: {api_info[0]['description']}\nTask: {description}"
        code = LLMService.generate_code(api_desc, language)
        
        return jsonify({'code': code})
        
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        return jsonify({'error': 'Failed to generate code'}), 500

@app.route('/api/ai/recommend-apis', methods=['POST'])
@jwt_required()
def recommend_apis():
    """Recommend APIs based on problem description"""
    try:
        data = request.get_json()
        problem_description = data.get('description', '')
        
        # Get AI recommendations
        recommended_categories = LLMService.recommend_apis(problem_description)
        
        # Find APIs in recommended categories
        if recommended_categories:
            category_list = "', '".join(recommended_categories)
            query = f"""
            SELECT a.*, AVG(ar.ease_of_use + ar.docs_quality) as avg_score
            FROM apis a
            LEFT JOIN api_ratings ar ON a.id = ar.api_id
            WHERE a.category IN ('{category_list}')
            GROUP BY a.id
            ORDER BY avg_score DESC, a.name
            LIMIT 10
            """
            apis = DatabaseManager.execute_query(query)
        else:
            # Fallback to semantic search
            similar_apis = EmbeddingManager.search_similar(problem_description, 'api_doc', 10)
            if similar_apis:
                api_ids = [item['entity_id'] for item in similar_apis]
                api_ids_str = "', '".join(api_ids)
                query = f"SELECT * FROM apis WHERE id IN ('{api_ids_str}')"
                apis = DatabaseManager.execute_query(query)
            else:
                apis = []
        
        return jsonify({
            'recommended_categories': recommended_categories,
            'apis': apis
        })
        
    except Exception as e:
        logger.error(f"API recommendation error: {e}")
        return jsonify({'error': 'Failed to recommend APIs'}), 500

# Reviews and Ratings
@app.route('/api/reviews', methods=['POST'])
@jwt_required()
def create_review():
    """Create an API review"""
    try:
        data = request.get_json()
        user_id = get_jwt_identity()
        review_id = str(uuid.uuid4())
        
        # Analyze sentiment
        sentiment_score = LLMService.analyze_sentiment(data.get('body_md', ''))
        
        review_data = {
            'id': review_id,
            'api_id': data.get('api_id'),
            'user_id': user_id,
            'title': data.get('title'),
            'body_md': data.get('body_md'),
            'sentiment_score': sentiment_score,
            'created_at': datetime.now()
        }
        
        DatabaseManager.execute_query(
            """INSERT INTO api_reviews (id, api_id, user_id, title, body_md, sentiment_score, created_at)
               VALUES (:id, :api_id, :user_id, :title, :body_md, :sentiment_score, :created_at)""",
            review_data
        )
        
        return jsonify({'message': 'Review created successfully', 'review_id': review_id})
        
    except Exception as e:
        logger.error(f"Create review error: {e}")
        return jsonify({'error': 'Failed to create review'}), 500

@app.route('/api/ratings', methods=['POST'])
@jwt_required()
def create_rating():
    """Create an API rating"""
    try:
        data = request.get_json()
        user_id = get_jwt_identity()
        rating_id = str(uuid.uuid4())
        
        rating_data = {
            'id': rating_id,
            'api_id': data.get('api_id'),
            'user_id': user_id,
            'latency_score': data.get('latency_score'),
            'ease_of_use': data.get('ease_of_use'),
            'docs_quality': data.get('docs_quality'),
            'cost_efficiency': data.get('cost_efficiency'),
            'created_at': datetime.now()
        }
        
        DatabaseManager.execute_query(
            """INSERT INTO api_ratings (id, api_id, user_id, latency_score, ease_of_use, docs_quality, cost_efficiency, created_at)
               VALUES (:id, :api_id, :user_id, :latency_score, :ease_of_use, :docs_quality, :cost_efficiency, :created_at)
               ON DUPLICATE KEY UPDATE 
               latency_score = VALUES(latency_score),
               ease_of_use = VALUES(ease_of_use),
               docs_quality = VALUES(docs_quality),
               cost_efficiency = VALUES(cost_efficiency)""",
            rating_data
        )
        
        return jsonify({'message': 'Rating submitted successfully'})
        
    except Exception as e:
        logger.error(f"Create rating error: {e}")
        return jsonify({'error': 'Failed to submit rating'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        DatabaseManager.execute_query("SELECT 1")
        return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'Unauthorized'}), 401

@app.errorhandler(403)
def forbidden(error):
    return jsonify({'error': 'Forbidden'}), 403

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)