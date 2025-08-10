"""
Flask Backend for API Discovery and Community Platform
Features: MySQL connectivity, LangChain integration (Cohere), Agentic API Discovery
"""

import os
import uuid
import json
import json5 # Use json5 to handle potentially malformed LLM output
import re
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from functools import lru_cache
from urllib.parse import urljoin
import requests # Added for URL validation

from flask import Flask, request, jsonify, g, render_template
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import pymysql

# --- New Imports for Agentic Workflow ---
from ddgs import DDGS # Use the updated library
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup


# LangChain imports for Cohere
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


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

# LangChain setup (using Cohere)
# Ensure you have set COHERE_API_KEY in your .env file
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=app.config['COHERE_API_KEY']
)
llm = ChatCohere(
    model=app.config['COHERE_MODEL'], 
    cohere_api_key=app.config['COHERE_API_KEY']
)


# --- NEW: ROUTE TO RENDER THE FRONTEND ---
@app.route('/')
def index():
    """Serves the main frontend application."""
    return render_template('index.html')

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
        
        embeddings_query = """
        SELECT entity_id, vector FROM embeddings 
        WHERE entity_type = :entity_type
        """
        embeddings_data = DatabaseManager.execute_query(
            embeddings_query, 
            {'entity_type': entity_type}
        )
        
        similarities = []
        for item in embeddings_data:
            stored_vector = json.loads(item['vector'])
            similarity = cosine_similarity(query_vector, stored_vector)
            similarities.append({
                'entity_id': item['entity_id'],
                'similarity': similarity
            })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    import numpy as np
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# ==============================================================================
# --- AGENTIC API DISCOVERY WORKFLOW (with enhanced logging and fixes) ---
# ==============================================================================

def _extract_json_from_llm(text: str) -> Optional[Dict]:
    """Extracts a JSON object or list from a string, even if it's wrapped in markdown."""
    # Updated regex to find JSON object OR list
    match = re.search(r'```json\s*([\[\{].*?[\]\}])\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text
    
    try:
        return json5.loads(json_str)
    except Exception as e:
        print(f"⚠️ JSON parsing failed: {e}")
        return None


class ApiDiscoveryAgent:
    """An agent that can find, scrape, and process information about APIs from the web."""

    def __init__(self):
        """Initializes the agent and its tools, handling potential setup errors."""
        self.driver = None
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        except Exception as e:
            print("\n--- ❌ SELENIUM WEBDRIVER FAILED TO INITIALIZE ---")
            print(f"Error: {e}")
            print("The agent's web scraping capabilities will be disabled.")
            print("Please ensure Google Chrome is installed on the system.\n")

    def _generate_search_queries(self, user_query: str) -> List[str]:
        """Uses an LLM to generate effective web search queries."""
        print("\n--- STEP 1: GENERATING SEARCH QUERIES ---")
        prompt = f"""
        As an expert developer, your task is to find the best API for a given problem.
        Based on the user's request, generate 3 diverse and high-quality web search queries to find the official API documentation, developer portal, or getting started guide.
        
        User request: "{user_query}"
        
        Return ONLY a JSON list of 3 string queries.
        """
        try:
            response = llm.invoke(prompt)
            queries_data = _extract_json_from_llm(response.content)
            if isinstance(queries_data, list):
                print(f"✅ LLM generated queries: {queries_data}")
                return queries_data
            else:
                 raise ValueError("LLM did not return a valid list.")
        except Exception as e:
            print(f"⚠️ LLM failed to generate queries, using fallback. Error: {e}")
            return [user_query]

    def _search_web_for_urls(self, queries: List[str], num_results: int = 3) -> List[str]:
        """Searches the web and returns a list of unique URLs."""
        print("\n--- STEP 2: SEARCHING THE WEB ---")
        urls = set()
        with DDGS() as ddgs:
            for query in queries:
                print(f"Executing search for: '{query}'")
                try:
                    results = list(ddgs.text(query, max_results=num_results))
                    for result in results:
                        urls.add(result['href'])
                except Exception as e:
                    print(f"⚠️ Web search failed for query '{query}': {e}")
        
        if urls:
            print(f"✅ Found unique URLs: {list(urls)}")
        else:
            print("❌ No URLs found from web search.")
        return list(urls)

    def _scrape_url_content(self, url: str) -> str:
        """Scrapes the textual content of a given URL using Selenium."""
        if not self.driver:
            return ""
        print(f"\n--- STEP 3: SCRAPING URL ---")
        print(f"Scraping content from: {url}")
        try:
            self.driver.get(url)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            print(f"✅ Successfully scraped {len(text)} characters.")
            return text
        except Exception as e:
            print(f"⚠️ Failed to scrape {url}: {e}")
            return ""

    def _validate_content_relevance(self, content: str, user_query: str) -> bool:
        """Uses an LLM to quickly check if the content appears to be API documentation."""
        print("\n--- STEP 3.5: VALIDATING CONTENT RELEVANCE ---")
        prompt = f"""
        Based on the user query "{user_query}", does the following text appear to be technical API documentation, a developer portal, or a getting-started guide?
        Answer with only "yes" or "no".

        Text snippet:
        ---
        {content[:1500]}
        ---
        """
        try:
            response = llm.invoke(prompt)
            answer = response.content.strip().lower()
            print(f"LLM validation result: '{answer}'")
            return "yes" in answer
        except Exception as e:
            print(f"⚠️ Content validation failed: {e}")
            return False

    def _verify_url_is_working(self, url: str) -> bool:
        """Checks if a URL is live and returns a 200 status code."""
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            if response.status_code == 200:
                return True
            return False
        except requests.RequestException:
            return False

    def _extract_api_info_with_llm(self, content: str, user_query: str) -> Optional[Dict]:
        """Uses an LLM to parse scraped content into a structured API format."""
        print("\n--- STEP 4: EXTRACTING INFO WITH LLM ---")
        prompt = f"""
        You are an expert API analyst. Based on the following scraped text from a website, extract the information for the API related to the user's query: "{user_query}".
        
        Scraped Text (first 8000 chars):
        ---
        {content[:8000]} 
        ---
        
        Extract the following fields and return them in a single, clean JSON object.
        - name: The official name of the API.
        - description: A concise, one-sentence summary of what the API does.
        - category: A single, relevant category (e.g., 'Payment', 'Messaging', 'Maps').
        - homepage_url: The root URL of the API provider's website.
        - docs_url: The specific URL for the documentation page.
        
        If you cannot find the information, return null for that field.
        """
        try:
            response = llm.invoke(prompt)
            api_data = _extract_json_from_llm(response.content)
            if api_data and api_data.get('name') and api_data.get('description'):
                print(f"✅ LLM extracted API info: {api_data}")
                return api_data
            print("⚠️ LLM could not extract valid 'name' and 'description'.")
            return None
        except Exception as e:
            print(f"⚠️ Failed to extract API info with LLM. Error: {e}")
            return None

    def _find_and_verify_docs_url(self, api_data: Dict) -> str:
        """Finds and verifies the best documentation URL."""
        print("\n--- STEP 4.5: FINDING & VERIFYING DOCS URL ---")
        base_url = api_data.get('homepage_url')
        if not base_url:
            print("⚠️ No homepage URL to search for docs.")
            return api_data.get('docs_url')

        print(f"Searching for docs link on: {base_url}")
        try:
            self.driver.get(base_url)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Find all links with keywords
            keywords = ['docs', 'documentation', 'api', 'developer', 'reference']
            potential_links = []
            for a in soup.find_all('a', href=True):
                if any(keyword in a.text.lower() or keyword in a['href'].lower() for keyword in keywords):
                    potential_links.append(urljoin(base_url, a['href']))
            
            print(f"Found potential docs links: {potential_links}")

            # Validate the most promising link
            if potential_links:
                best_link = potential_links[0]
                content = self._scrape_url_content(best_link)
                if self._validate_content_relevance(content, api_data['name']):
                    print(f"✅ Verified new docs URL: {best_link}")
                    return best_link
        except Exception as e:
            print(f"⚠️ Error while searching for docs link: {e}")

        print("⚠️ Could not verify a better docs URL, using original.")
        return api_data.get('docs_url')

    def _save_api_to_db(self, api_data: Dict) -> str:
        """Saves the newly discovered API to the database and creates an embedding."""
        print("\n--- STEP 5: SAVING TO DATABASE ---")
        api_id = str(uuid.uuid4())
        db_data = {
            'id': api_id,
            'name': api_data.get('name'),
            'description': api_data.get('description'),
            'category': api_data.get('category'),
            'homepage_url': api_data.get('homepage_url'),
            'docs_url': api_data.get('docs_url'),
            'last_fetched': datetime.now()
        }

        DatabaseManager.execute_query(
            """INSERT INTO apis (id, name, description, category, homepage_url, docs_url, last_fetched)
               VALUES (:id, :name, :description, :category, :homepage_url, :docs_url, :last_fetched)""",
            db_data
        )
        print(f"✅ Saved new API '{db_data['name']}' to database with ID {api_id}")

        embedding_text = f"Name: {db_data['name']}. Description: {db_data['description']}"
        vector = EmbeddingManager.create_embedding(embedding_text)
        if vector:
            EmbeddingManager.store_embedding('api_doc', api_id, vector)
            print(f"✅ Stored embedding for new API {api_id}")
        
        return api_id

    def run(self, user_query: str) -> None:
        """Executes the full discovery workflow for all found URLs."""
        if not self.driver:
            print("❌ Discovery Agent cannot run because WebDriver is not initialized.")
            return
            
        print(f"\n\n{'='*20} 🚀 AGENT WORKFLOW STARTED 🚀 {'='*20}")
        print(f"User Query: '{user_query}'")
        
        search_queries = self._generate_search_queries(user_query)
        urls = self._search_web_for_urls(search_queries)
        if not urls:
            print(f"\n{'='*20} 🏁 AGENT WORKFLOW FINISHED 🏁 {'='*20}\n")
            return

        apis_found_count = 0
        for scraped_url in urls:
            content = self._scrape_url_content(scraped_url)
            if not content:
                continue

            if not self._validate_content_relevance(content, user_query):
                print(f"❌ Content from {scraped_url} is not relevant. Discarding.")
                continue
            
            extracted_info = self._extract_api_info_with_llm(content, user_query)
            if extracted_info:
                print("\n--- STEP 4.7: VALIDATING EXTRACTED URLS ---")
                homepage_url = extracted_info.get('homepage_url')
                docs_url = extracted_info.get('docs_url')

                if homepage_url and not self._verify_url_is_working(homepage_url):
                    print(f"⚠️ Homepage URL {homepage_url} is not working. Correcting with scraped URL.")
                    extracted_info['homepage_url'] = scraped_url
                else:
                    print("✅ Homepage URL is valid.")

                if docs_url and not self._verify_url_is_working(docs_url):
                    print(f"⚠️ Docs URL {docs_url} is not working. Correcting with scraped URL.")
                    extracted_info['docs_url'] = scraped_url
                else:
                    print("✅ Docs URL is valid.")
                
                verified_docs_url = self._find_and_verify_docs_url(extracted_info)
                extracted_info['docs_url'] = verified_docs_url

                self._save_api_to_db(extracted_info)
                apis_found_count += 1
        
        print(f"\n{'='*20} 🏁 AGENT WORKFLOW FINISHED 🏁 {'='*20}")
        print(f"Found and saved {apis_found_count} new API(s).\n")
        return

    def __del__(self):
        """Ensure the browser is closed when the agent is destroyed."""
        if self.driver:
            self.driver.quit()


# ==============================================================================
# --- LLM SERVICE & OTHER CLASSES ---
# ==============================================================================

class LLMService:
    """LangChain LLM service for various AI tasks"""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def generate_code(api_description: str, language: str, user_task: str, docs_content: str = "") -> str:
        """Generate code snippet for API integration, using docs content if available."""
        prompt = f"""
        You are an expert developer tasked with writing a code snippet.
        
        **API Information:**
        {api_description}
        
        **User's Task:**
        {user_task}
        
        **Full API Documentation (for context):**
        ---
        {docs_content[:8000]}
        ---
        
        Based on the user's task and the provided documentation, generate a clean, runnable code snippet in {language}.
        The code should be complete and include necessary imports and error handling.
        Prioritize using examples from the documentation if available.
        """
        
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Code generation error: {e}")
            return "# Error generating code"
    
    @staticmethod
    @lru_cache(maxsize=128) # Simple in-memory cache
    def analyze_sentiment(text: str) -> float:
        """Analyze sentiment of review text"""
        prompt = f"""
        Analyze the sentiment of this API review and return a score between -1 (very negative) and 1 (very positive):
        
        {text}
        
        Return only a decimal number between -1 and 1.
        """
        
        try:
            response = llm.invoke(prompt)
            import re
            match = re.search(r'-?\d+\.?\d*', response.content)
            if match:
                return float(match.group())
            return 0.0
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return 0.0
    
    @staticmethod
    @lru_cache(maxsize=128) # Simple in-memory cache
    def recommend_apis(problem_description: str) -> List[str]:
        """Recommend APIs based on problem description"""
        prompt = f"""
        Based on this problem description, recommend relevant API categories or types:
        {problem_description}
        
        Return a comma-separated list of API categories.
        """
        
        try:
            response = llm.invoke(prompt)
            return [cat.strip() for cat in response.content.split(',')]
        except Exception as e:
            logger.error(f"API recommendation error: {e}")
            return []


# ==============================================================================
# --- FLASK ROUTES ---
# ==============================================================================

@app.route('/api/apis', methods=['GET'])
def get_apis():
    """
    Get list of APIs. Triggers the ApiDiscoveryAgent on every search to find
    and store new APIs before querying the local database with an improved
    search strategy (exact match first, then semantic).
    """
    try:
        search_query = request.args.get('search', '')
        category = request.args.get('category', '')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        apis = []
        params = {}
        
        # --- AGENTIC WORKFLOW & IMPROVED SEARCH LOGIC ---
        if search_query:
            # Step 1: ALWAYS run the agent to discover and store new APIs from the web.
            agent = ApiDiscoveryAgent()
            agent.run(search_query) # This saves new APIs to the DB, we don't need its return value here.

            # Step 2: Now, query the database which may contain the newly added API.
            # Prioritize exact name matches for precision.
            base_query = """
            SELECT a.*, 
                   AVG(ar.latency_score) as avg_latency, AVG(ar.ease_of_use) as avg_ease_of_use,
                   AVG(ar.docs_quality) as avg_docs_quality, AVG(ar.cost_efficiency) as avg_cost_efficiency,
                   COUNT(ar.id) as rating_count
            FROM apis a LEFT JOIN api_ratings ar ON a.id = ar.api_id
            """
            
            # A: Exact(ish) name search
            params['search'] = f"%{search_query}%"
            name_query = f"{base_query} WHERE a.name LIKE :search GROUP BY a.id ORDER BY rating_count DESC"
            apis = DatabaseManager.execute_query(name_query, {'search': params['search']})

            # B: If no direct name match, perform a broader semantic search.
            if not apis:
                print("No direct name match found. Performing semantic search...")
                similar_apis = EmbeddingManager.search_similar(search_query, 'api_doc')
                if similar_apis:
                    api_ids = tuple([item['entity_id'] for item in similar_apis])
                    if api_ids:
                        semantic_query = f"{base_query} WHERE a.id IN :api_ids GROUP BY a.id ORDER BY rating_count DESC"
                        apis = DatabaseManager.execute_query(semantic_query, {'api_ids': api_ids})

        # If there's no search query, just fetch all APIs.
        else:
            query = f"""
            SELECT a.*, 
                   AVG(ar.latency_score) as avg_latency, AVG(ar.ease_of_use) as avg_ease_of_use,
                   AVG(ar.docs_quality) as avg_docs_quality, AVG(ar.cost_efficiency) as avg_cost_efficiency,
                   COUNT(ar.id) as rating_count
            FROM apis a LEFT JOIN api_ratings ar ON a.id = ar.api_id
            GROUP BY a.id ORDER BY rating_count DESC, a.name 
            LIMIT {per_page} OFFSET {(page - 1) * per_page}
            """
            apis = DatabaseManager.execute_query(query)

        return jsonify({
            'apis': apis,
            'page': page,
            'per_page': per_page
        })
        
    except Exception as e:
        logger.error(f"Get APIs error: {e}")
        return jsonify({'error': 'Failed to fetch APIs'}), 500


# --- OTHER ROUTES (Unchanged) ---
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
        
        existing_user = DatabaseManager.execute_query(
            "SELECT id FROM users WHERE username = :username OR email = :email",
            {'username': username, 'email': email}
        )
        
        if existing_user:
            return jsonify({'error': 'User already exists'}), 409
        
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

@app.route('/api/apis', methods=['POST'])
@jwt_required()
def create_api():
    """Create a new API entry (Manual)"""
    try:
        data = request.get_json()
        user_id = get_jwt_identity()
        
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
        
        if data.get('description'):
            embedding_vector = EmbeddingManager.create_embedding(data['description'])
            EmbeddingManager.store_embedding('api_doc', api_id, embedding_vector)
        
        return jsonify({'message': 'API created successfully', 'api_id': api_id})
        
    except Exception as e:
        logger.error(f"Create API error: {e}")
        return jsonify({'error': 'Failed to create API'}), 500

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
            'body_md': data.get('body_md', ''), # Use default if body is missing
            'created_at': datetime.now()
        }
        
        DatabaseManager.execute_query(
            """INSERT INTO questions (id, user_id, api_id, title, body_md, created_at)
               VALUES (:id, :user_id, :api_id, :title, :body_md, :created_at)""",
            question_data
        )
        
        question_text = f"{question_data['title']} {question_data['body_md']}"
        embedding_vector = EmbeddingManager.create_embedding(question_text)
        EmbeddingManager.store_embedding('question', question_id, embedding_vector)
        
        return jsonify({'message': 'Question created successfully', 'question_id': question_id})
        
    except Exception as e:
        logger.error(f"Create question error: {e}")
        return jsonify({'error': 'Failed to create question'}), 500

@app.route('/api/ai/generate-code', methods=['POST'])
@jwt_required()
def generate_code():
    """Generate code snippet using AI"""
    try:
        data = request.get_json()
        api_id = data.get('api_id')
        language = data.get('language', 'python')
        user_task = data.get('description', '')
        
        api_info_list = DatabaseManager.execute_query(
            "SELECT name, description, docs_url FROM apis WHERE id = :api_id",
            {'api_id': api_id}
        )
        
        if not api_info_list:
            return jsonify({'error': 'API not found'}), 404
        
        api_info = api_info_list[0]
        api_description = f"API: {api_info['name']}\nDescription: {api_info['description']}"
        
        # Scrape the docs page for context
        docs_content = ""
        if api_info.get('docs_url'):
            agent = ApiDiscoveryAgent()
            docs_content = agent._scrape_url_content(api_info['docs_url'])

        code = LLMService.generate_code(api_description, language, user_task, docs_content)
        
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
        
        recommended_categories = LLMService.recommend_apis(problem_description)
        
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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        DatabaseManager.execute_query("SELECT 1")
        return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
