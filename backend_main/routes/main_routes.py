import uuid
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, create_access_token
from werkzeug.security import generate_password_hash, check_password_hash

# Import your service classes that contain the business logic
from ..services.db_manager import DatabaseManager
from ..services.embedding_manager import EmbeddingManager
from ..services.llm_service import LLMService
from ..services.api_agent import ApiDiscoveryAgent

logger = logging.getLogger(__name__)

# A Blueprint is a way to organize a group of related views and other code.
main_bp = Blueprint('main_routes', __name__)


# --- API Routes ---

@main_bp.route('/apis/fromdb', methods=['GET'])
def get_apis():
    """
    Get a list of APIs from the database.
    Supports 'fetch_all', search, category, and pagination.
    """
    try:
        search_query = request.args.get('search', '')
        category = request.args.get('category')
        fetch_all = request.args.get('fetch_all', 'false').lower() == 'true'

        # Build Query Conditions (same as before)
        where_clauses = []
        params = {}

        if search_query:
            where_clauses.append("(a.name LIKE :search OR a.description LIKE :search)")
            params['search'] = f"%{search_query}%"
        
        if category and category != 'all':
            where_clauses.append("a.category = :category")
            params['category'] = category

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # Get Total Count (same as before)
        count_query = f"SELECT COUNT(DISTINCT a.id) as total FROM apis a {where_sql}"
        count_params = params.copy()
        total_result = DatabaseManager.execute_query(count_query, count_params)
        total_apis = total_result[0]['total'] if total_result else 0
        
        # Build the main query
        base_query = """
        SELECT a.*, 
               AVG(ar.latency_score) as avg_latency, AVG(ar.ease_of_use) as avg_ease_of_use,
               AVG(ar.docs_quality) as avg_docs_quality, AVG(ar.cost_efficiency) as avg_cost_efficiency,
               COUNT(ar.id) as rating_count
        FROM apis a LEFT JOIN api_ratings ar ON a.id = ar.api_id
        """
        query = f"""
        {base_query}
        {where_sql}
        GROUP BY a.id
        ORDER BY rating_count DESC, a.name
        """
        
        if not fetch_all:
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 10))
            
            params['limit'] = per_page
            params['offset'] = (page - 1) * per_page
            query += " LIMIT :limit OFFSET :offset"

        apis = DatabaseManager.execute_query(query, params)
        
        # The response now handles both scenarios
        return jsonify({
            'apis': apis, 
            'page': page if not fetch_all else None,
            'per_page': per_page if not fetch_all else None,
            'total_apis': total_apis,
            'fetched_all': fetch_all
        })

    except Exception as e:
        logger.error(f"Get APIs error: {e}")
        return jsonify({'error': 'Failed to fetch APIs'}), 500

@main_bp.route('/apis/discover', methods=['POST'])
def discover_apis():
    """
    Discover new APIs from the web using the agent, store them, and return results.
    """
    try:
        data = request.get_json()
        search_query = data.get('search')
        if not search_query:
            return jsonify({'error': 'Missing search query'}), 400

        # Run the discovery agent
        agent = ApiDiscoveryAgent()
        agent.run(search_query)

        # Fetch the newly discovered APIs to return them in the response
        params = {'search': f"%{search_query}%"}
        query = """
        SELECT a.*, COUNT(ar.id) as rating_count
        FROM apis a LEFT JOIN api_ratings ar ON a.id = ar.api_id
        WHERE a.name LIKE :search
        GROUP BY a.id ORDER BY rating_count DESC
        """
        apis = DatabaseManager.execute_query(query, params)

        return jsonify({'apis': apis})

    except Exception as e:
        logger.error(f"Discover APIs error: {e}")
        return jsonify({'error': 'Failed to discover APIs'}), 500


# --- Authentication Routes ---

@main_bp.route('/auth/register', methods=['POST'])
def register():
    """User registration endpoint."""
    try:
        data = request.get_json()
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not all([username, email, password]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if user already exists
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
               VALUES (:id, :username, :email, :password_hash, 'user', :created_at)""",
            {'id': user_id, 'username': username, 'email': email, 'password_hash': password_hash, 'created_at': datetime.now()}
        )
        
        access_token = create_access_token(identity=user_id)
        return jsonify({
            'access_token': access_token,
            'user': {'id': user_id, 'username': username, 'email': email}
        })
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@main_bp.route('/auth/login', methods=['POST'])
def login():
    """User login endpoint."""
    try:
        data = request.get_json()
        username = data.get('username') # Can be username or email
        password = data.get('password')
        
        if not all([username, password]):
            return jsonify({'error': 'Username and password required'}), 400
        
        user_list = DatabaseManager.execute_query(
            "SELECT id, username, email, password_hash FROM users WHERE username = :username OR email = :username",
            {'username': username}
        )
        
        if not user_list or not check_password_hash(user_list[0]['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        user = user_list[0]
        access_token = create_access_token(identity=user['id'])
        
        return jsonify({
            'access_token': access_token,
            'user': {'id': user['id'], 'username': user['username'], 'email': user['email']}
        })
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


# --- AI-Powered Routes ---

@main_bp.route('/ai/generate-code', methods=['POST'])
@jwt_required()
def generate_code_route():
    """Generate a code snippet for a given API and task."""
    try:
        data = request.get_json()
        api_id = data.get('api_id')
        language = data.get('language', 'python')
        user_task = data.get('description', '')

        if not api_id or not user_task:
            return jsonify({'error': 'api_id and description are required'}), 400
        
        api_info_list = DatabaseManager.execute_query(
            "SELECT name, description, docs_url FROM apis WHERE id = :api_id",
            {'api_id': api_id}
        )
        if not api_info_list:
            return jsonify({'error': 'API not found'}), 404
        
        api_info = api_info_list[0]
        api_description = f"API: {api_info['name']}\nDescription: {api_info['description']}"
        
        # Scrape docs for better context if available
        docs_content = ""
        if api_info.get('docs_url'):
            agent = ApiDiscoveryAgent()
            docs_content = agent._scrape_url_content(api_info['docs_url'])

        print(docs_content)

        code = LLMService.generate_code(api_description, language, user_task, docs_content)
        return jsonify({'code': code})
        
    except Exception as e:
        logger.error(f"Code generation route error: {e}")
        return jsonify({'error': 'Failed to generate code'}), 500


@main_bp.route('/ai/recommend-apis', methods=['POST'])
@jwt_required()
def recommend_apis_route():
    """Recommend APIs based on a user's problem description."""
    try:
        data = request.get_json()
        problem_description = data.get('description', '')
        if not problem_description:
            return jsonify({'error': 'Description is required'}), 400

        # First, try to get recommendations by category from the LLM
        recommended_categories = LLMService.recommend_apis(problem_description)
        
        apis = []
        if recommended_categories:
            # Use safe parameter binding to avoid SQL injection
            cat_placeholders = ', '.join([':cat' + str(i) for i in range(len(recommended_categories))])
            params = {f'cat{i}': cat for i, cat in enumerate(recommended_categories)}
            query = f"""
            SELECT a.*, AVG(ar.ease_of_use + ar.docs_quality) as avg_score
            FROM apis a LEFT JOIN api_ratings ar ON a.id = ar.api_id
            WHERE a.category IN ({cat_placeholders})
            GROUP BY a.id ORDER BY avg_score DESC, a.name LIMIT 10
            """
            apis = DatabaseManager.execute_query(query, params)
        
        # If no category matches, fall back to semantic search
        if not apis:
            similar_apis = EmbeddingManager.search_similar(problem_description, 'api_doc', 10)
            if similar_apis:
                api_ids = tuple([item['entity_id'] for item in similar_apis])
                if api_ids:
                    query = f"SELECT * FROM apis WHERE id IN :api_ids"
                    apis = DatabaseManager.execute_query(query, {'api_ids': api_ids})

        return jsonify({
            'recommended_categories': recommended_categories,
            'apis': apis
        })
        
    except Exception as e:
        logger.error(f"API recommendation route error: {e}")
        return jsonify({'error': 'Failed to recommend APIs'}), 500


@main_bp.route('/ai/compare-apis', methods=['POST'])
@jwt_required()
def compare_apis_route():
    """Compare APIs based on user query and provide ranked recommendations."""
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400

        # Step 1: Get all available categories from database
        categories_result = DatabaseManager.execute_query(
            "SELECT DISTINCT category FROM apis WHERE category IS NOT NULL AND category != ''",
            {}
        )
        categories = [row['category'] for row in categories_result]
        
        if not categories:
            return jsonify({'error': 'No categories found in database'}), 404

        # Step 2: Use LLM to create SQL query based on user intent and categories
        sql_query = LLMService.create_category_sql_query(user_query, categories)
        
        # Step 3: Execute the generated SQL query to fetch relevant APIs
        try:
            apis = DatabaseManager.execute_query(sql_query, {})
        except Exception as sql_error:
            logger.error(f"Generated SQL query failed: {sql_error}")
            # Fallback to simple search if generated query fails
            fallback_query = """
            SELECT a.*, 
                   AVG(ar.latency_score) as avg_latency, 
                   AVG(ar.ease_of_use) as avg_ease_of_use,
                   AVG(ar.docs_quality) as avg_docs_quality, 
                   AVG(ar.cost_efficiency) as avg_cost_efficiency,
                   COUNT(ar.id) as rating_count
            FROM apis a 
            LEFT JOIN api_ratings ar ON a.id = ar.api_id
            WHERE a.description LIKE :search OR a.name LIKE :search
            GROUP BY a.id 
            ORDER BY rating_count DESC, a.name 
            LIMIT 10
            """
            search_term = f"%{user_query}%"
            apis = DatabaseManager.execute_query(fallback_query, {'search': search_term})

        if not apis:
            return jsonify({'error': 'No relevant APIs found for your query'}), 404

        # Step 4: Format API data for LLM comparison
        apis_data = ""
        for i, api in enumerate(apis, 1):
            rating_info = ""
            if api.get('avg_ease_of_use'):
                rating_info = f" (Ease: {api['avg_ease_of_use']:.1f}, Docs: {api.get('avg_docs_quality', 0):.1f}, Cost: {api.get('avg_cost_efficiency', 0):.1f})"
            
            apis_data += f"{i}. {api['name']}: {api['description']}{rating_info}\n"

        # Step 5: Use LLM to compare and rank APIs
        comparison_result = LLMService.compare_apis_for_task(user_query, apis_data)

        return jsonify({
            'query': user_query,
            'generated_sql': sql_query,
            'apis': apis,
            'comparison': comparison_result,
            'total_apis_found': len(apis)
        })
        
    except Exception as e:
        logger.error(f"API comparison route error: {e}")
        return jsonify({'error': 'Failed to compare APIs'}), 500


@main_bp.route('/ai/get-categories', methods=['GET'])
def get_categories_route():
    """Get all available API categories from the database."""
    try:
        categories_result = DatabaseManager.execute_query(
            """SELECT category, COUNT(*) as api_count 
               FROM apis 
               WHERE category IS NOT NULL AND category != '' 
               GROUP BY category 
               ORDER BY api_count DESC, category""",
            {}
        )
        
        return jsonify({
            'categories': categories_result,
            'total_categories': len(categories_result)
        })
        
    except Exception as e:
        logger.error(f"Get categories route error: {e}")
        return jsonify({'error': 'Failed to fetch categories'}), 500


@main_bp.route('/ai/compare-apis-advanced', methods=['POST'])
@jwt_required()
def compare_apis_advanced_route():
    """
    Advanced API comparison with custom filters and detailed analysis.
    Allows users to specify categories, minimum ratings, and other criteria.
    """
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        specified_categories = data.get('categories', [])  # Optional: user can specify categories
        min_ease_of_use = data.get('min_ease_of_use', 0)
        min_docs_quality = data.get('min_docs_quality', 0)
        limit = min(data.get('limit', 10), 20)  # Cap at 20 for performance
        
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400

        # Build dynamic query based on filters
        where_clauses = []
        params = {}
        
        if specified_categories:
            category_placeholders = ', '.join([f':cat{i}' for i in range(len(specified_categories))])
            where_clauses.append(f"a.category IN ({category_placeholders})")
            for i, cat in enumerate(specified_categories):
                params[f'cat{i}'] = cat
        
        # Add rating filters if specified
        having_clauses = []
        if min_ease_of_use > 0:
            having_clauses.append("AVG(ar.ease_of_use) >= :min_ease")
            params['min_ease'] = min_ease_of_use
        if min_docs_quality > 0:
            having_clauses.append("AVG(ar.docs_quality) >= :min_docs")
            params['min_docs'] = min_docs_quality

        # Construct the query
        base_query = """
        SELECT a.*, 
               AVG(ar.latency_score) as avg_latency, 
               AVG(ar.ease_of_use) as avg_ease_of_use,
               AVG(ar.docs_quality) as avg_docs_quality, 
               AVG(ar.cost_efficiency) as avg_cost_efficiency,
               COUNT(ar.id) as rating_count
        FROM apis a 
        LEFT JOIN api_ratings ar ON a.id = ar.api_id
        """
        
        if where_clauses:
            base_query += " WHERE " + " AND ".join(where_clauses)
        
        base_query += " GROUP BY a.id"
        
        if having_clauses:
            base_query += " HAVING " + " AND ".join(having_clauses)
        
        base_query += f" ORDER BY rating_count DESC, a.name LIMIT {limit}"
        
        apis = DatabaseManager.execute_query(base_query, params)

        if not apis:
            return jsonify({'error': 'No APIs found matching your criteria'}), 404

        # Format API data for LLM comparison with more details
        apis_data = ""
        for i, api in enumerate(apis, 1):
            rating_info = ""
            if api.get('rating_count', 0) > 0:
                rating_info = f" | Ratings: {api['rating_count']} users | "
                rating_info += f"Ease: {api.get('avg_ease_of_use', 0):.1f}/5, "
                rating_info += f"Docs: {api.get('avg_docs_quality', 0):.1f}/5, "
                rating_info += f"Cost: {api.get('avg_cost_efficiency', 0):.1f}/5, "
                rating_info += f"Latency: {api.get('avg_latency', 0):.1f}/5"
            
            homepage = f" | Homepage: {api['homepage_url']}" if api.get('homepage_url') else ""
            
            apis_data += f"{i}. {api['name']} ({api['category']})\n"
            apis_data += f"   Description: {api['description']}\n"
            apis_data += f"   {rating_info}{homepage}\n\n"

        # Get enhanced comparison from LLM
        comparison_result = LLMService.compare_apis_for_task(user_query, apis_data)

        return jsonify({
            'query': user_query,
            'filters_applied': {
                'categories': specified_categories,
                'min_ease_of_use': min_ease_of_use,
                'min_docs_quality': min_docs_quality,
                'limit': limit
            },
            'apis': apis,
            'comparison': comparison_result,
            'total_apis_found': len(apis)
        })
        
    except Exception as e:
        logger.error(f"Advanced API comparison route error: {e}")
        return jsonify({'error': 'Failed to perform advanced API comparison'}), 500

# --- Question & Answer Routes ---

@main_bp.route('/questions', methods=['GET'])
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

@main_bp.route('/questions', methods=['POST'])
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

@main_bp.route('/answers', methods=['POST'])
@jwt_required()
def create_answer():
    """Create a new answer for a question"""
    try:
        data = request.get_json()
        user_id = get_jwt_identity()
        answer_id = str(uuid.uuid4())
        
        answer_data = {
            'id': answer_id,
            'question_id': data.get('question_id'),
            'user_id': user_id,
            'body_md': data.get('body_md', ''),
            'created_at': datetime.now()
        }

        DatabaseManager.execute_query(
            """INSERT INTO answers (id, question_id, user_id, body_md, created_at)
               VALUES (:id, :question_id, :user_id, :body_md, :created_at)""",
            answer_data
        )

        answer_text = answer_data['body_md']
        embedding_vector = EmbeddingManager.create_embedding(answer_text)
        EmbeddingManager.store_embedding('answer', answer_id, embedding_vector)

        return jsonify({'message': 'Answer created successfully', 'answer_id': answer_id})

    except Exception as e:
        logger.error(f"Create answer error: {e}")
        return jsonify({'error': 'Failed to create answer'}), 500
    
@main_bp.route('/answers/<question_id>', methods=['GET'])
def get_answers(question_id):
    """Get all answers for a question"""
    try:
        query = """
        SELECT a.*, u.username,
               SUM(CASE WHEN v.vote_type = 'up' THEN 1 ELSE 0 END) as upvotes,
               SUM(CASE WHEN v.vote_type = 'down' THEN 1 ELSE 0 END) as downvotes
        FROM answers a
        JOIN users u ON a.user_id = u.id
        LEFT JOIN votes v ON a.id = v.entity_id AND v.entity_type = 'answer'
        WHERE a.question_id = :question_id
        GROUP BY a.id
        ORDER BY a.created_at ASC
        """
        answers = DatabaseManager.execute_query(query, {'question_id': question_id})
        return jsonify({'answers': answers})

    except Exception as e:
        logger.error(f"Get answers error: {e}")
        return jsonify({'error': 'Failed to fetch answers'}), 500

@main_bp.route('/votes', methods=['POST'])
@jwt_required()
def cast_vote():
    """Cast a vote (up/down) on a question or answer"""
    try:
        data = request.get_json()
        user_id = get_jwt_identity()
        entity_type = data.get('entity_type')   # "question" or "answer"
        entity_id = data.get('entity_id')
        vote_type = data.get('vote_type')       # "up" or "down"

        if entity_type not in ['question', 'answer'] or vote_type not in ['up', 'down']:
            return jsonify({'error': 'Invalid entity_type or vote_type'}), 400

        # Check if user already voted → update instead of duplicate
        existing_vote = DatabaseManager.execute_query(
            """SELECT id FROM votes 
               WHERE user_id = :user_id AND entity_id = :entity_id AND entity_type = :entity_type""",
            {'user_id': user_id, 'entity_id': entity_id, 'entity_type': entity_type}
        )

        if existing_vote:
            # Update existing vote
            DatabaseManager.execute_query(
                """UPDATE votes 
                   SET vote_type = :vote_type, created_at = :created_at
                   WHERE id = :id""",
                {'vote_type': vote_type, 'created_at': datetime.now(), 'id': existing_vote[0]['id']}
            )
            return jsonify({'message': 'Vote updated successfully'})
        else:
            # Insert new vote
            vote_id = str(uuid.uuid4())
            DatabaseManager.execute_query(
                """INSERT INTO votes (id, user_id, entity_type, entity_id, vote_type, created_at)
                   VALUES (:id, :user_id, :entity_type, :entity_id, :vote_type, :created_at)""",
                {
                    'id': vote_id,
                    'user_id': user_id,
                    'entity_type': entity_type,
                    'entity_id': entity_id,
                    'vote_type': vote_type,
                    'created_at': datetime.now()
                }
            )
            return jsonify({'message': 'Vote cast successfully', 'vote_id': vote_id})

    except Exception as e:
        logger.error(f"Vote error: {e}")
        return jsonify({'error': 'Failed to cast vote'}), 500

@main_bp.route('/questions/<question_id>/resolve', methods=['POST'])
@jwt_required()
def toggle_question_resolved(question_id):
    """Mark a question as resolved or unresolved"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        resolved = data.get('resolved', True)

        # Ensure only author can toggle
        result = DatabaseManager.execute_query(
            "SELECT user_id FROM questions WHERE id = :id", {"id": question_id}
        )
        if not result or result[0]['user_id'] != user_id:
            return jsonify({'error': 'Not authorized'}), 403

        DatabaseManager.execute_query(
            "UPDATE questions SET resolved = :resolved, updated_at = :updated_at WHERE id = :id",
            {"resolved": resolved, "updated_at": datetime.now(), "id": question_id}
        )

        return jsonify({'message': 'Question updated', 'resolved': resolved})
    except Exception as e:
        logger.error(f"Toggle resolved error: {e}")
        return jsonify({'error': 'Failed to update question'}), 500

@main_bp.route('/apis/rate', methods=['POST'])
@jwt_required()
def rate_api():
    """Submit or update a user's rating for an API"""
    try:
        data = request.get_json()
        user_id = get_jwt_identity()
        api_id = data.get('api_id')

        if not api_id:
            return jsonify({'error': 'api_id is required'}), 400

        rating_data = {
            'latency_score': data.get('latency_score'),
            'ease_of_use': data.get('ease_of_use'),
            'docs_quality': data.get('docs_quality'),
            'cost_efficiency': data.get('cost_efficiency'),
        }

        # Check if user has already rated
        existing = DatabaseManager.execute_query(
            "SELECT id FROM api_ratings WHERE api_id = :api_id AND user_id = :user_id",
            {'api_id': api_id, 'user_id': user_id}
        )

        if existing:
            DatabaseManager.execute_query(
                """UPDATE api_ratings
                   SET latency_score = :latency_score,
                       ease_of_use = :ease_of_use,
                       docs_quality = :docs_quality,
                       cost_efficiency = :cost_efficiency,
                       created_at = :created_at
                   WHERE id = :id""",
                {**rating_data, 'created_at': datetime.now(), 'id': existing[0]['id']}
            )
            return jsonify({'message': 'Rating updated successfully'})
        else:
            rating_id = str(uuid.uuid4())
            DatabaseManager.execute_query(
                """INSERT INTO api_ratings
                   (id, api_id, user_id, latency_score, ease_of_use, docs_quality, cost_efficiency, created_at)
                   VALUES (:id, :api_id, :user_id, :latency_score, :ease_of_use, :docs_quality, :cost_efficiency, :created_at)""",
                {**rating_data, 'id': rating_id, 'api_id': api_id, 'user_id': user_id, 'created_at': datetime.now()}
            )
            return jsonify({'message': 'Rating submitted successfully', 'rating_id': rating_id})

    except Exception as e:
        logger.error(f"Rate API error: {e}")
        return jsonify({'error': 'Failed to submit rating'}), 500