import logging
import re
from functools import lru_cache
from typing import List

# Import the shared LLM instance
from ..extensions import llm

logger = logging.getLogger(__name__)

class LLMService:
    """
    A service class for interacting with the Language Model.
    Uses caching to avoid redundant API calls for the same inputs.
    """
    
    @staticmethod
    @lru_cache(maxsize=128)
    def generate_code(api_description: str, language: str, user_task: str, docs_content: str = "") -> str:
        """
        Generates a code snippet for API integration using the LLM.
        """
        prompt = f"""
        You are an expert developer tasked with writing a code snippet.
        
        **API Information:**
        {api_description}
        
        **User's Task:**
        {user_task}
        
        **Relevant API Documentation (for context):**
        ---
        {docs_content[:8000]}
        ---
        
        Based on the user's task and the provided documentation, generate a clean, 
        runnable code snippet in {language}. The code should be complete, including 
        necessary imports and basic error handling.
        """
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"LLM code generation failed: {e}")
            return f"# Error: Could not generate code. {e}"
    
    @staticmethod
    @lru_cache(maxsize=128)
    def analyze_sentiment(text: str) -> float:
        """Analyzes the sentiment of a text and returns a score from -1.0 to 1.0."""
        prompt = f"""
        Analyze the sentiment of this API review. Return a single decimal score 
        between -1 (very negative) and 1 (very positive).
        
        Review: "{text}"
        
        Return only the number.
        """
        try:
            response = llm.invoke(prompt)
            # Use regex to find the first floating-point number in the response
            match = re.search(r'-?\d+\.\d+|-?\d+', response.content.strip())
            if match:
                return float(match.group())
            return 0.0
        except Exception as e:
            logger.error(f"LLM sentiment analysis failed: {e}")
            return 0.0
    
    @staticmethod
    @lru_cache(maxsize=128)
    def recommend_apis(problem_description: str) -> List[str]:
        """Recommends API categories based on a problem description."""
        prompt = f"""
        A user wants to solve the following problem: "{problem_description}"
        
        Based on this, what are the most relevant API categories?
        Return a comma-separated list of 2-3 categories (e.g., Payment, Maps, Weather).
        """
        try:
            response = llm.invoke(prompt)
            # Clean up the response and split into a list
            return [cat.strip() for cat in response.content.split(',') if cat.strip()]
        except Exception as e:
            logger.error(f"LLM API recommendation failed: {e}")
            return []
    
    @staticmethod
    @lru_cache(maxsize=128)
    def create_category_sql_query(user_query: str, categories: List[str]) -> str:
        """Creates a SQL query to fetch APIs of a specific category based on user intent."""
        categories_str = ", ".join([f"'{cat}'" for cat in categories])
        prompt = f"""
        You are a SQL expert. Based on the user's query and available categories, create a SQL query to fetch relevant APIs.
        
        User Query: "{user_query}"
        
        Available Categories: {categories_str}
        
        Database Schema:
        - Table: apis
        - Columns: id, name, base_url, category, description, homepage_url, last_known_version, docs_url, last_fetched
        
        Additional context from ratings table (api_ratings):
        - Columns: api_id, latency_score, ease_of_use, docs_quality, cost_efficiency
        
        Create a SQL query that:
        1. Selects APIs from the most relevant category based on the user's query
        2. Joins with api_ratings to get average scores
        3. Orders by relevance (use ratings if available)
        4. Limits to top 10 results
        
        Return ONLY the SQL query, no explanation.
        """
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM SQL query generation failed: {e}")
            return f"SELECT * FROM apis WHERE category IN ({categories_str}) LIMIT 10"
    
    @staticmethod
    @lru_cache(maxsize=128)
    def compare_apis_for_task(user_task: str, apis_data: str) -> str:
        """Compares APIs and provides ranking with reasoning for a specific task."""
        prompt = f"""
        You are an expert API consultant. A user wants to accomplish the following task:
        
        Task: "{user_task}"
        
        Available APIs:
        {apis_data}
        
        Please analyze these APIs and provide:
        1. A brief summary of your reasoning (2-3 sentences)
        2. A ranked list of APIs from most suitable to least suitable for this task
        
        Consider factors like:
        - Relevance to the task
        - Ease of use (if rating available)
        - Documentation quality (if rating available)
        - Cost efficiency (if rating available)
        - API description and features
        
        Format your response as:
        **Reasoning:** [Your reasoning here]
        
        **Ranked APIs:**
        1. [API Name] - [Brief explanation why it's suitable]
        2. [API Name] - [Brief explanation why it's suitable]
        3. [Continue for all APIs...]
        """
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"LLM API comparison failed: {e}")
            return f"Error: Could not analyze APIs. {e}"
