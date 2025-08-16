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
