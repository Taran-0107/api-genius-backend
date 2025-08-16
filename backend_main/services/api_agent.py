import uuid
import requests
import logging
from datetime import datetime
from urllib.parse import urljoin
from typing import List, Dict, Optional

# Web scraping and automation imports
from ddgs import DDGS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Local/relative imports from within the application
from ..extensions import llm
from ..utils.helpers import extract_json_from_llm
from .db_manager import DatabaseManager
from .embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

class ApiDiscoveryAgent:
    """
    An agentic workflow to discover, scrape, analyze, and store new API information.
    """

    def __init__(self):
        """Initializes the agent and its web scraping tools."""
        self.driver = None
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            self.driver = webdriver.Chrome(
                service=ChromeService(ChromeDriverManager().install()), 
                options=chrome_options
            )
        except Exception as e:
            logger.error(f"Selenium WebDriver failed to initialize: {e}")
            # The agent can continue without scraping capabilities, though it will be limited.

    def _generate_search_queries(self, user_query: str) -> List[str]:
        """Uses an LLM to generate effective web search queries."""
        logger.info("Step 1: Generating search queries...")
        prompt = f"""
        Generate 3 diverse, high-quality web search queries to find the official 
        API documentation or developer portal for: "{user_query}"
        Return ONLY a JSON list of 3 string queries.
        """
        try:
            response = llm.invoke(prompt)
            queries_data = extract_json_from_llm(response.content)
            if isinstance(queries_data, list) and all(isinstance(q, str) for q in queries_data):
                logger.info(f"Generated queries: {queries_data}")
                return queries_data
            raise ValueError("LLM did not return a valid list of strings.")
        except Exception as e:
            logger.warning(f"LLM query generation failed, using fallback. Error: {e}")
            return [user_query] # Fallback to the original user query

    def _search_web_for_urls(self, queries: List[str], num_results: int = 3) -> List[str]:
        """Searches the web and returns a list of unique URLs."""
        logger.info("Step 2: Searching the web...")
        unique_urls = set()
        with DDGS() as ddgs:
            for query in queries:
                try:
                    results = ddgs.text(query, max_results=num_results)
                    for result in results:
                        if 'href' in result:
                            unique_urls.add(result['href'])
                except Exception as e:
                    logger.error(f"Web search failed for query '{query}': {e}")
        
        logger.info(f"Found {len(unique_urls)} unique URLs.")
        return list(unique_urls)

    def _scrape_url_content(self, url: str) -> str:
        """Scrapes the primary textual content of a given URL."""
        if not self.driver:
            logger.warning("Scraping skipped: WebDriver not available.")
            return ""
        
        logger.info(f"Step 3: Scraping content from: {url}")
        try:
            self.driver.get(url)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            # Remove non-content tags
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.extract()
            text = soup.get_text(separator=' ', strip=True)
            logger.info(f"Successfully scraped {len(text)} characters.")
            return text
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return ""

    def _extract_api_info_with_llm(self, content: str, user_query: str) -> Optional[Dict]:
        """Uses an LLM to parse scraped content into a structured API format."""
        logger.info("Step 4: Extracting API info with LLM...")
        prompt = f"""
        Analyze the following text from a website related to "{user_query}".
        Extract the key information and return it as a single JSON object.
        
        Scraped Text (first 8000 chars):
        ---
        {content[:8000]} 
        ---
        
        Extract these fields:
        - name: The official API name.
        - description: A concise, one-sentence summary.
        - category: A single, relevant category (e.g., 'Payment', 'Messaging', 'Maps').
        - homepage_url: The root URL of the API provider's website.
        - docs_url: The specific URL for the documentation page.
        
        If a field is not found, return null for its value.
        """
        try:
            response = llm.invoke(prompt)
            api_data = extract_json_from_llm(response.content)
            if api_data and api_data.get('name') and api_data.get('description'):
                logger.info(f"LLM extracted API info: {api_data}")
                return api_data
            logger.warning("LLM could not extract required 'name' and 'description'.")
            return None
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return None

    def _save_api_to_db(self, api_data: Dict, source_url: str) -> str:
        """Saves the newly discovered API to the database and creates an embedding."""
        logger.info("Step 5: Saving API to database...")
        api_id = str(uuid.uuid4())
        
        # Use the source URL as a fallback if specific URLs weren't found
        db_data = {
            'id': api_id,
            'name': api_data.get('name'),
            'description': api_data.get('description'),
            'category': api_data.get('category'),
            'homepage_url': api_data.get('homepage_url') or source_url,
            'docs_url': api_data.get('docs_url') or source_url,
            'last_fetched': datetime.now()
        }

        DatabaseManager.execute_query(
            """INSERT INTO apis (id, name, description, category, homepage_url, docs_url, last_fetched)
               VALUES (:id, :name, :description, :category, :homepage_url, :docs_url, :last_fetched)""",
            db_data
        )
        logger.info(f"Saved new API '{db_data['name']}' with ID {api_id}")

        # Create and store an embedding for the new API
        embedding_text = f"Name: {db_data['name']}. Description: {db_data['description']}"
        vector = EmbeddingManager.create_embedding(embedding_text)
        if vector:
            EmbeddingManager.store_embedding('api_doc', api_id, vector)
            logger.info(f"Stored embedding for new API {api_id}")
        
        return api_id

    def run(self, user_query: str):
        """Executes the full discovery workflow."""
        logger.info(f"--- 🚀 AGENT WORKFLOW STARTED for query: '{user_query}' 🚀 ---")
        
        search_queries = self._generate_search_queries(user_query)
        urls = self._search_web_for_urls(search_queries)
        if not urls:
            logger.info("Workflow finished: No URLs found.")
            return

        apis_found_count = 0
        for url in urls:
            content = self._scrape_url_content(url)
            if not content or len(content) < 100: # Skip empty or tiny pages
                continue
            
            extracted_info = self._extract_api_info_with_llm(content, user_query)
            if extracted_info:
                self._save_api_to_db(extracted_info, url)
                apis_found_count += 1
        
        logger.info(f"--- 🏁 AGENT WORKFLOW FINISHED: Found and saved {apis_found_count} new API(s). 🏁 ---")

    def __del__(self):
        """Ensures the browser is closed when the agent object is destroyed."""
        if self.driver:
            self.driver.quit()
