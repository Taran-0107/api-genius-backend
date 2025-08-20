import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flask_cors import CORS
from flask_jwt_extended import JWTManager
# --- LangChain (Gemini) Imports ---
# Replaced Cohere with Google's Generative AI libraries
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Create instances of extensions that don't depend on .env variables
cors = CORS()
jwt = JWTManager()

# --- Database Setup ---
engine = None
SessionLocal = None

def init_database(database_url: str):
    """Initializes the database engine and session maker."""
    global engine, SessionLocal
    engine = create_engine(database_url, pool_size=10, max_overflow=20, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- LangChain (Gemini) Setup ---
# DECLARE the variables here, but set them to None.
# They will be initialized inside the app factory to ensure .env is loaded.
llm = None
embeddings = None

def init_langchain():
    """
    Initializes the LangChain clients for Google Gemini. This is called from the app factory
    to ensure environment variables from .env are loaded first.
    """
    global llm, embeddings
    
    # Use GOOGLE_API_KEY for Gemini
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        # This will give a clearer error if the key is still not found
        raise ValueError("FATAL: GOOGLE_API_KEY environment variable not found. Ensure it is set in your .env file.")
        
    # Initialize the Gemini Chat Model
    llm = ChatGoogleGenerativeAI(
        model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'), 
        google_api_key=api_key,
        # Optional: add a temperature setting for creativity
        temperature=0.7 
    )
    
    # Initialize the Gemini Embeddings Model
    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv('GEMINI_EMBEDDING_MODEL', 'models/embedding-001'),
        google_api_key=api_key
    )