import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from langchain_cohere import ChatCohere, CohereEmbeddings

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

# --- LangChain (Cohere) Setup ---
# DECLARE the variables here, but set them to None.
# They will be initialized inside the app factory to ensure .env is loaded.
llm = None
embeddings = None

def init_langchain():
    """
    Initializes the LangChain clients. This is called from the app factory
    to ensure environment variables from .env are loaded first.
    """
    global llm, embeddings
    
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        # This will give a clearer error if the key is still not found
        raise ValueError("FATAL: COHERE_API_KEY environment variable not found. Ensure it is set in your .env file.")
        
    llm = ChatCohere(
        model=os.getenv('COHERE_MODEL', 'command-r-plus'), 
        cohere_api_key=api_key
    )
    embeddings = CohereEmbeddings(
        model=os.getenv('COHERE_EMBEDDING_MODEL', 'embed-english-v3.0'),
        cohere_api_key=api_key
    )
