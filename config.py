# config.py
import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    TESTING = False
    
    # JWT settings
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=int(os.getenv('JWT_EXPIRES_HOURS', 24)))
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=int(os.getenv('JWT_REFRESH_DAYS', 30)))
    
    # Database settings
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 3306))
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'api_app')
    
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # AI/LLM settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # Redis settings (for caching/background tasks)
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # API Rate limiting
    RATELIMIT_STORAGE_URL = REDIS_URL
    RATELIMIT_DEFAULT = os.getenv('RATE_LIMIT_DEFAULT', '100/hour')
    
    # File upload settings
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_FILE_SIZE', 16 * 1024 * 1024))  # 16MB
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    
    # Pagination
    QUESTIONS_PER_PAGE = int(os.getenv('QUESTIONS_PER_PAGE', 20))
    APIS_PER_PAGE = int(os.getenv('APIS_PER_PAGE', 20))
    
    # Email settings (for notifications)
    MAIL_SERVER = os.getenv('MAIL_SERVER')
    MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
    MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
    MAIL_USERNAME = os.getenv('MAIL_USERNAME')
    MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)  # Shorter for dev

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DB_NAME = 'api_app_test'
    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/api_app_test?charset=utf8mb4"

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Production should have stronger settings
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    
    # SSL in production
    PREFERRED_URL_SCHEME = 'https'

# Config dictionary for easy selection
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}