import os
import logging
from flask import Flask, jsonify, render_template
from datetime import datetime

# Import your configurations
from .config import config

# Import the extensions and their initializer functions
from .extensions import cors, jwt, init_database, init_langchain

def create_app(config_name='development'):
    """
    Application Factory Function: Creates and configures the Flask app.
    This pattern ensures a clean and organized startup sequence.
    """
    app = Flask(__name__)
    
    # Load configuration from the config object
    app.config.from_object(config[config_name])

    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    
    # --- Initialize Extensions ---
    cors.init_app(app, origins=["http://localhost:3000", "http://localhost:5000"])
    jwt.init_app(app)
    init_database(app.config['SQLALCHEMY_DATABASE_URI'])
    
    # --- Initialize LangChain clients here ---
    # This is the crucial change: we call this function after the app is
    # configured, ensuring .env has been loaded before we access the API key.
    init_langchain()

    # --- Register Blueprints ---
    # This is where your routes get attached to the application.
    from .routes.main_routes import main_bp
    app.register_blueprint(main_bp, url_prefix='/api')

    # --- Toplevel Routes ---
    @app.route('/')
    def index():
        """Serves the main frontend application."""
        return render_template('index.html')

    @app.route('/health')
    def health_check():
        """Provides a simple health check endpoint."""
        from .services.db_manager import DatabaseManager
        try:
            DatabaseManager.execute_query("SELECT 1")
            db_status = 'ok'
        except Exception as e:
            app.logger.error(f"Health check database error: {e}")
            db_status = 'error'
            
        return jsonify({
            'status': 'ok' if db_status == 'ok' else 'error',
            'database': db_status,
            'timestamp': datetime.now().isoformat()
        })

    return app
