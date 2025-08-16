import os
from backend_main import create_app

# It's good practice to load environment variables at the very start
from dotenv import load_dotenv
load_dotenv()

# Determine which configuration to use from environment variables
config_name = os.getenv('FLASK_CONFIG', 'development')

# Create the Flask app instance using the factory function
app = create_app(config_name)

if __name__ == '__main__':
    # Run the application
    # Host and port can be configured in your config.py or directly here
    app.run(host='0.0.0.0', port=5000)
