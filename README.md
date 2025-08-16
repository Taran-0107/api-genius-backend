🚀 API Genius: AI-Powered API Discovery and Integration Platform (Backend)API Genius is a full-stack web application designed to streamline the process of discovering, understanding, and integrating third-party APIs. It leverages the power of Large Language Models (LLMs) through Cohere to provide an intelligent, agentic workflow that can actively find new APIs on the web, recommend the best tools for a given task, and even generate ready-to-use code snippets.This repository contains the Flask backend which powers the entire application and serves a simple HTML/Alpine.js frontend for testing and demonstration purposes.✨ Key Features🤖 Agentic API Discovery: Describe a need (e.g., "an API for sending emails"), and an autonomous agent will search the web, scrape documentation, and intelligently extract and save relevant API information to the database.🧠 AI-Powered Recommendations: Not sure which API to use? Describe your problem, and the AI Assistant will analyze your request and recommend the most suitable APIs from its knowledge base.💻 Instant Code Generation: Select an API and describe a task. The AI will generate a functional code snippet in your desired language (e.g., Python, JavaScript), complete with necessary imports and error handling.💬 Community Q&A: A built-in forum for users to ask questions, share solutions, and discuss integrations related to specific APIs.🔍 Semantic Search: Find APIs not just by name, but by describing what you want to do. The system uses vector embeddings to understand the meaning behind your query.🔐 Secure User Authentication: JWT-based authentication for user registration, login, and securing protected endpoints.🛠️ Tech StackAreaTechnologyBackendPython, Flask, SQLAlchemy, Cohere (for LLM & Embeddings), LangChain, Selenium, JWTDatabaseMySQL (or any SQLAlchemy-compatible database like PostgreSQL, SQLite)FrontendA simple test interface using HTML, Tailwind CSS, and Alpine.js served directly from Flask.📂 Project StructureThe project is organized as a standard Flask application./api-genius-backend/
├── /yourapi/               # Main Flask application package
│   ├── /routes/            # API endpoints (Blueprints)
│   ├── /services/          # Business logic (classes for DB, AI, etc.)
│   ├── /templates/         # Contains the index.html test frontend
│   │   └── index.html
│   ├── __init__.py         # Flask application factory
│   ├── extensions.py       # Extension initializers
│   └── ...
│
├── .env                    # Environment variables (API keys, DB URL)
└── run.py                  # Entry point to start the Flask server
⚙️ Setup and InstallationFollow these steps to get the project running locally on your machine.PrerequisitesPython 3.9+A running MySQL database (or other SQL database)A Cohere API KeyBackend Setup# 1. Clone the repository
git clone [https://github.com/your-username/api-genius-backend.git](https://github.com/your-username/api-genius-backend.git)
cd api-genius-backend

# 2. Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Create the environment file
# Create a new file named .env in the root directory
touch .env
Now, open the .env file and add your configuration variables..env# Your Cohere API Key
COHERE_API_KEY="your_cohere_api_key_goes_here"

# Your database connection string
# Example for MySQL:
SQLALCHEMY_DATABASE_URI="mysql+pymysql://user:password@host:port/database_name"

# Flask secret key for JWT
SECRET_KEY="a_strong_and_random_secret_key"
▶️ Running the ApplicationOnce the setup is complete, you can run the application with a single command.# Make sure you are in the root /api-genius-backend folder
# and your virtual environment is activated.
python run.py

# The Flask server should now be running!
Open your browser and navigate to http://localhost:5000. The backend will serve the index.html file, and you can use the test interface to interact with all the API endpoints.🤝 ContributingContributions are welcome! If you'd like to help improve API Genius, please follow these steps:Fork the repository on GitHub.Clone your forked repository to your local machine.Create a new branch for your feature or bug fix.Make your changes and commit them with clear, descriptive messages.Push your changes to your forked repository.Open a Pull Request to the main repository, detailing the changes you've made.📄 LicenseThis project is licensed under the MIT License. See the LICENSE file for more details.
