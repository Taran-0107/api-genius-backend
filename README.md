# 🚀 API Genius: AI-Powered API Discovery and Integration Platform (Backend)

**API Genius** is a full-stack web application designed to streamline the process of discovering, understanding, and integrating third-party APIs.  
It leverages **Large Language Models (LLMs)** through Cohere to provide an intelligent, agentic workflow that can:  
- Actively find new APIs on the web  
- Recommend the best tools for a given task  
- Generate ready-to-use code snippets  

This repository contains the **Flask backend**, which powers the entire application and serves a simple HTML/Alpine.js frontend for testing and demonstration purposes.

---

## ✨ Key Features

- 🤖 **Agentic API Discovery** – Describe a need (e.g., *"an API for sending emails"*), and an autonomous agent will search the web, scrape documentation, and save relevant API information.  
- 🧠 **AI-Powered Recommendations** – Unsure which API to use? Describe your problem, and the AI Assistant will recommend the most suitable APIs.  
- 💻 **Instant Code Generation** – Get functional code snippets (Python, JavaScript, etc.) with imports and error handling.  
- 💬 **Community Q&A** – Ask questions, share solutions, and discuss API integrations.  
- 🔍 **Semantic Search** – Search by meaning, not just keywords, using vector embeddings.  
- 🔐 **Secure User Authentication** – JWT-based authentication for user login and protected endpoints.  

---

## 🛠️ Tech Stack

| Area       | Technology |
|------------|------------|
| **Backend** | Python, Flask, SQLAlchemy, Cohere (LLMs & Embeddings), LangChain, Selenium, JWT |
| **Database** | MySQL (or any SQLAlchemy-compatible DB such as PostgreSQL, SQLite) |
| **Frontend** | HTML, Tailwind CSS, Alpine.js (served via Flask) |

---

## 📂 Project Structure

```
/api-genius-backend/
├── /yourapi/               # Main Flask application package
│   ├── /routes/            # API endpoints (Blueprints)
│   ├── /services/          # Business logic (DB, AI, etc.)
│   ├── /templates/         # Test frontend (index.html)
│   ├── __init__.py         # Flask app factory
│   ├── extensions.py       # Extensions initialization
│   └── ...
│
├── .env                    # Environment variables (API keys, DB URL)
└── run.py                  # Flask server entry point
```

---

## ⚙️ Setup and Installation

### Prerequisites
- Python **3.9+**
- Running **MySQL database** (or PostgreSQL/SQLite)
- **Cohere API Key**

### Backend Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/api-genius-backend.git
cd api-genius-backend

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create environment file
touch .env
```

Inside `.env` add:

```env
# Cohere API Key
COHERE_API_KEY="your_cohere_api_key"

# Database connection string
# Example for MySQL:
SQLALCHEMY_DATABASE_URI="mysql+pymysql://user:password@host:port/database_name"

# Flask secret key
SECRET_KEY="a_strong_random_secret_key"
```

---

## ▶️ Running the Application

```bash
# Run from the project root
python run.py
```

Open your browser at 👉 [http://localhost:5000](http://localhost:5000)  
The backend will serve **index.html** and you can test all endpoints.



