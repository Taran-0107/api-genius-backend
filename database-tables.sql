-- Enable InnoDB for FK support
CREATE DATABASE api_app;
USE api_app;

-- USERS
CREATE TABLE users (
    id CHAR(36) PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role ENUM('admin', 'user', 'moderator') DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- APIS
CREATE TABLE apis (
    id CHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    base_url TEXT,
    category VARCHAR(100),
    description TEXT,
    homepage_url TEXT,
    last_known_version VARCHAR(50),
    docs_url TEXT,
    last_fetched TIMESTAMP NULL
);

-- API VERSIONS
CREATE TABLE api_versions (
    id CHAR(36) PRIMARY KEY,
    api_id CHAR(36) NOT NULL,
    version VARCHAR(50) NOT NULL,
    spec_url TEXT,
    release_date DATE,
    changelog TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (api_id) REFERENCES apis(id) ON DELETE CASCADE
);

-- API CHANGES
CREATE TABLE api_changes (
    id CHAR(36) PRIMARY KEY,
    api_version_old CHAR(36) NOT NULL,
    api_version_new CHAR(36) NOT NULL,
    change_type ENUM('deprecated', 'added', 'modified') NOT NULL,
    affected_endpoint TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (api_version_old) REFERENCES api_versions(id) ON DELETE CASCADE,
    FOREIGN KEY (api_version_new) REFERENCES api_versions(id) ON DELETE CASCADE
);

-- QUESTIONS
CREATE TABLE questions (
    id CHAR(36) PRIMARY KEY,
    user_id CHAR(36) NOT NULL,
    api_id CHAR(36),
    title TEXT NOT NULL,
    body_md TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (api_id) REFERENCES apis(id) ON DELETE SET NULL
);

-- ANSWERS
CREATE TABLE answers (
    id CHAR(36) PRIMARY KEY,
    question_id CHAR(36) NOT NULL,
    user_id CHAR(36) NOT NULL,
    body_md TEXT NOT NULL,
    is_accepted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- VOTES
CREATE TABLE votes (
    id CHAR(36) PRIMARY KEY,
    user_id CHAR(36) NOT NULL,
    entity_type ENUM('question', 'answer') NOT NULL,
    entity_id CHAR(36) NOT NULL,
    vote_type ENUM('up', 'down') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- API KEYS META (Metadata only - actual keys stored locally)
CREATE TABLE api_keys_meta (
    id CHAR(36) PRIMARY KEY,
    user_id CHAR(36) NOT NULL,
    api_id CHAR(36) NOT NULL,
    label TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (api_id) REFERENCES apis(id) ON DELETE CASCADE
);

-- API REVIEWS
CREATE TABLE api_reviews (
    id CHAR(36) PRIMARY KEY,
    api_id CHAR(36) NOT NULL,
    user_id CHAR(36) NOT NULL,
    title TEXT,
    body_md TEXT,
    sentiment_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (api_id) REFERENCES apis(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- API RATINGS
CREATE TABLE api_ratings (
    id CHAR(36) PRIMARY KEY,
    api_id CHAR(36) NOT NULL,
    user_id CHAR(36) NOT NULL,
    latency_score SMALLINT,
    ease_of_use SMALLINT,
    docs_quality SMALLINT,
    cost_efficiency SMALLINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (api_id) REFERENCES apis(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- EMBEDDINGS (requires MySQL 8.0+; storing as JSON or blob for vectors)
CREATE TABLE embeddings (
    id CHAR(36) PRIMARY KEY,
    entity_type ENUM('api_doc', 'question', 'answer', 'review') NOT NULL,
    entity_id CHAR(36) NOT NULL,
    vector JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for faster lookups
CREATE INDEX idx_questions_api_id ON questions(api_id);
CREATE INDEX idx_questions_user_id ON questions(user_id);
CREATE INDEX idx_answers_question_id ON answers(question_id);
CREATE INDEX idx_votes_entity ON votes(entity_type, entity_id);
CREATE INDEX idx_embeddings_entity ON embeddings(entity_type, entity_id);
