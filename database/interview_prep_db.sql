
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    is_suspended BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE job_roles (
    role_id SERIAL PRIMARY KEY,
    role_name VARCHAR(50) NOT NULL
);

CREATE TABLE interview_sessions (
    session_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    role_id INT NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (role_id) REFERENCES job_roles(role_id)
);

CREATE TABLE session_questions (
    session_question_id SERIAL PRIMARY KEY,
    session_id INT NOT NULL,
    dataset_question_id INT NOT NULL,
    question_type VARCHAR(20) CHECK (question_type IN ('HR','APTITUDE','TECHNICAL')) NOT NULL,
    difficulty VARCHAR(20) CHECK (difficulty IN ('Easy','Medium','Hard')) NOT NULL,
    question_order INT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES interview_sessions(session_id)
);

CREATE TABLE user_answers (
    answer_id SERIAL PRIMARY KEY,
    session_question_id INT NOT NULL,
    answer_text TEXT NOT NULL,
    answer_time_seconds INT,
    FOREIGN KEY (session_question_id) REFERENCES session_questions(session_question_id)
);

CREATE TABLE evaluation_results (
    evaluation_id SERIAL PRIMARY KEY,
    answer_id INT NOT NULL,
    final_score FLOAT,
    semantic_similarity FLOAT,
    keyword_score FLOAT,
    feedback TEXT,
    FOREIGN KEY (answer_id) REFERENCES user_answers(answer_id)
);

CREATE TABLE resume_analysis (
    resume_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    ats_score FLOAT,
    matched_skills TEXT,
    missing_skills TEXT,
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

INSERT INTO job_roles (role_name) VALUES
('Data Scientist'),
('Cybersecurity Engineer'),
('Web Developer');
