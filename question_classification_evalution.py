# ======================================================
# QUESTION CLASSIFICATION + EVALUATION MODULE
# Flask-Safe | Production-Ready
# ======================================================

import random
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ======================================================
# LOAD BERT MODEL (Loads Once)
# ======================================================
bert_model = SentenceTransformer('all-MiniLM-L6-v2')


# ======================================================
# DIFFICULTY INFERENCE
# ======================================================
def infer_difficulty(question):
    q = question.lower()
    length = len(q.split())

    if any(k in q for k in ["compare", "difference", "why", "how",
                            "tradeoff", "counterfactual", "shap"]):
        return "Hard"
    elif length > 8 or any(k in q for k in ["explain", "describe", "define"]):
        return "Medium"
    else:
        return "Easy"


# ======================================================
# QUESTION CLASSIFICATION FUNCTION
# ======================================================
def classify_questions(filtered_df, max_questions=5):

    if filtered_df.empty:
        return []

    filtered_df = filtered_df.copy()
    filtered_df["difficulty"] = filtered_df["question"].apply(infer_difficulty)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(filtered_df["question"])

    classified_indices = []
    difficulty_order = ["Easy", "Medium", "Hard"]

    for level in difficulty_order:

        level_indices = filtered_df[
            filtered_df["difficulty"] == level
        ].index.tolist()

        if not level_indices:
            continue

        current_index = random.choice(level_indices)
        classified_indices.append(current_index)

        while True:

            if len(classified_indices) >= max_questions:
                break

            similarities = cosine_similarity(
                tfidf_matrix[current_index],
                tfidf_matrix
            ).flatten()

            for i in range(len(similarities)):
                if i in classified_indices or filtered_df.loc[i, "difficulty"] != level:
                    similarities[i] = -1

            next_index = similarities.argmax()

            if similarities[next_index] == -1:
                break

            classified_indices.append(next_index)
            current_index = next_index

    return classified_indices[:max_questions]


# ======================================================
# KEY TERM EXTRACTION
# ======================================================
def extract_key_terms(text, top_n=6):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    keywords = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return list(dict.fromkeys(keywords))[:top_n]


# ======================================================
# IMPROVEMENT SUGGESTIONS
# ======================================================
def generate_improvement_suggestions(user_answer, ideal_answer, missing_terms):
    suggestions = []

    if missing_terms:
        suggestions.append(
            "Try including key concepts such as: " +
            ", ".join(missing_terms[:3])
        )

    if len(user_answer.split()) < len(ideal_answer.split()) * 0.5:
        suggestions.append(
            "Your answer is brief. Consider adding more explanation."
        )

    if "example" not in user_answer.lower():
        suggestions.append(
            "Adding a simple example could strengthen your answer."
        )

    if "impact" not in user_answer.lower() and "effect" not in user_answer.lower():
        suggestions.append(
            "You may also explain the impact or consequence of this concept."
        )

    return suggestions


# ======================================================
# ANSWER EVALUATION FUNCTION
# ======================================================
def evaluate_answer(user_answer, ideal_answer):

    user_answer = user_answer.strip()
    ideal_answer = ideal_answer.strip()

    # -----------------------------
    # MCQ Handling
    # -----------------------------
    if len(user_answer.split()) <= 1 and len(ideal_answer.split()) <= 1:
        if user_answer.lower() == ideal_answer.lower():
            return {
                "final_score": 1.0,
                "semantic_similarity": 1.0,
                "keyword_score": 1.0,
                "feedback": "Correct answer.",
                "improvement_suggestions": []
            }
        else:
            return {
                "final_score": 0.0,
                "semantic_similarity": 0.0,
                "keyword_score": 0.0,
                "feedback": "Incorrect answer. Review the concept and try again.",
                "improvement_suggestions": []
            }

    # -----------------------------
    # BERT Semantic Similarity
    # -----------------------------
    embeddings = bert_model.encode(
        [ideal_answer, user_answer],
        convert_to_numpy=True
    )

    semantic_similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    semantic_similarity = round(float(semantic_similarity), 2)

    # -----------------------------
    # Keyword Coverage
    # -----------------------------
    key_terms = extract_key_terms(ideal_answer)
    missing_terms = [k for k in key_terms if k not in user_answer.lower()]

    keyword_score = (
        1 - (len(missing_terms) / len(key_terms))
        if key_terms else 1
    )

    keyword_score = round(keyword_score, 2)

    # -----------------------------
    # Length Score (Answer Depth)
    # -----------------------------
    ideal_len = len(ideal_answer.split())
    user_len = len(user_answer.split())

    length_ratio = min(user_len / ideal_len, 1.0)
    length_score = round(length_ratio, 2)

    # -----------------------------
    # Weighted Final Score
    # -----------------------------
    final_score = (
        (semantic_similarity * 0.6) +
        (keyword_score * 0.3) +
        (length_score * 0.1)
    )

    final_score = round(final_score, 2)

    # -----------------------------
    # Feedback Logic
    # -----------------------------
    if final_score >= 0.85:
        feedback = "Outstanding answer. Strong conceptual clarity and coverage."

    elif final_score >= 0.70:
        feedback = "Good answer. You understand the concept well but can improve depth."

    elif final_score >= 0.50:
        feedback = "Average answer. Some important concepts are missing."

    else:
        feedback = "Weak answer. Focus on explaining core ideas clearly."

    # -----------------------------
    # Improvement Suggestions
    # -----------------------------
    improvement_suggestions = generate_improvement_suggestions(
        user_answer,
        ideal_answer,
        missing_terms
    )

    return {
        "final_score": final_score,
        "semantic_similarity": semantic_similarity,
        "keyword_score": keyword_score,
        "feedback": feedback,
        "improvement_suggestions": improvement_suggestions
    }
