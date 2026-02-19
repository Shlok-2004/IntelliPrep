from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_mysqldb import MySQL
import pandas as pd
import os
import tempfile
from intelliprep_resume_engine.core.resume_parser import extract_resume_text, clean_text
from intelliprep_resume_engine.core.role_profiles import ROLE_SKILLS, ROLE_CRITICAL_SKILLS
from intelliprep_resume_engine.core.skill_extractor import extract_skills
from intelliprep_resume_engine.core.scorer import (
    skill_match_score,
    semantic_similarity,
    final_ats_score,
    critical_skill_gaps
)
from intelliprep_resume_engine.core.feedback import generate_feedback
from question_classification_evalution import classify_questions, evaluate_answer
from hr_analysis import run_hr_video_analysis


# =========================================================
# APP SETUP
# =========================================================
app = Flask(__name__)
app.secret_key = "interview_secret"
CORS(app)

# =========================================================
# MYSQL CONFIG
# =========================================================
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "shlok19"
app.config["MYSQL_DB"] = "interview_prep_db"

mysql = MySQL(app)

# =========================================================
# LOAD QUESTIONS DATASET
# =========================================================
df = pd.read_csv("questions.csv", encoding="latin1")
from question_classification_evalution import infer_difficulty

# =========================================================
# LANDING
# =========================================================
@app.route("/")
def landing():
    return render_template("landing.html")

# =========================================================
# AUTH SYSTEM
# =========================================================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        cursor = mysql.connection.cursor()
        cursor.execute(
            "SELECT user_id, full_name FROM users WHERE email=%s AND password=%s",
            (email, password)
        )
        user = cursor.fetchone()
        cursor.close()

        if user:
            session["user_id"] = user[0]
            session["user_name"] = user[1]
            return redirect(url_for("dashboard"))
        else:
            return "Invalid email or password"

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        fullname = request.form.get("fullname")
        email = request.form.get("email")
        password = request.form.get("password")

        cursor = mysql.connection.cursor()

        cursor.execute("SELECT user_id FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            cursor.close()
            return "Email already exists"

        cursor.execute(
            "INSERT INTO users(full_name,email,password) VALUES(%s,%s,%s)",
            (fullname, email, password)
        )
        mysql.connection.commit()
        cursor.close()

        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))

# =========================================================
# DASHBOARD (FULL DYNAMIC DB VERSION)
# =========================================================
@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    cursor = mysql.connection.cursor()

    # USER NAME
    cursor.execute("SELECT full_name FROM users WHERE user_id=%s", (user_id,))
    user_row = cursor.fetchone()
    user_name = user_row[0] if user_row else "User"

    # TOTAL INTERVIEWS
    cursor.execute("SELECT COUNT(*) FROM interview_sessions WHERE user_id=%s", (user_id,))
    total_interviews = cursor.fetchone()[0] or 0

    # AVG INTERVIEW SCORE (already stored in percentage scale)
    cursor.execute("""
        SELECT AVG(er.final_score)
        FROM interview_sessions s
        JOIN session_questions sq ON s.session_id = sq.session_id
        JOIN user_answers ua ON sq.session_question_id = ua.session_question_id
        JOIN evaluation_results er ON ua.answer_id = er.answer_id
        WHERE s.user_id = %s
    """, (user_id,))
    avg_score = cursor.fetchone()[0]
    avg_score = round(avg_score, 1) if avg_score else 0

    # INTERVIEW TREND GRAPH
    cursor.execute("""
        SELECT s.session_id, AVG(er.final_score)
        FROM interview_sessions s
        JOIN session_questions sq ON s.session_id = sq.session_id
        JOIN user_answers ua ON sq.session_question_id = ua.session_question_id
        JOIN evaluation_results er ON ua.answer_id = er.answer_id
        WHERE s.user_id = %s
        GROUP BY s.session_id
        ORDER BY s.session_id
    """, (user_id,))
    trend_data = cursor.fetchall()

    progress = [
        [i + 1, round(score, 1)]
        for i, (_, score) in enumerate(trend_data)
        if score is not None
    ]

    # ✅ RESTORED RESUME SCORE
    cursor.execute("""
        SELECT ats_score
        FROM resume_analysis
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
    """, (user_id,))
    resume_row = cursor.fetchone()
    resume_score = resume_row[0] if resume_row else 0

    cursor.close()

    return render_template(
        "dashboard.html",
        user_name=user_name,
        total_interviews=total_interviews,
        avg_score=avg_score,
        resume_score=resume_score,
        progress=progress,
        skill_percent=[],
        activities=[]
    )


# =========================================================
# INTERVIEW START
# =========================================================
@app.route("/start-interview", methods=["POST"])
def start_interview():

    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    job_role = data["job_role"].strip()
    question_type = data["question_type"].strip()
    session["job_role"] = job_role
    session["question_type"] = question_type


    # Filter dataset
    filtered_df = df[
        (df["job_role"].str.strip().str.lower() == job_role.lower()) &
        (df["question_type"].str.strip().str.lower() == question_type.lower())
    ].reset_index(drop=True)

    if filtered_df.empty:
        return jsonify({"error": "No questions found"}), 404

    filtered_df["difficulty"] = filtered_df["question"].apply(infer_difficulty)
    # ML Classification
    classified_indices = classify_questions(filtered_df)

    # Create Interview Session in DB
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT role_id FROM job_roles WHERE role_name=%s",
        (job_role,)
    )
    role_row = cur.fetchone()
    role_id = role_row[0] if role_row else None

    cur.execute(
        "INSERT INTO interview_sessions (user_id, role_id) VALUES (%s,%s)",
        (session["user_id"], role_id)
    )
    mysql.connection.commit()

    session["session_id"] = int(cur.lastrowid)
    cur.close()
    # Store first question in DB
    first_question_index = classified_indices[0]

    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO session_questions
        (session_id, dataset_question_id, question_type, difficulty, question_order)
        VALUES (%s,%s,%s,%s,%s)
    """, (
        session["session_id"],
        first_question_index,
        filtered_df.loc[first_question_index, "question_type"],
        filtered_df.loc[first_question_index, "difficulty"],
        1
    ))
    mysql.connection.commit()

    session["session_question_id"] = cur.lastrowid
    cur.close()


    # Store in session
    session["questions"] = [int(i) for i in classified_indices]
    session["current_index"] = 0
    

    first_question = filtered_df.loc[classified_indices[0], "question"]

    return jsonify({
        "question_number": 1,
        "question": first_question
    })



@app.route("/next-question", methods=["GET"])
def next_question():
    if "current_index" not in session or "questions" not in session:
        return jsonify({"error": "Session expired"}), 400
    
    session["current_index"] += 1
    idx = session["current_index"]

    questions = session["questions"]
    filtered_df = df[
    (df["job_role"].str.strip().str.lower() == session["job_role"].lower()) &
    (df["question_type"].str.strip().str.lower() == session["question_type"].lower())
    ].reset_index(drop=True)

    filtered_df["difficulty"] = filtered_df["question"].apply(infer_difficulty)


    if idx >= len(questions):
        return jsonify({"message": "Interview completed"})

    question_index = questions[idx]
    question = filtered_df.loc[question_index, "question"]

    # Store question in DB
    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO session_questions
        (session_id, dataset_question_id, question_type, difficulty, question_order)
        VALUES (%s,%s,%s,%s,%s)
    """, (
        session["session_id"],
        question_index,
        filtered_df.loc[question_index, "question_type"],
        filtered_df.loc[question_index, "difficulty"],
        idx + 1
    ))
    mysql.connection.commit()

    session["session_question_id"] = cur.lastrowid
    cur.close()

    return jsonify({
        "question_number": idx + 1,
        "question": question
    })

@app.route("/evaluate", methods=["POST"])
def evaluate():

    if "session_question_id" not in session:
        return jsonify({"error": "Session expired"}), 400

    user_answer = request.form.get("answer", "")

    filtered_df = df[
        (df["job_role"].str.lower() == session["job_role"].lower()) &
        (df["question_type"].str.lower() == session["question_type"].lower())
    ].reset_index(drop=True)

    question_idx = session["questions"][session["current_index"]]
    ideal_answer = filtered_df.loc[question_idx, "answer"]

    # -----------------------------
    # TEXT MODEL
    # -----------------------------
    text_result = evaluate_answer(user_answer, ideal_answer)
    text_score_percent = round(text_result["final_score"] * 100, 2)

    # -----------------------------
    # HR MODEL (if HR selected)
    # -----------------------------
    hr_score = 0

    if session["question_type"].lower() == "hr" and "video" in request.files:
        video_file = request.files["video"]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            video_path = tmp.name
            video_file.save(video_path)

        hr_result = run_hr_video_analysis(video_path)
        os.remove(video_path)

        hr_score = hr_result["final_hr_score"]

    # -----------------------------
    # COMBINE SCORES
    # -----------------------------
    if session["question_type"].lower() == "hr":
        final_score = round((text_score_percent * 0.6) + (hr_score * 0.4), 2)
    else:
        final_score = text_score_percent

    # -----------------------------
    # STORE IN DB (0–100 SCALE)
    # -----------------------------
    cur = mysql.connection.cursor()

    cur.execute("""
        INSERT INTO user_answers
        (session_question_id, answer_text)
        VALUES (%s,%s)
    """, (
        session["session_question_id"],
        user_answer
    ))
    mysql.connection.commit()

    answer_id = cur.lastrowid

    cur.execute("""
        INSERT INTO evaluation_results
        (answer_id, final_score, semantic_similarity, keyword_score, feedback)
        VALUES (%s,%s,%s,%s,%s)
    """, (
        answer_id,
        final_score,  # already percentage
        round(text_result["semantic_similarity"] * 100, 2),
        round(text_result["keyword_score"] * 100, 2),
        text_result["feedback"]
    ))
    mysql.connection.commit()
    cur.close()

    return jsonify({
        "final_score": final_score,
        "text_score": text_score_percent,
        "hr_score": hr_score,
        "feedback": text_result["feedback"]
    })




# =========================================================
# HR VIDEO ANALYSIS (REAL MODEL INTEGRATION)
# =========================================================
@app.route("/analyze-hr-video", methods=["POST"])
def analyze_hr_video():

    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    if "session_question_id" not in session:
        return jsonify({"error": "No active HR question"}), 400

    video_file = request.files["video"]

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        video_file.save(video_path)

    try:
        result = run_hr_video_analysis(video_path)

        os.remove(video_path)

        cur = mysql.connection.cursor()

        cur.execute("""
            INSERT INTO user_answers(session_question_id, answer_text)
            VALUES(%s,%s)
        """, (
            session.get("session_question_id", None),
            "HR Video Response"
        ))
        mysql.connection.commit()

        answer_id = cur.lastrowid

        cur.execute("""
            INSERT INTO evaluation_results
            (answer_id, final_score, semantic_similarity, keyword_score, feedback)
            VALUES(%s,%s,%s,%s,%s)
        """, (
            answer_id,
            result["final_hr_score"],
            result["emotion_score"],
            result["eye_contact_score"],
            f"Confidence: {result['confidence_score']} | "
            f"Dominant Emotion: {result['dominant_emotion']}"
        ))
        mysql.connection.commit()
        cur.close()

        return jsonify(result)

    except Exception as e:
        os.remove(video_path)
        return jsonify({"error": str(e)}), 500


# =========================================================
# RESUME ANALYSIS
# =========================================================
@app.route("/analyze-resume", methods=["POST"])
def analyze_resume():

    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    resume_file = request.files["resume"]
    job_description = request.form.get("job_description", "")

    # Save file temporarily
    filename = resume_file.filename
    extension = filename.split('.')[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
        resume_path = tmp.name
        resume_file.save(resume_path)


    # ================================
    # 🔹 REAL ML ENGINE STARTS HERE
    # ================================

    # Extract text
    resume_text = extract_resume_text(resume_path)
    resume_text = clean_text(resume_text)
    jd_text = clean_text(job_description)

    os.remove(resume_path)

    # Select role (you can improve this later)
    role = "data_scientist"

    role_skills = ROLE_SKILLS.get(role, [])
    critical_skills = ROLE_CRITICAL_SKILLS.get(role, [])

    # Skill extraction
    found_skills = extract_skills(resume_text, role_skills)

    # Scoring
    skill_score, matched, missing = skill_match_score(found_skills, role_skills)
    semantic_score = semantic_similarity(resume_text, jd_text)
    ats_score = final_ats_score(skill_score, semantic_score)

    _, missing_critical = critical_skill_gaps(found_skills, critical_skills)

    feedback = generate_feedback(
        role,
        skill_score,
        semantic_score,
        matched,
        missing,
        missing_critical
    )

    # ================================
    # 🔹 STORE IN DATABASE
    # ================================
    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO resume_analysis
        (user_id, ats_score, matched_skills, missing_skills, feedback)
        VALUES (%s,%s,%s,%s,%s)
    """, (
        session["user_id"],
        ats_score,
        ",".join(matched),
        ",".join(missing),
        " | ".join(feedback)
    ))
    mysql.connection.commit()
    cur.close()

    # ================================
    # 🔹 RETURN REAL DATA TO JS
    # ================================
    return jsonify({
        "score": round(ats_score, 1),
        "matched": matched,
        "missing": missing,
        "feedback": feedback
    })

# =========================================================
# STATIC PAGES (FIXES BuildError)
# =========================================================
@app.route("/progress")
def progress():
    return render_template("progress.html")

@app.route("/interview")
def interview():
    return render_template("interview.html")

@app.route("/resume")
def resume():
    return render_template("resume.html")

@app.route("/resources")
def resources():
    return render_template("resources.html")

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
