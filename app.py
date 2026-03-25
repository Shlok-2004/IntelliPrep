from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import psycopg2
from psycopg2.extras import DictCursor
import os
import pandas as pd
import re
import tempfile
from dotenv import load_dotenv

load_dotenv()
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

import platform
import multiprocessing as mp

def _child_runner(_module_name, _func_name, _args):
    import importlib
    mod = importlib.import_module(_module_name)
    func = getattr(mod, _func_name)
    return func(*_args)

def run_isolated_eval(module_name, func_name, *args):
    """
    Executes heavy ML logic in isolated OS scopes.
    Windows -> uses fresh subprocess (avoids thread deadlocks).
    Linux -> uses 'fork' process pool (re-uses memory via Copy-On-Write, avoids Render OOM kills).
    """
    if platform.system() == "Windows":
        import subprocess, json, tempfile, uuid, os
        out_json = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.json")
        encoded_args = json.dumps(args)
        script = f"import sys, json; import importlib; mod = importlib.import_module('{module_name}'); func = getattr(mod, '{func_name}'); res = func(*json.loads(sys.argv[1])); json.dump(res, open(sys.argv[2], 'w'))"
        subprocess.run(["python", "-c", script, encoded_args, out_json], check=True)
        with open(out_json, 'r') as f:
            res = json.load(f)
        if os.path.exists(out_json):
            os.remove(out_json)
        return res
    else:
        ctx = mp.get_context('fork')
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            future = executor.submit(_child_runner, module_name, func_name, args)
            return future.result()


# =========================================================
# APP SETUP
# =========================================================
app = Flask(__name__)
app.secret_key = "interview_secret"
CORS(app)

# =========================================================
# POSTGRES CONFIG
# =========================================================
DB_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
    conn = psycopg2.connect(DB_URL, cursor_factory=DictCursor)
    return conn

class DummyMySQL:
    @property
    def connection(self):
        if not hasattr(g, 'db_conn'):
            g.db_conn = get_db_connection()
        return g.db_conn

from flask import g
mysql = DummyMySQL()

@app.teardown_appcontext
def close_connection(exception):
    conn = getattr(g, 'db_conn', None)
    if conn is not None:
        conn.close()


# =========================================================
# LOAD QUESTIONS DATASET (LAZY)
# =========================================================
_df = None

def get_df():
    global _df
    if _df is None:
        import pandas as pd
        import os
        print("DIAGNOSTIC LOG: Contents of /app: ", os.listdir(os.path.dirname(os.path.abspath(__file__))), flush=True)
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "questions.csv")
        _df = pd.read_csv(csv_path, encoding="latin1")
    return _df

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


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        new_password = request.form.get("new_password")

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT user_id FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            cursor.execute("UPDATE users SET password=%s WHERE email=%s", (new_password, email))
            mysql.connection.commit()
            cursor.close()
            return render_template("forgot_password.html", success="Password successfully updated! You can now login.")
        else:
            cursor.close()
            return render_template("forgot_password.html", error="Email address not found.")

    return render_template("forgot_password.html")


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

    # NEW: PROGRESS BY QUESTION TYPE (for Pie Chart)
    cursor.execute("""
        SELECT sq.question_type, AVG(er.final_score)
        FROM interview_sessions s
        JOIN session_questions sq ON s.session_id = sq.session_id
        JOIN user_answers ua ON sq.session_question_id = ua.session_question_id
        JOIN evaluation_results er ON ua.answer_id = er.answer_id
        WHERE s.user_id = %s
        GROUP BY sq.question_type
    """, (user_id,))
    question_type_data = cursor.fetchall()
    
    question_types = [row[0] for row in question_type_data] if question_type_data else ["None"]
    question_scores = [round(row[1], 1) for row in question_type_data] if question_type_data else [0]

    # RESUME SCORE (Latest)
    cursor.execute("""
        SELECT ats_score
        FROM resume_analysis
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
    """, (user_id,))
    resume_row = cursor.fetchone()
    resume_score = resume_row[0] if resume_row else 0

    # RESUME ANALYSIS CHART (Matched VS Missing from latest resume) 
    cursor.execute("""
        SELECT matched_skills, missing_skills
        FROM resume_analysis
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 1
    """, (user_id,))
    latest_resume = cursor.fetchone()
    
    matched_count = 0
    missing_count = 0
    if latest_resume:
        matched_str = latest_resume[0]
        missing_str = latest_resume[1]
        matched_count = len([s for s in matched_str.split(",") if s.strip()]) if matched_str else 0
        missing_count = len([s for s in missing_str.split(",") if s.strip()]) if missing_str else 0

    # RECENT ACTIVITY
    cursor.execute("""
        SELECT sq.question_type, (s.start_time + INTERVAL '5 hours 30 minutes'), er.final_score
        FROM interview_sessions s
        JOIN session_questions sq ON s.session_id = sq.session_id
        JOIN user_answers ua ON sq.session_question_id = ua.session_question_id
        JOIN evaluation_results er ON ua.answer_id = er.answer_id
        WHERE s.user_id = %s
        ORDER BY s.start_time DESC
        LIMIT 4
    """, (user_id,))
    recent_activities = cursor.fetchall()
    
    activities = [
        [row[0].capitalize() + " Question", row[1].strftime("%b %d, %Y") if hasattr(row[1], "strftime") else "Recently", round(row[2],1)]
        for row in recent_activities
    ]

    cursor.close()

    return render_template(
        "dashboard.html",
        user_name=user_name,
        total_interviews=total_interviews,
        avg_score=avg_score,
        resume_score=resume_score,
        question_types=question_types,
        question_scores=question_scores,
        matched_count=matched_count,
        missing_count=missing_count,
        activities=activities
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
    num_questions = int(data.get("num_questions", 5))
    session["job_role"] = job_role
    session["question_type"] = question_type


    # Filter dataset
    df_instance = get_df()
    filtered_df = df_instance[
        (df_instance["job_role"].str.strip().str.lower() == job_role.lower()) &
        (df_instance["question_type"].str.strip().str.lower() == question_type.lower())
    ].reset_index(drop=True)

    if filtered_df.empty:
        return jsonify({"error": "No questions found"}), 404

    filtered_df["difficulty"] = filtered_df["question"].apply(infer_difficulty)
    # ML Classification
    classified_indices = classify_questions(filtered_df)
    classified_indices = classified_indices[:num_questions]

    # Create Interview Session in DB
    cur = mysql.connection.cursor()
    cur.execute(
        "SELECT role_id FROM job_roles WHERE role_name=%s",
        (job_role,)
    )
    role_row = cur.fetchone()
    role_id = role_row[0] if role_row else None

    cur.execute(
        "INSERT INTO interview_sessions (user_id, role_id) VALUES (%s,%s) RETURNING session_id",
        (session["user_id"], role_id)
    )
    session["session_id"] = cur.fetchone()[0]
    mysql.connection.commit()
    cur.close()
    # Store first question in DB
    first_question_index = classified_indices[0]

    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO session_questions
        (session_id, dataset_question_id, question_type, difficulty, question_order)
        VALUES (%s,%s,%s,%s,%s) RETURNING session_question_id
    """, (
        session["session_id"],
        first_question_index,
        str(filtered_df.loc[first_question_index, "question_type"]).strip().upper(),
        filtered_df.loc[first_question_index, "difficulty"],
        1
    ))
    session["session_question_id"] = cur.fetchone()[0]
    mysql.connection.commit()
    cur.close()


    # Store in session
    session["questions"] = [int(i) for i in classified_indices]
    session["current_index"] = 0
    

    first_question = filtered_df.loc[classified_indices[0], "question"]

    response_data = {
        "question_number": 1,
        "question": first_question,
        "is_mcq": False
    }

    if question_type.lower() == "aptitude":
        q_text = str(filtered_df.loc[classified_indices[0], "question"])
        match = re.search(r'(.*?)\s*\([Aa]\)\s*(.*?)\s*\([Bb]\)\s*(.*?)\s*\([Cc]\)\s*(.*?)\s*\([Dd]\)\s*(.*)', q_text)
        if match:
            response_data["question"] = match.group(1).strip()
            response_data["is_mcq"] = True
            response_data["options"] = {
                "A": match.group(2).strip(),
                "B": match.group(3).strip(),
                "C": match.group(4).strip(),
                "D": match.group(5).strip()
            }

    return jsonify(response_data)



@app.route("/next-question", methods=["GET"])
def next_question():
    if "current_index" not in session or "questions" not in session:
        return jsonify({"error": "Session expired"}), 400
    
    session["current_index"] += 1
    idx = session["current_index"]

    questions = session["questions"]
    df_instance = get_df()
    filtered_df = df_instance[
    (df_instance["job_role"].str.strip().str.lower() == session["job_role"].lower()) &
    (df_instance["question_type"].str.strip().str.lower() == session["question_type"].lower())
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
        VALUES (%s,%s,%s,%s,%s) RETURNING session_question_id
    """, (
        session["session_id"],
        question_index,
        str(filtered_df.loc[question_index, "question_type"]).strip().upper(),
        filtered_df.loc[question_index, "difficulty"],
        idx + 1
    ))
    session["session_question_id"] = cur.fetchone()[0]
    mysql.connection.commit()
    cur.close()

    response_data = {
        "question_number": idx + 1,
        "question": question,
        "is_mcq": False
    }

    if session["question_type"].lower() == "aptitude":
        q_text = str(filtered_df.loc[question_index, "question"])
        match = re.search(r'(.*?)\s*\([Aa]\)\s*(.*?)\s*\([Bb]\)\s*(.*?)\s*\([Cc]\)\s*(.*?)\s*\([Dd]\)\s*(.*)', q_text)
        if match:
            response_data["question"] = match.group(1).strip()
            response_data["is_mcq"] = True
            response_data["options"] = {
                "A": match.group(2).strip(),
                "B": match.group(3).strip(),
                "C": match.group(4).strip(),
                "D": match.group(5).strip()
            }

    return jsonify(response_data)

@app.route("/evaluate", methods=["POST"])
def evaluate():

    if "session_question_id" not in session:
        return jsonify({"error": "Session expired"}), 400

    user_answer = request.form.get("answer", "")

    df_instance = get_df()
    filtered_df = df_instance[
        (df_instance["job_role"].str.strip().str.lower() == session["job_role"].lower()) &
        (df_instance["question_type"].str.strip().str.lower() == session["question_type"].lower())
    ].reset_index(drop=True)

    question_idx = session["questions"][session["current_index"]]
    ideal_answer = filtered_df.loc[question_idx, "answer"]

    # -----------------------------
    # EVALUATION
    # -----------------------------
    hr_score = 0

    if session["question_type"].lower() == "aptitude":
        q_user_ans = str(user_answer).strip().lower()
        q_ideal_ans_clean = str(ideal_answer).strip().lower().replace("(", "").replace(")", "")
        correct = q_user_ans == q_ideal_ans_clean
        text_score_percent = 100 if correct else 0
        text_result = {
            "final_score": 1.0 if correct else 0.0,
            "semantic_similarity": 1.0 if correct else 0.0,
            "keyword_score": 1.0 if correct else 0.0,
            "feedback": "Correct Answer! Great job." if correct else f"Incorrect. The correct answer was Option {str(ideal_answer).upper()}."
        }
    else:
        # 1. PROCESS VIDEO IN OS-SHARED SUBPROCESS
        if session["question_type"].lower() == "hr" and "video" in request.files:
            video_file = request.files["video"]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                video_path = tmp.name
                video_file.save(video_path)
    
            hr_result = run_isolated_eval("hr_analysis", "run_hr_video_analysis", video_path)
                
            os.remove(video_path)
            hr_score = hr_result["final_hr_score"]

        # 2. PROCESS TEXT IN OS-SHARED SUBPROCESS
        text_result = run_isolated_eval("question_classification_evalution", "evaluate_answer", str(user_answer), str(ideal_answer))
            
        text_score_percent = round(text_result["final_score"] * 100, 2)

    # -----------------------------
    # COMBINE SCORES
    # -----------------------------
    if session["question_type"].lower() == "hr":
        final_score = round(float((text_score_percent * 0.6) + (hr_score * 0.4)), 2)
    else:
        final_score = text_score_percent

    # -----------------------------
    # STORE IN DB (0–100 SCALE)
    # -----------------------------
    cur = mysql.connection.cursor()

    cur.execute("""
        INSERT INTO user_answers
        (session_question_id, answer_text)
        VALUES (%s,%s) RETURNING answer_id
    """, (
        session["session_question_id"],
        user_answer
    ))
    answer_id = cur.fetchone()[0]
    mysql.connection.commit()

    cur.execute("""
        INSERT INTO evaluation_results
        (answer_id, final_score, semantic_similarity, keyword_score, feedback)
        VALUES (%s,%s,%s,%s,%s)
    """, (
        answer_id,
        final_score,  # already percentage
        round(float(text_result["semantic_similarity"]) * 100, 2) if "semantic_similarity" in text_result else 100,
        round(float(text_result["keyword_score"]) * 100, 2) if "keyword_score" in text_result else 100,
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
        result = run_isolated_eval("hr_analysis", "run_hr_video_analysis", video_path)
        os.remove(video_path)

        cur = mysql.connection.cursor()

        cur.execute("""
            INSERT INTO user_answers(session_question_id, answer_text)
            VALUES(%s,%s) RETURNING answer_id
        """, (
            session.get("session_question_id", None),
            "HR Video Response"
        ))
        answer_id = cur.fetchone()[0]
        mysql.connection.commit()

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
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    cursor = mysql.connection.cursor()

    # 1. RADAR CHART (Average score by category)
    cursor.execute("""
        SELECT sq.question_type, AVG(er.final_score)
        FROM interview_sessions s
        JOIN session_questions sq ON s.session_id = sq.session_id
        JOIN user_answers ua ON sq.session_question_id = ua.session_question_id
        JOIN evaluation_results er ON ua.answer_id = er.answer_id
        WHERE s.user_id = %s
        GROUP BY sq.question_type
    """, (user_id,))
    radar_data = cursor.fetchall()
    
    radar_dict = {"HR": 0, "Aptitude": 0, "Technical": 0}
    for row in radar_data:
        val = round(float(row[1]), 1) if row[1] else 0
        q_type = str(row[0]).strip().upper()
        if q_type == "HR":
            radar_dict["HR"] = val
        elif q_type == "TECHNICAL":
            radar_dict["Technical"] = val
        elif q_type == "APTITUDE":
            radar_dict["Aptitude"] = val

    # 2. TREND CHART (Average score per day)
    cursor.execute("""
        SELECT DATE(s.start_time + INTERVAL '5 hours 30 minutes') as session_date, AVG(er.final_score)
        FROM interview_sessions s
        JOIN session_questions sq ON s.session_id = sq.session_id
        JOIN user_answers ua ON sq.session_question_id = ua.session_question_id
        JOIN evaluation_results er ON ua.answer_id = er.answer_id
        WHERE s.user_id = %s
        GROUP BY session_date
        ORDER BY session_date ASC
    """, (user_id,))
    trend_data = cursor.fetchall()
    
    trend_dates = [row[0].strftime("%b %d") if hasattr(row[0], "strftime") else str(row[0]) for row in trend_data]
    trend_scores = [round(row[1], 1) for row in trend_data]

    # 3. DETAILED HISTORY TABLE
    cursor.execute("""
        SELECT (s.start_time + INTERVAL '5 hours 30 minutes'), jr.role_name, sq.question_type, er.final_score, er.feedback
        FROM interview_sessions s
        JOIN job_roles jr ON s.role_id = jr.role_id
        JOIN session_questions sq ON s.session_id = sq.session_id
        JOIN user_answers ua ON sq.session_question_id = ua.session_question_id
        JOIN evaluation_results er ON ua.answer_id = er.answer_id
        WHERE s.user_id = %s
        ORDER BY s.start_time DESC
    """, (user_id,))
    history_rows = cursor.fetchall()
    
    history = [
        {
            "date": row[0].strftime("%b %d, %Y %I:%M %p") if hasattr(row[0], "strftime") else str(row[0]),
            "role": row[1],
            "type": "HR" if row[2].upper() == "HR" else row[2].capitalize(),
            "score": round(row[3], 1),
            "feedback": row[4]
        }
        for row in history_rows
    ]
    cursor.close()

    return render_template(
        "progress.html",
        radar_data=radar_dict,
        trend_dates=trend_dates,
        trend_scores=trend_scores,
        history=history
    )

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
