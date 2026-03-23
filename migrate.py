import re
import os

app_path = r"e:\intelliprep\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace imports
content = content.replace("from flask_mysqldb import MySQL", "import psycopg2\nfrom psycopg2.extras import DictCursor\nimport os")

# Replace configs and mysql = MySQL(app)
config_pattern = r'# =========================================================\n# MYSQL CONFIG\n# =========================================================\napp\.config\["MYSQL_HOST"\].*?mysql = MySQL\(app\)'
pg_config = """# =========================================================
# POSTGRES CONFIG
# =========================================================
DB_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
    conn = psycopg2.connect(DB_URL)
    return conn"""

content = re.sub(config_pattern, pg_config, content, flags=re.DOTALL)

# Replace cursor = mysql.connection.cursor() with conn = get_db_connection() \n cursor = conn.cursor()
# We also have to handle commitments: mysql.connection.commit() -> conn.commit()
# It's better to just write a simple wrapper class that mimics mysql.connection

wrapper = """
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
"""
content = content.replace("def get_db_connection():\n    conn = psycopg2.connect(DB_URL)\n    return conn", "def get_db_connection():\n    conn = psycopg2.connect(DB_URL, cursor_factory=DictCursor)\n    return conn\n" + wrapper)

# Now what about cur.lastrowid?
# Psycopg2 doesn't have lastrowid. We need to add ' RETURNING <id_col>' to INSERT queries.
# Let's manually replace the known INSERT queries using lastrowid.

# 1. sessions
s1 = """    cur.execute(
        "INSERT INTO interview_sessions (user_id, role_id) VALUES (%s,%s)",
        (session["user_id"], role_id)
    )
    mysql.connection.commit()

    session["session_id"] = int(cur.lastrowid)"""
r1 = """    cur.execute(
        "INSERT INTO interview_sessions (user_id, role_id) VALUES (%s,%s) RETURNING session_id",
        (session["user_id"], role_id)
    )
    session["session_id"] = cur.fetchone()[0]
    mysql.connection.commit()"""
content = content.replace(s1, r1)

# 2. session_questions 1
s2 = """    cur.execute(\"\"\"
        INSERT INTO session_questions
        (session_id, dataset_question_id, question_type, difficulty, question_order)
        VALUES (%s,%s,%s,%s,%s)
    \"\"\", (
        session["session_id"],
        first_question_index,
        filtered_df.loc[first_question_index, "question_type"],
        filtered_df.loc[first_question_index, "difficulty"],
        1
    ))
    mysql.connection.commit()

    session["session_question_id"] = cur.lastrowid"""
r2 = """    cur.execute(\"\"\"
        INSERT INTO session_questions
        (session_id, dataset_question_id, question_type, difficulty, question_order)
        VALUES (%s,%s,%s,%s,%s) RETURNING session_question_id
    \"\"\", (
        session["session_id"],
        first_question_index,
        filtered_df.loc[first_question_index, "question_type"],
        filtered_df.loc[first_question_index, "difficulty"],
        1
    ))
    session["session_question_id"] = cur.fetchone()[0]
    mysql.connection.commit()"""
content = content.replace(s2, r2)

# 3. session_questions 2
s3 = """    cur.execute(\"\"\"
        INSERT INTO session_questions
        (session_id, dataset_question_id, question_type, difficulty, question_order)
        VALUES (%s,%s,%s,%s,%s)
    \"\"\", (
        session["session_id"],
        question_index,
        filtered_df.loc[question_index, "question_type"],
        filtered_df.loc[question_index, "difficulty"],
        idx + 1
    ))
    mysql.connection.commit()

    session["session_question_id"] = cur.lastrowid"""
r3 = """    cur.execute(\"\"\"
        INSERT INTO session_questions
        (session_id, dataset_question_id, question_type, difficulty, question_order)
        VALUES (%s,%s,%s,%s,%s) RETURNING session_question_id
    \"\"\", (
        session["session_id"],
        question_index,
        filtered_df.loc[question_index, "question_type"],
        filtered_df.loc[question_index, "difficulty"],
        idx + 1
    ))
    session["session_question_id"] = cur.fetchone()[0]
    mysql.connection.commit()"""
content = content.replace(s3, r3)

# 4. user_answers 1
s4 = """    cur.execute(\"\"\"
        INSERT INTO user_answers
        (session_question_id, answer_text)
        VALUES (%s,%s)
    \"\"\", (
        session["session_question_id"],
        user_answer
    ))
    mysql.connection.commit()

    answer_id = cur.lastrowid"""
r4 = """    cur.execute(\"\"\"
        INSERT INTO user_answers
        (session_question_id, answer_text)
        VALUES (%s,%s) RETURNING answer_id
    \"\"\", (
        session["session_question_id"],
        user_answer
    ))
    answer_id = cur.fetchone()[0]
    mysql.connection.commit()"""
content = content.replace(s4, r4)

# 5. user_answers 2
s5 = """        cur.execute(\"\"\"
            INSERT INTO user_answers(session_question_id, answer_text)
            VALUES(%s,%s)
        \"\"\", (
            session.get("session_question_id", None),
            "HR Video Response"
        ))
        mysql.connection.commit()

        answer_id = cur.lastrowid"""
r5 = """        cur.execute(\"\"\"
            INSERT INTO user_answers(session_question_id, answer_text)
            VALUES(%s,%s) RETURNING answer_id
        \"\"\", (
            session.get("session_question_id", None),
            "HR Video Response"
        ))
        answer_id = cur.fetchone()[0]
        mysql.connection.commit()"""
content = content.replace(s5, r5)

# Fix query placeholders: Postgres psycopg2 uses %s too, so NO CHANGE needed for placeholders!
# But DictCursor returns dict-like rows. If code uses row[0], it still works because DictRow supports index access.
# Wait, fetchone() with DictCursor returns a DictRow, which supports index access row[0]. So it's safe.

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Migration done")
