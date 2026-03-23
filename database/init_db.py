import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ.get("DATABASE_URL")

if not DB_URL:
    print("DATABASE_URL environment variable is missing!")
    exit(1)

def init_db():
    print(f"Connecting to database...")
    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
        
        sql_file_path = os.path.join(os.path.dirname(__file__), 'interview_prep_db.sql')
        
        with open(sql_file_path, 'r', encoding='utf-8') as file:
            sql_script = file.read()
            
        print("Executing SQL script...")
        cursor.execute(sql_script)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    init_db()
