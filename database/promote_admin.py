import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

db_url = os.environ.get("DATABASE_URL")
if not db_url:
    print("Error: DATABASE_URL not found.")
    exit(1)

try:
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    # Add is_suspended column
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='users' AND column_name='is_suspended';
    """)
    if not cursor.fetchone():
        print("Adding is_suspended column...")
        cursor.execute("ALTER TABLE users ADD COLUMN is_suspended BOOLEAN DEFAULT FALSE;")
    else:
        print("is_suspended column already exists.")

    conn.commit()
    print("Database updated successfully.")

except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals():
        conn.close()
