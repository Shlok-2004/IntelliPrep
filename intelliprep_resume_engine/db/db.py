import pymysql

def get_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="intelliprep",
        cursorclass=pymysql.cursors.DictCursor
    )
