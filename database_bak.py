import sqlite3
from bcrypt import hashpw, gensalt, checkpw

DB_PATH = "us3rs_23.db"

def initialize_database():
    """
    Initialize the SQLite database and create the users table if it doesn't exist.
    Add default admin credentials if not already present.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Create the users table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()

    # Check if the admin user exists, and insert it if not
    cursor.execute("SELECT * FROM users WHERE username = ?", ("admin",))
    if not cursor.fetchone():
        default_password = "admin123"
        hashed_password = hashpw(default_password.encode('utf-8'), gensalt())
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", hashed_password))
        conn.commit()
        #print("Default admin credentials added: username='admin', password='admin123'")

    conn.close()

def add_user(username, password):
    """
    Add a new user to the database with a hashed password.
    """
    hashed_password = hashpw(password.encode('utf-8'), gensalt())
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
    except sqlite3.IntegrityError:
        print("Error: Username already exists.")
    conn.close()

def authenticate_user(username, password):
    """
    Authenticate a user by verifying the password against the database.
    Returns True if authenticated, otherwise False.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()

    if result:
        stored_password = result[0]  # This is likely a bytes object from SQLite
        return checkpw(password.encode('utf-8'), stored_password)  # No need to encode stored_password
    return False
