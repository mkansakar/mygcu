import sqlite3
from bcrypt import hashpw, gensalt, checkpw

DB_PATH = "us3rs_23.db"

def initialize_database():
    """
    Initialize the SQLite database and create the ustbl3s table if it doesn't exist.
    Add default admin credentials if not already present.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Create the ustbl3s table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ustbl3s (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'normal'
        )
    """)
    conn.commit()

    # Check if the admin user exists, and insert it if not
    cursor.execute("SELECT * FROM ustbl3s WHERE username = ?", ("admin",))
    if not cursor.fetchone():
        
        default_password = "admin123"
        hashed_password = hashpw(default_password.encode('utf-8'), gensalt())
        cursor.execute("INSERT INTO ustbl3s (username, password, role) VALUES (?, ?, ?)", ("admin", hashed_password, "admin"))
        cursor.execute("INSERT INTO ustbl3s (username, password, role) VALUES (?, ?, ?)", ("user1", hashed_password, "normal"))
        conn.commit()
        

    conn.close()

def add_user(username, password, role="normal"):
    """
    Add a new user to the database with a hashed password and specified role.
    """
    hashed_password = hashpw(password.encode('utf-8'), gensalt())
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO ustbl3s (username, password, role) VALUES (?, ?, ?)", (username, hashed_password, role))
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
    cursor.execute("SELECT password, role FROM ustbl3s WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()

    if result:
        stored_password, role = result  
        if checkpw(password.encode('utf-8'), stored_password):
            return True, role  # No need to encode stored_password
    return False, None
    
