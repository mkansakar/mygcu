#database.py
import sqlite3
from bcrypt import hashpw, gensalt, checkpw

DB_PATH = "us3rs_23.db"

def initialize_database():
    """
    Initialize the SQLite database and create the ustbl3s table if it doesn't exist.
    Add default admin credentials if not already present.
    """
    try:
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
            try:
                cursor.execute("INSERT INTO ustbl3s (username, password, role) VALUES (?, ?, ?)", ("admin", hashed_password, "admin"))
                cursor.execute("INSERT INTO ustbl3s (username, password, role) VALUES (?, ?, ?)", ("user1", hashed_password, "normal"))
                conn.commit()
            except sqlite3.IntegrityError:
                print("Admin or default user already exists. Skipping insertion.")


    except sqlite3.Error as e:
        print(f"Database error during initialization: {e}")
    except Exception as e:
        print(f"Unexpected error during database initialization: {e}")
    finally:
        conn.close()

def add_user(username, password, role="admin"):
    """
    Add a new user to the database with a hashed password and specified role.
    """
    hashed_password = hashpw(password.encode('utf-8'), gensalt())
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO ustbl3s (username, password, role) VALUES (?, ?, ?)", 
                       (username, hashed_password, role))
        conn.commit()
        print(f"User '{username}' added successfully.")
    except sqlite3.IntegrityError:
        print(f"Error: Username '{username}' already exists.")
    except sqlite3.Error as e:
        print(f"Database error while adding user: {e}")
    except Exception as e:
        print(f"Unexpected error while adding user: {e}")
    finally:
        conn.close()

def authenticate_user(username, password):
    """
    Authenticate a user by verifying the password against the database.
    Returns True if authenticated, otherwise False.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT password, role FROM ustbl3s WHERE username = ?", (username,))
        result = cursor.fetchone()

        if result:
            stored_password, role = result  
            if checkpw(password.encode('utf-8'), stored_password):
                return True, role  # Return True and user role if authentication succeeds
        return False, None  # Authentication failed

    except sqlite3.Error as e:
        print(f"Database error during authentication: {e}")
        return False, None
    except Exception as e:
        print(f"Unexpected error during authentication: {e}")
        return False, None
    finally:
        conn.close()
    
