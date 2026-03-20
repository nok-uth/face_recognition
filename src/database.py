import sqlite3
import datetime
import os

# Define where the database file will live
DB_PATH = '../data/attendance.db'

def setup_db():
    """Creates the database and the attendance table if they don't exist."""
    
    # Ensure the data folder actually exists before we try to put a file in it
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Connect to the database (this creates the file if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create our table with columns for ID, Name, Date, and Time
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database setup complete. Ready to track attendance!")

def log_attendance(name):
    """Logs a person's name with the current date and time. 
       Prevents double-logging the same person on the same day."""
       
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Grab the exact current date and time
    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    
    # Check if this person was already logged today
    cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
    record = cursor.fetchone()
    
    # If they aren't in the database for today, insert a new row!
    if record is None:
        cursor.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, current_date, current_time))
        conn.commit()
        print(f"Success: Logged {name} at {current_time}")
    
    conn.close()

# If you run this specific file, it will trigger the setup function
if __name__ == "__main__":
    setup_db()