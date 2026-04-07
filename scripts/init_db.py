"""One-time script to create SQLite tables. Also runs automatically at app startup."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import create_tables

if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully.")
