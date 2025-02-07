import sqlite3
from pathlib import Path


def init_db():

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS TelegramPostInfo (
            MessageID INTEGER PRIMARY KEY AUTOINCREMENT,
            Message TEXT NOT NULL,
            Channel TEXT NOT NULL,
            MessageDate TEXT NOT NULL,
            Name TEXT,
            Location TEXT,
            Weapons TEXT,
            Observation TEXT,
            Discussion TEXT,
            Conclusion TEXT,
            Recommendation TEXT
        )
        """
    )

    conn.commit()
    conn.close()
    print("✅ База даних створена або вже існує.")



if __name__ == "__main__":
    init_db()




