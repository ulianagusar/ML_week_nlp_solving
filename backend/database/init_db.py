
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "database" / "database.db"

def init_db():
    """Ініціалізація бази даних та створення таблиць"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Таблиця для повідомлень
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS TelegramPostInfo (
            MessageID INTEGER PRIMARY KEY AUTOINCREMENT,
            Message TEXT NOT NULL,
            Channel TEXT NOT NULL,
            MessageDate TEXT NOT NULL,
            Name TEXT,
            Location TEXT,
            Weapons TEXT
        )
        """
    )

    # Таблиця для аналізу O-D-C-R
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ODCRAnalysis (
            AnalysisID INTEGER PRIMARY KEY AUTOINCREMENT,
            MessageID INTEGER,
            Observation TEXT,
            Discussion TEXT,
            Conclusion TEXT,
            Recommendation TEXT,
            FOREIGN KEY (MessageID) REFERENCES TelegramPostInfo(MessageID) ON DELETE CASCADE
        )
        """
    )

    conn.commit()
    conn.close()
    print("✅ База даних успішно ініціалізована.")

if __name__ == "__main__":
    init_db()




