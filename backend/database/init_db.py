import sqlite3
from pathlib import Path



def init_db():

    conn = sqlite3.connect("database.db")
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
    print(f"✅ База даних створена (або вже існує) за шляхом:")

# Запуск ініціалізації, якщо цей файл виконується напряму
if __name__ == "__main__":
    init_db()




