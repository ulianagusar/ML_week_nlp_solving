import sqlite3

def init_db():
    conn = sqlite3.connect("database.db")  # Створюємо SQLite БД
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS TelegramPostInfo2 (
            TelegramPostInfoID INTEGER PRIMARY KEY AUTOINCREMENT,
            Message TEXT NOT NULL,
            Channel TEXT NOT NULL,
            MessageDate TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    print("✅ База даних успішно ініціалізована.")
init_db()