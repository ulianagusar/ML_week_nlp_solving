import logging
import os
import csv
import io
import asyncio
import sqlite3
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from pyrogram import Client
from services.remove_dublicates import MessageManager, rm_duplicates_time_range
from services.preproc import preprocessing
from services.odsr import generate_odcr_report
from services.ner import get_name, get_location, get_weapons
from sentence_transformers import SentenceTransformer

# ==========================
# Налаштування логування
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

API_ID = 28167910
API_HASH = "7d7f7bb60be610415488ecd8bc8731e9"

CHANNELS_FRONTEND = ["Вертолатте", "ДРОННИЦА", "Донбасс Россия", "VictoryDrones"]
CHANNELS = ["@vertolatte", "@dronnitsa", "@donbassrussiazvo", "@victorydrones"]
CHANNELS_MAP = dict(zip(CHANNELS_FRONTEND, CHANNELS))

app = Flask(__name__)
CORS(app)

scheduler = BackgroundScheduler()

DB_PATH = Path(__file__).resolve().parent / "database" / "database.db"
logging.info(f"DB Path: {DB_PATH}")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def save_data(message_id, message_text, channel, date_time, name, location, weapon, observation, discussion, conclusion, recommendation):
    try:
        messagelink = "https://t.me/" + channel[1:] + "/" + str(message_id)
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO TelegramPostInfo 
            (MessageID, Message, MessageLink, Channel, MessageDate, Name, Location, Weapons, Observation, Discussion, Conclusion, Recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, message_text, messagelink, channel, date_time, name, location, weapon, observation, discussion, conclusion, recommendation)
        )

        inserted_rows = cursor.rowcount
        conn.commit()
        conn.close()

        if inserted_rows > 0:
            logging.info(f"✅ Успішно вставлено {inserted_rows} записів")
        else:
            logging.warning("🚨 Дані не були додані!")

    except Exception as e:
        logging.error(f"🚨 Помилка збереження в БД: {e}")

@app.route('/api/clear_db', methods=['POST'])
def clear_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM TelegramPostInfo")
        conn.commit()
        conn.close()
        logging.info("Очистка таблиці TelegramPostInfo")
    except Exception as e:
        return jsonify({"error": f"DB Clear Error: {e}"}), 500

    try:
        for file_path in ["faiss.index", "data.pkl"]:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Файл {file_path} видалено.")
            else:
                logging.info(f"Файл {file_path} не знайдено.")
    except Exception as e:
        return jsonify({"error": f"FAISS File Delete Error: {e}"}), 500

    return jsonify({"message": "Database cleared successfully"}), 200

async def fetch_messages(start_date, end_date, channel_name):
    channels_fetch = [CHANNELS_MAP[channel_name]] if channel_name in CHANNELS_MAP else CHANNELS
    messages, dates, channels, ids = [], [], [], []

    async with Client("military_bot", API_ID, API_HASH) as app:
        try:
            for channel in channels_fetch:
                chat = await app.get_chat(channel)

                async for message in app.get_chat_history(chat.id):
                    if not message.date or message.date < start_date:
                        break
                    if start_date <= message.date <= end_date:
                        message_text = message.text or message.caption
                        if message_text:
                            messages.append(message_text)
                            dates.append(message.date.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
                            channels.append(channel)
                            ids.append(message.id)

        except Exception as e:
            logging.error(f"Помилка отримання повідомлень: {e}")
        finally:
            await app.stop()

    return messages, dates, channels, ids

def get_messages_sync(start_date, end_date, channel_name):
    return asyncio.run(fetch_messages(start_date, end_date, channel_name))

def process_and_save_posts(data):
    channel_name = data.get('channel')
    start_date = datetime.strptime(data.get('start_date'), "%Y-%m-%d")
    end_date = datetime.strptime(data.get('end_date'), "%Y-%m-%d")
    model = data.get('model')

    start_time = datetime.now()
    logging.info(f"🔄 Початок обробки повідомлень: {start_time}")

    messages, dates, channels, ids = get_messages_sync(start_date, end_date, channel_name)
    total = len(messages)
    logging.info(f"Загальна кількість отриманих повідомлень: {total}")

    cleaned_messages, exp_only_mes, exp_only_date, exp_only_id, exp_only_channels = [], [], [], [], []

    for i in range(total):
        cleaned_message = preprocessing(messages[i])
        try:
            endpoint = "http://backend2:5003/predict/bert" if model == "ruBert" else "http://backend2:5003/predict/xgboost"
            response = requests.post(endpoint, json={"text": cleaned_message})
            exp_class = response.json().get("prediction")

            logging.info(f"[{i+1}/{total}] EXP_CLASS={exp_class}")

            if exp_class == 1:
                exp_only_mes.append(messages[i])
                exp_only_date.append(dates[i])
                exp_only_id.append(ids[i])
                exp_only_channels.append(channels[i])
                cleaned_messages.append(cleaned_message)

        except Exception as e:
            logging.error(f"[{i+1}/{total}] Помилка моделі: {e}")

    manager = MessageManager()
    ids_unique = rm_duplicates_time_range(manager, cleaned_messages, exp_only_date, exp_only_id)
    logging.info(f"Унікальні повідомлення для збереження: {len(ids_unique)}")

    for i in range(len(exp_only_mes)):
        if exp_only_id[i] in ids_unique:
            o, d, c, r, t = generate_odcr_report(cleaned_messages[i])
            save_data(exp_only_id[i], cleaned_messages[i], exp_only_channels[i], exp_only_date[i],
                      get_name(cleaned_messages[i]), get_location(cleaned_messages[i]),
                      get_weapons(cleaned_messages[i]), o, d, c, r)

@app.route('/api/fetch_posts', methods=['POST'])
def fetch_posts():
    data = request.json
    logging.info(f"Запит fetch_posts: {data}")
    try:
        process_and_save_posts(data)
        return jsonify({"message": "Повідомлення оброблено"}), 200
    except Exception as e:
        logging.error(f"Помилка в endpoint fetch_posts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/posts', methods=['GET'])
def get_posts():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT MessageID, Message, Channel, MessageDate FROM TelegramPostInfo")
        rows = cursor.fetchall()
        conn.close()

        posts = [{"TelegramPostInfoID": row["MessageID"], "Message": row["Message"],
                  "Channel": row["Channel"], "MessageDate": row["MessageDate"]} for row in rows]

        return jsonify(posts), 200
    except Exception as e:
        logging.error(f"Помилка get_posts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/odsr', methods=['GET'])
def get_odsr_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = ("SELECT MessageID, MessageDate, Observation, Discussion, Conclusion, Recommendation "
                 "FROM TelegramPostInfo WHERE LENGTH(COALESCE(Observation, '')) > 0 "
                 "ORDER BY MessageDate LIMIT 100")
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        odsrs = [{"TelegramPostInfoID": row["MessageID"], "MessageDate": row["MessageDate"],
                  "Observation": row["Observation"], "Discussion": row["Discussion"],
                  "Conclusion": row["Conclusion"], "Recommendation": row["Recommendation"]} for row in rows]

        return jsonify(odsrs), 200
    except Exception as e:
        logging.error(f"Помилка ODSR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_report', methods=['GET'])
def download_csv():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM TelegramPostInfo")
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    conn.close()

    if not rows:
        return Response("Немає даних для експорту", status=204)

    output = io.StringIO()
    csv_writer = csv.writer(output)
    csv_writer.writerow(column_names)
    csv_writer.writerows(rows)

    response = Response(output.getvalue(), content_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=report.csv"
    return response

def schedule_post_fetch():
    try:
        process_and_save_posts({
            "channel": "",
            "start_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "model": "ruBert"
        })
    except Exception as e:
        logging.error(f"Помилка під час запланованого запуску: {e}")

scheduler.add_job(schedule_post_fetch, 'cron', hour=0, minute=0)
scheduler.start()

if __name__ == '__main__':
    logging.info(f"🔧 Старт сервера {datetime.now()}")
    app.run(debug=False, host='0.0.0.0', port=5001)
