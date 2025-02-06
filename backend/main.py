#pip install TgCrypto


import sys
import xgboost as xgb
from pyrogram import Client
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import asyncio
import pandas as pd
import emoji
import openai
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import spacy
from transformers import pipeline
import torch
import pandas as pd
import csv
from ML_week_nlp_solving.backend.services.remove_dublicates import MessageManager , rm_dublicates
import sqlite3
import pandas as pd
import csv
from ML_week_nlp_solving.backend.services.preproc import preprocessing
from ML_week_nlp_solving.backend.services.odsr import get_o , get_d , get_c , get_r
from ML_week_nlp_solving.backend.services.ner import get_name , get_location , get_weapons 
from ML_week_nlp_solving.backend.services.experience import military_classification
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

API_ID = 28167910
API_HASH = "7d7f7bb60be610415488ecd8bc8731e9"

CHANNELS = ["@vertolatte", "@dronnitsa", "@donbassrussiazvo"]
received_messages = []



app = Flask(__name__)
CORS(app)


DB_PATH = Path(__file__).resolve().parent / "database" / "database.db"

def get_db_connection():
    """Підключення до бази даних"""
    conn = sqlite3.connect(DB_PATH)  
    conn.row_factory = sqlite3.Row  # Дозволяє доступ до стовпців за назвами
    return conn

def save_final_mess(message_text, channel, date_time, name, location, weapon):
    """Зберігає повідомлення в таблиці TelegramPostInfo та повертає його MessageID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO TelegramPostInfo (Message, Channel, MessageDate, Name, Location, Weapons)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_text, channel, date_time, name, location, weapon)
        )

        message_id = cursor.lastrowid  # Отримуємо ID нового запису

        conn.commit()
        conn.close()

        print(f"✅ [{channel}] Збережено нове повідомлення з ID {message_id}.")
        return message_id  # Повертаємо MessageID для подальшого використання
    except Exception as e:
        print(f"❌ [{channel}] Помилка збереження в БД: {e}")
        return None

def save_odcr(message_id, observation, discussion, conclusion, recommendation):
    """Зберігає результати O-D-C-R аналізу у таблиці ODCRAnalysis"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO ODCRAnalysis (MessageID, Observation, Discussion, Conclusion, Recommendation)
            VALUES (?, ?, ?, ?, ?)
            """,
            (message_id, observation, discussion, conclusion, recommendation)
        )

        conn.commit()
        conn.close()

        print(f"✅ Збережено аналіз O-D-C-R для повідомлення ID {message_id}.")
    except Exception as e:
        print(f"❌ Помилка збереження O-D-C-R в БД: {e}")

def delete_old_posts():
    """Видаляє всі повідомлення з TelegramPostInfo та пов’язані O-D-C-R аналізи"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Видаляємо всі записи у головній таблиці (завдяки ON DELETE CASCADE видаляться пов’язані записи у ODCRAnalysis)
        cursor.execute("DELETE FROM TelegramPostInfo")
        conn.commit()
        conn.close()

        print(f"✅ Видалено всі повідомлення.")
    except Exception as e:
        print(f"❌ Помилка при видаленні старих повідомлень: {e}")

def get_tg_messages(app, start_date, end_date, channel_name):
    """ Отримуємо 10 повідомлень у заданому діапазоні дат """

    try:
        messages = ["Росія планує наступ завтра" , "Бои под Авдєєвкой продолжаются" ,"Получили новое вооружение - гранати" ]
        dates = ["2029-01-28 19:28:10.123", "2029-01-29 10:00:00.000", "2029-01-28 19:28:10.123"]
        channels = ["c1", "c2", "c3"]
        ids = [1, 2, 3]

        # messages = {
        #     "Message": ["Росія планує наступ завтра" , "Бои под Авдєєвкой продолжаются" ,"Получили новое вооружение - гранати" ],
        #     "MessageDate": ["2029-01-28 19:28:10.123", "2029-01-29 10:00:00.000", "2029-01-28 19:28:10.123"],
        #     "TelegramPostInfoID": [1, 2, 3]
        # }
        return messages , dates , channels , ids
    except Exception as e:
        print(f"❌ Помилка при отриманні повідомлень: {e}")
        return None , None , None



@app.route('/api/fetch_posts', methods=['POST'])
def fetch_posts():
    data = request.json
    print(data)
    channel_name = data.get('channel')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')

    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        delete_old_posts()
        messages , dates , channels , ids  = get_tg_messages(app, start_date, end_date, channel_name)

        exp_only_mes = []
        exp_only_date = []
        exp_only_id = []
        exp_only_channels = []


        cleaned_messages = []

        for i in range(len(messages)) :
                cleaned_message = preprocessing(messages[i])
                exp_class = military_classification(cleaned_message)
                if exp_class == 1:
                    exp_only_mes.append(messages[i])
                    exp_only_date.append(dates[i])
                    exp_only_id.append(ids[i])
                    exp_only_channels.append(channels[i])

                    cleaned_messages.append(cleaned_message)

        manager = MessageManager()
        ids_unique = rm_dublicates(manager , cleaned_messages)

        for i in range(len(exp_only_mes)) :
            if i in ids_unique :
                 mes_id = save_final_mess(exp_only_mes[i], exp_only_channels[i], exp_only_date[i] ,
                        get_name(cleaned_messages[i]) ,get_location(cleaned_messages[i]) , get_weapons(cleaned_messages[i]) )  
                 save_odcr(mes_id ,  get_o(cleaned_messages[i]) ,
                        get_d(cleaned_messages[i]) , get_c(cleaned_messages[i]) , get_r(cleaned_messages[i]))


        return jsonify({"message": "Повідомлення успішно отримані та збережені"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": f"Помилка: {e}"}), 500




@app.route('/api/posts', methods=['GET'])
def get_posts():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = "SELECT * FROM TelegramPostInfo2"
        cursor.execute(query)
        rows = cursor.fetchall()

        # В SQLite потрібно передавати значення як tuple-елементи
        posts = [{"TelegramPostInfoID": row[0], "Message": row[1], "Channel": row[2], "MessageDate": row[3]} for row in rows]

        conn.close()

        return jsonify(posts), 200
    except Exception as e:
        print(e)
        return jsonify({"error": f"Помилка при отриманні повідомлень: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
