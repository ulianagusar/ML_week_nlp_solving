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
from services.remove_dublicates import MessageManager , rm_dublicates
import sqlite3
import pandas as pd
import csv
from services.preproc import preprocessing
from services.odsr import get_o , get_d , get_c , get_r
from services.ner import get_name , get_location , get_weapons 
from services.experience import experience_bert , experience_xg_boost
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

API_ID = 28167910
API_HASH = "7d7f7bb60be610415488ecd8bc8731e9"

CHANNELS = ["@vertolatte", "@dronnitsa", "@donbassrussiazvo"]
received_messages = []



app = Flask(__name__)
CORS(app)


DB_PATH = Path(__file__).resolve().parent / "database" / "database.db"
print(DB_PATH)

def get_db_connection():

    conn = sqlite3.connect(DB_PATH)  
    conn.row_factory = sqlite3.Row 
    return conn



def save_data(message_id, message_text, channel, date_time, name, location, weapon, observation, discussion, conclusion, recommendation):

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO TelegramPostInfo 
            (MessageID, Message, Channel, MessageDate, Name, Location, Weapons, Observation, Discussion, Conclusion, Recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, message_text, channel, date_time, name, location, weapon, observation, discussion, conclusion, recommendation)
        )

        conn.commit()
        conn.close()

    except Exception as e:
        print(f"Error saving to the database: {e}")



def delete_old_posts():

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM TelegramPostInfo")

        conn.commit()
        conn.close()

        print("Deleted all records from the TelegramPostInfo table.")
    except Exception as e:
        print(f"Error deleting records:  {e}")



def get_tg_messages(app, start_date, end_date, channel_name):


    try:
        messages = ["Росія планує наступ завтра" , "Бои под Авдєєвкой продолжаются , скоро откроют новую продвижную дорогу , по которой пройдет наш пихотний батальйон" ,"Получили новое вооружение - гранати","Получили новое вооружение - гранати" ]
        dates = ["2029-01-28 19:28:10.123", "2029-01-29 10:00:00.000", "2029-01-28 19:28:10.123" , "2029-01-28 19:28:10.123"]
        channels = ["c1", "c2", "c3", "c3"]
        ids = [5, 6, 7, 8]

        # messages = {
        #     "Message": ["Росія планує наступ завтра" , "Бои под Авдєєвкой продолжаются" ,"Получили новое вооружение - гранати" ],
        #     "MessageDate": ["2029-01-28 19:28:10.123", "2029-01-29 10:00:00.000", "2029-01-28 19:28:10.123"],
        #     "TelegramPostInfoID": [1, 2, 3]
        # }
        return messages , dates , channels , ids
    except Exception as e:
        print(f"Error when receiving messages {e}")
        return None , None , None



@app.route('/api/fetch_posts', methods=['POST'])
def fetch_posts():
    data = request.json
    print(data)
    channel_name = data.get('channel')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')
    model = data.get('model')
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        delete_old_posts()
        messages , dates , channels , ids  = get_tg_messages(app, start_date, end_date, channel_name)

        exp_only_mes = []
        exp_only_date = []
        exp_only_id = []
        exp_only_channels = []

        print(messages)
        cleaned_messages = []

        for i in range(len(messages)) :
                cleaned_message = preprocessing(messages[i])
                if model == "Bert":
                     exp_class = experience_bert(cleaned_message)
                else :
                     exp_class = experience_xg_boost(cleaned_message)

                if exp_class == 1:
                    exp_only_mes.append(messages[i])
                    exp_only_date.append(dates[i])
                    exp_only_id.append(ids[i])
                    exp_only_channels.append(channels[i])

                    cleaned_messages.append(cleaned_message)

        manager = MessageManager()

        ids_unique = rm_dublicates(manager , cleaned_messages ,exp_only_date ,exp_only_id ) #rm_dublicates(manager , cleaned_messages)
        print(ids_unique)
        for i in range(len(exp_only_mes)) :
            if exp_only_id[i] in ids_unique :
                 
                 save_data(exp_only_id[i], exp_only_mes[i], exp_only_channels[i], exp_only_date[i], get_name(cleaned_messages[i]), get_location(cleaned_messages[i]),
                            get_weapons(cleaned_messages[i]) , get_o(cleaned_messages[i]), get_d(cleaned_messages[i]), get_c(cleaned_messages[i]), get_r(cleaned_messages[i]))



        return jsonify({"message": "Messages successfully received and saved"}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": f"error fetch_posts: {e}"}), 500




@app.route('/api/posts', methods=['GET'])
def get_posts():

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT MessageID, Message, Channel, MessageDate FROM TelegramPostInfo"
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()


        posts = [
            {
                "TelegramPostInfoID": row["MessageID"],  
                "Message": row["Message"],
                "Channel": row["Channel"],
                "MessageDate": row["MessageDate"]
            }
            for row in rows
        ]

        return jsonify(posts), 200
    except Exception as e:
        print(f"Error receiving messages {e}")
        return jsonify({"error": f"Error when receiving messages: {e}"}), 500






if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
