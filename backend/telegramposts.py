import sys
import xgboost as xgb
from pyrogram import Client
from datetime import datetime
from flask import Flask, jsonify, request
import pyodbc
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
from transformers import pipeline
import torch
from deeppavlov import build_model, configs

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

API_ID = 28167910
API_HASH = "7d7f7bb60be610415488ecd8bc8731e9"

CHANNELS = ["@vertolatte", "@dronnitsa", "@donbassrussiazvo"]
received_messages = []


#ЗМІНИТИ ПІДКЛЮЧЕННЯ
DB_CONNECTION_STRING = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\SQLEXPRESS;"
    "DATABASE=MilitaryExperienceDetection;"
    "UID=sa;"
    "PWD=Admin@1234"
)

app = Flask(__name__)
CORS(app)

def model1(choice):
    class MilitaryAnalyzer:
        def __init__(self):
            # Initialize DeepPavlov NER model
            self.ner_model = build_model(configs.ner.ner_rus_bert, download=True)

            # Initialize transformer model
            self.model_name = "bodomerka/Milytary_exp_class_classification_sber_ai_based"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.transformers_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # Set up device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.transformers_model.to(self.device)

            # Initialize XGBoost
            self.xgb_model = None
            self.initialize_xgboost()

            # Load weapons list
            self.weapons = self.load_weapons()

            # Get model choice
            self.model_choice = choice

        def initialize_xgboost(self):
            """Initialize XGBoost model with error handling"""
            try:
                if torch.cuda.is_available():
                    self.xgb_model = xgb.Booster()
                    self.xgb_model.load_model("xgb_model.ubj")
            except Exception as e:
                print(f"Warning: Could not load XGBoost model: {e}")
                print("Will use Transformer model only")

        def load_weapons(self):
            """Load weapons list with error handling"""
            try:
                with open("weapon.txt", "r", encoding="utf-8") as file:
                    return {line.strip() for line in file}
            except FileNotFoundError:
                print("Warning: weapons.txt not found. Proceeding with empty weapons list.")
                return set()

        def get_model_choice(self):
            """Get model choice with validation"""
            while True:
                choice = input("Введіть 0 для Transformers або 1 для XGBoost: ").strip()
                if choice in ['0', '1']:
                    if choice == '1' and self.xgb_model is None:
                        print("XGBoost model not available. Using Transformers instead.")
                        return '0'
                    return choice
                print("Невірний вибір. Будь ласка, введіть 0 або 1.")

        def classify_text(self, text):
            """Classify text with error handling"""
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

                if self.model_choice == '0' or self.xgb_model is None:
                    with torch.no_grad():
                        outputs = self.transformers_model(**inputs)
                        return torch.argmax(outputs.logits).item() == 1
                else:
                    with torch.no_grad():
                        outputs = self.transformers_model(**inputs, output_hidden_states=True)
                        embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
                        dmatrix = xgb.DMatrix(embeddings)
                        return self.xgb_model.predict(dmatrix)[0] > 0.5
            except Exception as e:
                print(f"Error in classification: {e}")
                return False

        def analyze_text(self, text):
            """Analyze text with DeepPavlov NER"""
            try:
                ner_results = self.ner_model([text])
                entities = list(zip(ner_results[0][0], ner_results[1][0]))
                
                names = set()
                locations = set()
                
                for word, label in entities:
                    if label == "B-PER" or label == "I-PER":
                        names.add(word)
                    elif label == "B-LOC" or label == "I-LOC":
                        locations.add(word)
                
                return {
                    "military_experience": self.classify_text(text),
                    "names": list(names),
                    "locations": list(locations),
                    "mentioned_weapons": [w for w in self.weapons if w.lower() in text.lower()]
                }
            except Exception as e:
                print(f"Error in text analysis: {e}")
                return {
                    "military_experience": False,
                    "names": [],
                    "locations": [],
                    "mentioned_weapons": []
                }

        def generate_odcr_report(self, input_message):
            """Generate ODCR report with error handling"""
            try:
                prompt = f"""
                Convert message to ODCR and categorize the interested type of forces:

                Message:
                {input_message}

                Format ODCR:
                1. Observation: Briefly describe the issue or problem and its resolution.
                2. Discussion: Expand on the observation with key details (who, what, where, when, why, how) and its impact on operations.
                3. Conclusion: Summarize key points and support the recommendation.
                4. Recommendation: Suggest actions to resolve the issue, including responsible parties.
                5. Type: Specify which branch of the military this information may be useful for: Ground Forces, Air Force, Navy, Airborne Assault Forces, Support Forces, or None. Only one type.

                Follow the same example and answer in Ukrainian.
                """

                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500
                )
                return completion['choices'][0]['message']['content'].strip()
            except Exception as e:
                print(f"Error generating ODCR report: {e}")
                return "Error generating report"

        def process_csv(self, input_csv, output_csv):
            """Process CSV with progress tracking and error handling"""
            try:
                df = pd.read_csv(input_csv ,delimiter=";")
                print(df)
                if 'Message' not in df.columns:
                    raise ValueError("Відсутня колонка 'Message' у CSV файлі")

                total_rows = len(df)
                print(f"Processing {total_rows} messages...")

                results = []
                for idx, row in df.iterrows():
                    if idx % 10 == 0:
                        print(f"Progress: {idx}/{total_rows}")

                    analysis = self.analyze_text(row['Message'])
                    odcr = self.generate_odcr_report(row['Message'])
                    results.append({**analysis, 'odcr': odcr})

                # Update DataFrame
                df['military_experience'] = [r['military_experience'] for r in results]
                df['names'] = [', '.join(r['names']) for r in results]
                df['locations'] = [', '.join(r['locations']) for r in results]
                df['mentioned_weapons'] = [', '.join(r['mentioned_weapons']) for r in results]
                df['odcr_report'] = [r['odcr'] for r in results]

                df.to_csv(output_csv, index=False )
                print(f"Results saved to {output_csv}")

            except Exception as e:
                print(f"Error processing CSV: {e}")
              
//ВСТАВИТИ ОПЕН АРІ КЛЮЧ
    openai.api_key = ''

    analyzer = MilitaryAnalyzer()
    if len(sys.argv) > 1:
        input_message = sys.argv[1]
        analysis = analyzer.analyze_text(input_message)
        odcr_report = analyzer.generate_odcr_report(input_message)
    else:
        input_csv = "telegram_posts.csv"
        output_csv = "output_odcr_reports.csv"
        analyzer.process_csv(input_csv, output_csv)

def split_csv_files(input_file, no_odcr_output, odcr_output):
    """
    Split input CSV file into two separate files with specified columns.

    Args:
        input_file (str): Path to input CSV file
        no_odcr_output (str): Path for output file without ODCR columns
        odcr_output (str): Path for output file with only ODCR columns
    """
    try:
        df = pd.read_csv(input_file ,delimiter=";")

        # Define columns for each output file
        no_odcr_columns = [
            'TelegramPostInfoID',
            'Message',
            'Channel',
            'MessageDate',
            'military_experience',
            'names',
            'locations',
            'mentioned_weapons',
            'Type'
        ]

        odcr_columns = [
            'Observation',
            'Discussion',
            'Conclusion',
            'Recommendation'
        ]

        # Create separate dataframes for each output file
        no_odcr_df = df[no_odcr_columns]
        odcr_df = df[odcr_columns]

        # Save to separate CSV files
        no_odcr_df.to_csv(no_odcr_output, index=False)
        odcr_df.to_csv(odcr_output, index=False)

        print(f"Successfully split input file into:")
        print(f"1. {no_odcr_output} - {len(no_odcr_df)} rows")
        print(f"2. {odcr_output} - {len(odcr_df)} rows")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except KeyError as e:
        print(f"Error: Column {e} not found in input file")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def vector_db(df_current):
    class MessageManager:
        def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', index_path='faiss.index',
                     data_path='data.pkl'):
            self.model = SentenceTransformer(model_name)
            self.messages = []
            self.embeddings = np.empty((0, self.model.get_sentence_embedding_dimension()), dtype='float32')
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.index_path = index_path
            self.data_path = data_path
            self.index = faiss.IndexFlatL2(self.dimension)
            self.load_data()

        def add_new_message(self, new_message):
            new_embedding = self.model.encode([new_message]).astype('float32')
            self.index.add(new_embedding)
            self.messages.append(new_message)
            self.embeddings = np.vstack([self.embeddings, new_embedding])

        def is_similar(self, new_message, threshold=0.8):
            if len(self.messages) == 0:
                return False, False

            new_embedding = self.model.encode([new_message]).astype('float32')

            # Пошук найближчого сусіда
            D, I = self.index.search(new_embedding, k=1)  # D: відстані, I: індекси
            nearest_index = I[0][0]  # Отримання першого індексу найближчого сусіда
            nearest_message = self.messages[nearest_index]

            existing_embedding = self.embeddings[nearest_index]
            similarity = np.dot(new_embedding, existing_embedding) / (
                    np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding))

            similar = similarity > threshold

            return nearest_message, similar

        def save_data(self):
            faiss.write_index(self.index, self.index_path)
            with open(self.data_path, 'wb') as f:
                pickle.dump({'messages': self.messages, 'embeddings': self.embeddings}, f)

        def load_data(self):
            if os.path.exists(self.data_path) and os.path.exists(self.index_path):
                with open(self.data_path, 'rb') as f:
                    data = pickle.load(f)
                    self.messages = data['messages']
                    self.embeddings = data['embeddings']
                self.index = faiss.read_index(self.index_path)

        def shutdown(self):
            self.save_data()

    manager = MessageManager()
    manager.shutdown()

    print("HERE")

    def check_message_in_date_range(df, start_date, end_date, similar_message):
        df["date"] = pd.to_datetime(df["date"])
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        filtered_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        print("here check_message_in_date_range")
        return any(filtered_df["message"].str.contains(similar_message, case=False, na=False))

    file_path_stable = "C:\\Users\\User\\Desktop\\messages_data.csv"

    def rm_dublicates(df_current, file_path_stable):
        print("here rm_dublicates")
        manager = MessageManager()
        manager.shutdown()

        messages = df_current["Message"].to_list()
        ids = df_current["TelegramPostInfoID"].to_list()
        dates = df_current["MessageDate"].to_list()
        df_stable = pd.read_csv(file_path_stable)

        start_date = "2024-02-20"
        end_date = "2026-03-08"
        res_id = []
        threshold = 0.8

        for i in range(len(messages)):
            message = messages[i]
            id = ids[i]
            date = dates[i]

            nearest_message, similar = manager.is_similar(message, threshold)

            if not similar:
                manager.add_new_message(message)
                new_row = [(date, message)]
                df_new = pd.DataFrame(new_row, columns=["date", "message"])
                df_new.to_csv(file_path_stable, mode='a', index=False, header=False, encoding="utf-8")
                df_stable = pd.read_csv(file_path_stable)
                res_id.append(id)
            else:
                message_in_date_range = check_message_in_date_range(df_stable, start_date, end_date, nearest_message)
                if not message_in_date_range:
                    manager.add_new_message(message)
                    new_row = [(date, message)]
                    df_new = pd.DataFrame(new_row, columns=["date", "message"])
                    df_new.to_csv(file_path_stable, mode='a', index=False, header=False, encoding="utf-8")
                    df_stable = pd.read_csv(file_path_stable)
                    res_id.append(id)

        manager.shutdown()
        return res_id

    res_id = rm_dublicates(df_current, file_path_stable)
    return res_id


def get_db_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost\\SQLEXPRESS;"
        "DATABASE=MilitaryExperienceDetection;"
        "UID=sa;"
        "PWD=Admin@1234;"
    )
    return conn

def emoji_free(text):
    allchars = [c for c in text]
    emoji_list = [c for c in allchars if c in emoji.EMOJI_DATA]
    clean_text = ' '.join([word for word in text.split() if not any(e in word for e in emoji_list)])
    return clean_text

def remove_flags(text):
    flag_pattern = re.compile(r'[\U0001F1E6-\U0001F1FF]{2}')
    return flag_pattern.sub('', text)

def remove_flags_and_keycaps(text):
    flag_pattern = re.compile(r'[\U0001F1E6-\U0001F1FF]{2}')
    # *️⃣, 1️⃣, 2️⃣
    keycap_pattern = re.compile(r'[\*\d#]\uFE0F\u20E3')
    text_no_flags = flag_pattern.sub('', text)
    text_cleaned = keycap_pattern.sub('', text_no_flags)

    return text_cleaned

def process_clean_message(text):
    text = text.lower()
    text = re.sub(r'[^\w]', '', text, flags=re.UNICODE)
    text = text.replace(' ', '')
    return text

def preprocessing(mess):
    new_mes = emoji_free(mess)
    mes2 = remove_flags(new_mes)
    mes3 = remove_flags_and_keycaps(mes2)
    return mes3

def ensemble_classification(sentences, model_name_1, model_name_2):
    print("here1", sentences)

    tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
    model_1 = AutoModelForSequenceClassification.from_pretrained(model_name_1)

    tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
    model_2 = AutoModelForSequenceClassification.from_pretrained(model_name_2)

    final_labels = []
    inputs_1 = tokenizer_1(sentences, return_tensors="pt")
    inputs_2 = tokenizer_2(sentences, return_tensors="pt")

    with torch.no_grad():
        outputs_1 = model_1(**inputs_1)
        probs_1 = torch.nn.functional.softmax(outputs_1.logits, dim=-1)

    with torch.no_grad():
        outputs_2 = model_2(**inputs_2)
        probs_2 = torch.nn.functional.softmax(outputs_2.logits, dim=-1)

    avg_probs = (probs_1 + probs_2) / 2

    final_class = avg_probs.argmax().item()

    final_labels.append(final_class)

    return final_labels

def delete_old_posts():
    """ Видаляємо старі повідомлення для вказаного каналу перед отриманням нових """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM TelegramPostInfo2")
        conn.commit()
        conn.close()

        print(f"✅ Видалено старі повідомлення для каналу.")
    except Exception as e:
        print(f"❌ Помилка при видаленні старих повідомлень для каналу: {e}")


import pandas as pd
import csv

def export_posts_to_csv():
    try:
        conn = get_db_connection()

        query = "SELECT TelegramPostInfoID, Message, Channel, MessageDate FROM TelegramPostInfo2"
        df = pd.read_sql(query, conn)

        conn.close()

        csv_path = "telegram_posts.csv"

        df.to_csv(csv_path, index=False, encoding='utf-8', sep=";", quoting=csv.QUOTE_ALL)

        print(f"✅ Дані успішно експортовані в {csv_path}.")
        return csv_path

    except Exception as e:
        print(f"❌ Помилка під час експорту: {e}")
        return None


def save_to_db(message_text, channel, date_time):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO TelegramPostInfo2 (Message, Channel, MessageDate)
            VALUES (?, ?, ?)
            """,
            message_text, channel, date_time
        )
        conn.commit()
        conn.close()

        print(f"✅ [{channel}] Збережено нове повідомлення.")
    except Exception as e:
        print(f"❌ [{channel}] Помилка збереження в БД: {e}")

def save_to_db2(message_text, date_time):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO TelegramPostInfoPermanent (Message, MessageDate)
            VALUES (?, ?)
            """,
            message_text, date_time
        )
        conn.commit()
        conn.close()

        print(f"✅ Збережено нове повідомлення.")
    except Exception as e:
        print(f"❌ Помилка збереження в БД: {e}")

def predict_text(text):
    print(text)

    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruBert-base")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    model = AutoModelForSequenceClassification.from_pretrained(
        "bodomerka/Milytary_exp_class_classification_sber_ai_based", num_labels=2)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    print("here6")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred = torch.argmax(logits, dim=-1).item()  # Перетворюємо в скалярне значення

    return pred

def andre_model1():
    nlp = spacy.load("ru_core_news_md")
    model_name = "bodomerka/Milytary_exp_class_classification_sber_ai_based"
    classifier_pipeline = pipeline("text-classification", model=model_name)
    with open("weapon.txt", "r", encoding="utf-8") as file:
        weapons = {line.strip() for line in file}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
async def fetch_messages(app, start_date, end_date, channel_name):
    """ Отримуємо 10 повідомлень у заданому діапазоні дат """
    global received_messages
    received_messages.clear()
    print("heer")

    a = " "
    if channel_name == "Вертолатте":
        a = "@vertolatte"
    elif channel_name == "ДРОННИЦА":
        a = "@dronnitsa"
    else:
        a = "@donbassrussiazvo"

    try:
        chat = await app.get_chat(a)
        count = 0

        async for message in app.get_chat_history(chat.id):
            if not message.date:
                continue

            if message.date < start_date:
                break

            print("here2")
            message_text = message.text if message.text else message.caption

            if start_date <= message.date <= end_date and message_text:
                print(f"✅ Знайдено повідомлення в діапазоні: {message_text}")

                model_name_1 = "bodomerka/Milytary_exp_class_classification"
                model_name_2 = "bodomerka/Mil_class_exp_sber_balanssedclass"
                print(predict_text(message_text))
                labels = ensemble_classification(message_text, model_name_1, model_name_2)

                if labels[0] == 1:
                    save_to_db(message_text, channel_name, message.date)
                    save_to_db2(message_text, message.date)

                count += 1
                if count == 20:
                    break

        print(f"🔍 Всього знайдено {count} повідомлень.")
        #csv_path = export_posts_to_csv()
        #model1(0)
        #split_csv_files("output_odcr_reports.csv", "no_odcr.csv", "odcr.csv")
        #res = vector_db(csv_path)
        #print(res)

    except Exception as e:
        print(f"❌ Помилка при отриманні повідомлень: {e}")


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

        async def run_fetch_messages():
            async with Client("my_userbot", API_ID, API_HASH) as app:
                await fetch_messages(app, start_date, end_date, channel_name)

        asyncio.run(run_fetch_messages())

        new_array = []
        print("here2")
        for i in received_messages:
            mess5 = preprocessing(i)
            new_array.append(mess5)

        print(new_array)

        return jsonify({"message": "Повідомлення успішно отримані та збережені"}), 200
    except Exception as e:
        return jsonify({"error": f"Помилка: {e}"}), 500

@app.route('/api/posts', methods=['GET'])
def get_posts():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM TelegramPostInfo2"
        cursor.execute(query)
        rows = cursor.fetchall()

        posts = [{"TelegramPostInfoID": row[0], "Message": row[1], "Channel": row[2], "MessageDate": row[3]} for row in rows]

        return jsonify(posts), 200
    except Exception as e:
        return jsonify({"error": f"Помилка при отриманні повідомлень: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
