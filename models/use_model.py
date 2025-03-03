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
import openai
import json


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

    def classify_text(self, text):
        """Classify text with error handling"""
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512
            ).to(self.device)


            if self.xgb_model is None:
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

    def clean_text(self, text):
        """Очистка та обмеження довжини тексту"""
        text = emoji.get_emoji_regexp().sub(r'', text)  
        text = re.sub(r'[^\w\s]', '', text)  
        text = ' '.join(text.split())  
        words = text.split()[:512]
        return ' '.join(words)


    def generate_odcr_report(self, input_message):
        """Generate ODCR report with error handling"""
        try:
            # Clean the input message before passing it to OpenAI
            cleaned_message = self.clean_text(input_message)

            # Debug: Print cleaned message
            print(f"Cleaned Message: {cleaned_message}")

            prompt = f"""
            Convert message to ODCR and categorize the interested type of forces:

            Message:
            {cleaned_message}

            Format ODCR:
            1. Observation: Briefly describe the issue or problem and its resolution.
            2. Discussion: Expand on the observation with key details (who, what, where, when, why, how) and its impact on operations.
            3. Conclusion: Summarize key points and support the recommendation.
            4. Recommendation: Suggest actions to resolve the issue, including responsible parties.
            5. Type: Specify which branch of the military this information may be useful for: Ground Forces, Air Force, Navy, Airborne Assault Forces, Support Forces, or None. Only one type.

            Follow the same example and answer in Ukrainian.
            """

            # Debugging the formatted prompt before sending it
            print(f"Prompt: {prompt}")

            # Send the cleaned prompt to OpenAI
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                max_tokens=500
            )
            return completion['choices'][0]['message']['content'].strip()

        except Exception as e:
            print(f"Error generating ODCR report: {e}")
            return "Error generating report"


    def process_csv(self, input_csv, output_csv):
        """Process CSV with progress tracking and error handling"""
        try:
            df = pd.read_csv(input_csv, delimiter=";")
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

            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")

        except Exception as e:
            print(f"Error processing CSV: {e}")

# import sys
# import xgboost as xgb
# from pyrogram import Client
# from datetime import datetime
# from flask import Flask, jsonify, request
# import pyodbc
# from flask_cors import CORS
# import asyncio
# import pandas as pd
# import emoji
# import openai
# import re
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# import pickle
# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from transformers import pipeline
# import torch
# from deeppavlov import build_model, configs

# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# API_ID = 28167910
# API_HASH = "7d7f7bb60be610415488ecd8bc8731e9"

# CHANNELS = ["@vertolatte", "@dronnitsa", "@donbassrussiazvo"]
# received_messages = []


# #ЗМІНИТИ ПІДКЛЮЧЕННЯ
# DB_CONNECTION_STRING = (
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     "SERVER=localhost\\SQLEXPRESS;"
#     "DATABASE=MilitaryExperienceDetection;"
#     "UID=sa;"
#     "PWD=Admin@1234"
# )

# app = Flask(__name__)
# CORS(app)

# def model1(choice):
#     class MilitaryAnalyzer:
#         def __init__(self):
#             # Initialize DeepPavlov NER model
#             self.ner_model = build_model(configs.ner.ner_rus_bert, download=True)

#             # Initialize transformer model
#             self.model_name = "bodomerka/Milytary_exp_class_classification_sber_ai_based"
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#             self.transformers_model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

#             # Set up device
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             self.transformers_model.to(self.device)

#             # Initialize XGBoost
#             self.xgb_model = None
#             self.initialize_xgboost()

#             # Load weapons list
#             self.weapons = self.load_weapons()

#             # Get model choice
#             self.model_choice = choice

#         def initialize_xgboost(self):
#             """Initialize XGBoost model with error handling"""
#             try:
#                 if torch.cuda.is_available():
#                     self.xgb_model = xgb.Booster()
#                     self.xgb_model.load_model("xgb_model.ubj")
#             except Exception as e:
#                 print(f"Warning: Could not load XGBoost model: {e}")
#                 print("Will use Transformer model only")

#         def load_weapons(self):
#             """Load weapons list with error handling"""
#             try:
#                 with open("weapon.txt", "r", encoding="utf-8") as file:
#                     return {line.strip() for line in file}
#             except FileNotFoundError:
#                 print("Warning: weapons.txt not found. Proceeding with empty weapons list.")
#                 return set()

#         def get_model_choice(self):
#             """Get model choice with validation"""
#             while True:
#                 choice = input("Введіть 0 для Transformers або 1 для XGBoost: ").strip()
#                 if choice in ['0', '1']:
#                     if choice == '1' and self.xgb_model is None:
#                         print("XGBoost model not available. Using Transformers instead.")
#                         return '0'
#                     return choice
#                 print("Невірний вибір. Будь ласка, введіть 0 або 1.")

#         def classify_text(self, text):
#             """Classify text with error handling"""
#             try:
#                 inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

#                 if self.model_choice == '0' or self.xgb_model is None:
#                     with torch.no_grad():
#                         outputs = self.transformers_model(**inputs)
#                         return torch.argmax(outputs.logits).item() == 1
#                 else:
#                     with torch.no_grad():
#                         outputs = self.transformers_model(**inputs, output_hidden_states=True)
#                         embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
#                         dmatrix = xgb.DMatrix(embeddings)
#                         return self.xgb_model.predict(dmatrix)[0] > 0.5
#             except Exception as e:
#                 print(f"Error in classification: {e}")
#                 return False

#         def analyze_text(self, text):
#             """Analyze text with DeepPavlov NER"""
#             try:
#                 ner_results = self.ner_model([text])
#                 entities = list(zip(ner_results[0][0], ner_results[1][0]))
                
#                 names = set()
#                 locations = set()
                
#                 for word, label in entities:
#                     if label == "B-PER" or label == "I-PER":
#                         names.add(word)
#                     elif label == "B-LOC" or label == "I-LOC":
#                         locations.add(word)
                
#                 return {
#                     "military_experience": self.classify_text(text),
#                     "names": list(names),
#                     "locations": list(locations),
#                     "mentioned_weapons": [w for w in self.weapons if w.lower() in text.lower()]
#                 }
#             except Exception as e:
#                 print(f"Error in text analysis: {e}")
#                 return {
#                     "military_experience": False,
#                     "names": [],
#                     "locations": [],
#                     "mentioned_weapons": []
#                 }

#         def generate_odcr_report(self, input_message):
#             """Generate ODCR report with error handling"""
#             try:
#                 prompt = f"""
#                 Convert message to ODCR and categorize the interested type of forces:

#                 Message:
#                 {input_message}

#                 Format ODCR:
#                 1. Observation: Briefly describe the issue or problem and its resolution.
#                 2. Discussion: Expand on the observation with key details (who, what, where, when, why, how) and its impact on operations.
#                 3. Conclusion: Summarize key points and support the recommendation.
#                 4. Recommendation: Suggest actions to resolve the issue, including responsible parties.
#                 5. Type: Specify which branch of the military this information may be useful for: Ground Forces, Air Force, Navy, Airborne Assault Forces, Support Forces, or None. Only one type.

#                 Follow the same example and answer in Ukrainian.
#                 """

#                 completion = openai.ChatCompletion.create(
#                     model="gpt-4",
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     max_tokens=500
#                 )
#                 return completion['choices'][0]['message']['content'].strip()
#             except Exception as e:
#                 print(f"Error generating ODCR report: {e}")
#                 return "Error generating report"

#         def process_csv(self, input_csv, output_csv):
#             """Process CSV with progress tracking and error handling"""
#             try:
#                 df = pd.read_csv(input_csv ,delimiter=";")
#                 print(df)
#                 if 'Message' not in df.columns:
#                     raise ValueError("Відсутня колонка 'Message' у CSV файлі")

#                 total_rows = len(df)
#                 print(f"Processing {total_rows} messages...")

#                 results = []
#                 for idx, row in df.iterrows():
#                     if idx % 10 == 0:
#                         print(f"Progress: {idx}/{total_rows}")

#                     analysis = self.analyze_text(row['Message'])
#                     odcr = self.generate_odcr_report(row['Message'])
#                     results.append({**analysis, 'odcr': odcr})

#                 # Update DataFrame
#                 df['military_experience'] = [r['military_experience'] for r in results]
#                 df['names'] = [', '.join(r['names']) for r in results]
#                 df['locations'] = [', '.join(r['locations']) for r in results]
#                 df['mentioned_weapons'] = [', '.join(r['mentioned_weapons']) for r in results]
#                 df['odcr_report'] = [r['odcr'] for r in results]

#                 df.to_csv(output_csv, index=False )
#                 print(f"Results saved to {output_csv}")

#             except Exception as e:
#                 print(f"Error processing CSV: {e}")
