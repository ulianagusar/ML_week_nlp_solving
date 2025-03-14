from flask import Flask, request, jsonify
import torch
import xgboost as xgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os

app = Flask(__name__)


MODEL_NAME = "bodomerka/Milytary_exp_class_classification_sber_ai_based"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

model_path = os.path.join(os.path.dirname(__file__), "xgb_model.json")

xgb_model = xgb.Booster()
xgb_model.load_model(model_path)

@app.route('/predict/bert', methods=['POST'])
def predict_bert():
    print("here3")
    data = request.json
    text = data.get("text", "")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        prediction = torch.argmax(outputs.logits).item()

    return jsonify({"prediction": prediction})

@app.route('/predict/xgboost', methods=['POST'])
def predict_xgboost():
    data = request.json
    text = data.get("text", "")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()

    dmatrix = xgb.DMatrix(embeddings)
    prediction = int(xgb_model.predict(dmatrix)[0] > 0.5)

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
