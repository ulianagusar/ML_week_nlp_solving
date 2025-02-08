

import torch
import xgboost as xgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pymorphy2 import MorphAnalyzer


def experience_bert(text, model_name="bodomerka/Milytary_exp_class_classification_sber_ai_based"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        return torch.argmax(outputs.logits).item()

def experience_xg_boost(text, xgb_model_path="xgb_model.ubj", bert_model_name="bodomerka/Milytary_exp_class_classification_sber_ai_based"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        transformer_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformer_model.to(device)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = transformer_model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
        
        xgb_model = xgb.Booster()
        xgb_model.load_model(xgb_model_path)
        dmatrix = xgb.DMatrix(embeddings)
        return int(xgb_model.predict(dmatrix)[0] > 0.5)
    except Exception as e:
        print(f"Error in experience_xg_boost: {e}")
        return 0