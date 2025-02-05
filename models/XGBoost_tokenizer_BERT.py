import torch
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load dataset and select relevant columns
df = pd.read_csv('/content/merged_clean_data.csv')[["clean_message", "exp"]]

# Balance the dataset by undersampling majority classes
min_class_size = df['exp'].value_counts().min()
df_balanced = df.groupby('exp').apply(lambda x: x.sample(min_class_size, random_state=42)).reset_index(drop=True)

# Add extra samples for class `exp = 0` to improve balance
df_extra = df[df['exp'] == 0].sample(500, random_state=42)
df_balanced = pd.concat([df_balanced, df_extra], ignore_index=True)

# Split data into train, validation, and test sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df_balanced["clean_message"], df_balanced["exp"], test_size=0.2, stratify=df_balanced["exp"], random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.4, stratify=temp_labels, random_state=42  # 0.4 * 0.2 = 8%
)

# Load pre-trained BERT tokenizer and model
model_name = "bodomerka/Milytary_exp_class_classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Function to extract embeddings from text using BERT
def get_embeddings(texts, tokenizer, model, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embeddings)

# Generate embeddings for train, validation, and test sets
train_embeddings = get_embeddings(train_texts.tolist(), tokenizer, model)
val_embeddings = get_embeddings(val_texts.tolist(), tokenizer, model)
test_embeddings = get_embeddings(test_texts.tolist(), tokenizer, model)

# Define XGBoost classifier with tuned hyperparameters
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic", eval_metric="logloss", max_depth=2,
    learning_rate=0.01, n_estimators=300, subsample=0.5,
    colsample_bytree=0.5, gamma=17, reg_lambda=17.0,
    reg_alpha=17, min_child_weight=8, use_label_encoder=False,
    random_state=13, enable_categorical=False
)

# Train XGBoost model
evals_result = {}
xgb_model.fit(
    train_embeddings, train_labels,
    eval_set=[(train_embeddings, train_labels), (test_embeddings, test_labels)],
    verbose=50
)

# Evaluate model performance
train_preds = xgb_model.predict(train_embeddings)
test_preds = xgb_model.predict(test_embeddings)

print(f"Train Accuracy: {accuracy_score(train_labels, train_preds):.4f}")
print(f"Test Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
print(f"Train Precision: {precision_score(train_labels, train_preds):.4f}")
print(f"Test Precision: {precision_score(test_labels, test_preds):.4f}")
print(f"Train Recall: {recall_score(train_labels, train_preds):.4f}")
print(f"Test Recall: {recall_score(test_labels, test_preds):.4f}")
print(f"Train F1 Score: {f1_score(train_labels, train_preds):.4f}")
print(f"Test F1 Score: {f1_score(test_labels, test_preds):.4f}")

# Print classification report
print("Train Classification Report:")
print(classification_report(train_labels, train_preds))
print("Test Classification Report:")
print(classification_report(test_labels, test_preds))

# Save model to JSON format
xgb_model.save_model("xgb_model.json")

# Train Classification Report:
#               precision    recall  f1-score   support

#            0       0.96      0.94      0.95      2829
#            1       0.93      0.95      0.94      2430

#     accuracy                           0.94      5259
#    macro avg       0.94      0.94      0.94      5259
# weighted avg       0.94      0.94      0.94      5259

# Test Classification Report:
#               precision    recall  f1-score   support

#            0       0.96      0.94      0.95       283
#            1       0.94      0.95      0.95       243

#     accuracy                           0.95       526
#    macro avg       0.95      0.95      0.95       526
# weighted avg       0.95      0.95      0.95       526