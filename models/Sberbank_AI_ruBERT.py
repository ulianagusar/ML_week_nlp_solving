import torch
import pandas as pd
import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, 
                          Trainer, TrainingArguments)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset (replace with your path)
df = pd.read_csv('/content/merged_clean_data.csv')
df = df[['clean_message', 'exp']]  # Select only relevant columns

# Balance classes by undersampling the majority class
min_class_size = df['exp'].value_counts().min()
df_balanced = df.groupby('exp').apply(lambda x: x.sample(min_class_size, random_state=42)).reset_index(drop=True)

# Add 500 extra samples to class 'exp = 0'
df_extra = df[df['exp'] == 0].sample(500, random_state=42)
df_balanced = pd.concat([df_balanced, df_extra], ignore_index=True)

# Split into train, validation, and test sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df_balanced['clean_message'], df_balanced['exp'], test_size=0.2, random_state=42, stratify=df_balanced['exp']
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.4, random_state=42, stratify=temp_labels  # 0.4 * 0.2 = 8%
)

# Load tokenizer and model
model_name = "sberbank-ai/ruBert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

# Create Hugging Face datasets
train_dataset = Dataset.from_dict({'text': train_texts.tolist(), 'labels': train_labels.tolist()})
val_dataset = Dataset.from_dict({'text': val_texts.tolist(), 'labels': val_labels.tolist()})
test_dataset = Dataset.from_dict({'text': test_texts.tolist(), 'labels': test_labels.tolist()})

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set dataset format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load classification model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.1,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    learning_rate=3e-5,
    lr_scheduler_type='cosine',
    load_best_model_at_end=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Evaluate the model on test set
results = trainer.evaluate(test_dataset)
print("Test Results:", results)

# Generate classification reports
y_true_train = train_labels.tolist()
y_pred_train = trainer.predict(train_dataset).predictions.argmax(axis=1)
print("\n# Train Classification Report:")
print(classification_report(y_true_train, y_pred_train))

y_true_test = test_labels.tolist()
y_pred_test = trainer.predict(test_dataset).predictions.argmax(axis=1)
print("\n# Test Classification Report:")
print(classification_report(y_true_test, y_pred_test))


# # Train Classification Report:
#               precision    recall  f1-score   support

#            0       0.99      0.97      0.98      2829
#            1       0.96      0.99      0.97      2430

#     accuracy                           0.98      5259
#    macro avg       0.98      0.98      0.98      5259
# weighted avg       0.98      0.98      0.98      5259


# # Test Classification Report:
#               precision    recall  f1-score   support

#            0       0.94      0.93      0.93       283
#            1       0.92      0.93      0.92       243

#     accuracy                           0.93       526
#    macro avg       0.93      0.93      0.93       526
# weighted avg       0.93      0.93      0.93       526