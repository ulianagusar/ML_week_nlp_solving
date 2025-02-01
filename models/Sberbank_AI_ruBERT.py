import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from huggingface_hub import login, create_repo, upload_folder

# Load and preprocess dataset
# Ensure the dataset is properly formatted and contains required columns
# Replace 'df_new.csv' with the appropriate file path

df = pd.read_csv('df_new.csv')
df = df[["clean_message", "exp"]]  # Select relevant columns only

# Split the dataset into train, validation, and test sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df["clean_message"], df["exp"], test_size=0.2, random_state=42, stratify=df["exp"]
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.4, random_state=42, stratify=temp_labels
)  # 0.4 * 0.2 = 8%

# Load the tokenizer for the pre-trained model
model_name = "sberbank-ai/ruBert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

# Create Hugging Face datasets
train_dataset = Dataset.from_dict({"text": train_texts.tolist(), "labels": train_labels.tolist()})
val_dataset = Dataset.from_dict({"text": val_texts.tolist(), "labels": val_labels.tolist()})
test_dataset = Dataset.from_dict({"text": test_texts.tolist(), "labels": test_labels.tolist()})

# Apply tokenization to all datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Format datasets for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Compute class weights for handling imbalanced datasets
class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=df["exp"].values)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Custom trainer class to use weighted loss function
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.1,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    load_best_model_at_end=True
)

# Initialize the trainer with custom class weights
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
results = trainer.evaluate(test_dataset)
print(results)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Predictions on the validation set
predictions = trainer.predict(val_dataset)
logits = predictions.predictions  # Model's raw outputs
labels = predictions.label_ids  # True labels

# Convert logits to predicted labels
preds = np.argmax(logits, axis=-1)

# Compute evaluation metrics
accuracy = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

# Display evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Prediction function for single text input
def predict_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

    # Move inputs to the model's device
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Get predictions without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits and predict the class label
    logits = outputs.logits
    pred = torch.argmax(logits, dim=-1).item()
    return pred

# Example text prediction
text_input = "Війська РФ продовжують наступ на південному фронті."
prediction = predict_text(text_input)
print(f"Predicted label for the input text: {prediction}")

# Authenticate with Hugging Face Hub
login(token="your_token")

# Create a new repository on Hugging Face
create_repo("Milytary_exp_class_classification_sber_ai_based")

# Save the model and tokenizer locally
model_name = "bodomerka/Milytary_exp_class_classification_sber_ai_based"
model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)

# Upload model and tokenizer to Hugging Face Model Hub
upload_folder(repo_id=model_name, folder_path=model_name)