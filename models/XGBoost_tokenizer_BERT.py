import torch
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress visualization
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def get_embeddings(texts, tokenizer, model, batch_size=32):
    """
    Generate embeddings for a list of texts using a pre-trained transformer model.
    
    Parameters:
    texts (list): List of input texts.
    tokenizer: Tokenizer for the model.
    model: Pre-trained transformer model.
    batch_size (int): Number of samples per batch.
    
    Returns:
    np.array: Embeddings for all input texts.
    """
    embeddings = []
    
    # Use tqdm to visualize progress
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        
        with torch.no_grad():  # Disable gradient calculation for efficiency
            outputs = model.bert(**inputs)
        
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Move tensors to CPU
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)  # Stack all embeddings into a single array


# Obtain vector representations for datasets
train_embeddings = get_embeddings(train_texts.tolist(), tokenizer, model)
val_embeddings = get_embeddings(val_texts.tolist(), tokenizer, model)
test_embeddings = get_embeddings(test_texts.tolist(), tokenizer, model)

# Initialize XGBoost classifier with predefined hyperparameters
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",  # Define evaluation metric here instead of .fit()
    max_depth=2,
    learning_rate=0.01,
    n_estimators=300,
    subsample=0.5,
    colsample_bytree=0.5,
    gamma=17,
    reg_lambda=17.0,
    reg_alpha=17,
    min_child_weight=8,
    use_label_encoder=False,
    random_seed=13
)

# Train the model and store evaluation results
xgb_model.fit(
    train_embeddings, train_labels,
    eval_set=[(train_embeddings, train_labels), (test_embeddings, test_labels)],
    verbose=50
)

# Evaluate the model on the test set
test_preds = xgb_model.predict(test_embeddings)
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Evaluate the model on the training set
train_preds = xgb_model.predict(train_embeddings)
train_accuracy = accuracy_score(train_labels, train_preds)
print(f"Train Accuracy: {train_accuracy:.4f}")

