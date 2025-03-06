#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 00:12:20 2025

@author: charliemurray
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 00:23:16 2025

@author: charliemurray
"""
"""This code builds on the first model by implementing Optuna- a Bayseian optimisation method for hyperparameter tuning - lets see whether the accuracy improves"""

import torch
import optuna
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------------------------------
# 1. Check GPU Availability
# -------------------------------
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# -------------------------------
# 2. Load Dataset (LIAR)
# -------------------------------
dataset = load_dataset('liar')

# -------------------------------
# 3. Tokenize Data
# -------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['statement'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert dataset to PyTorch format
train_dataset = tokenized_datasets['train'].with_format("torch")
val_dataset = tokenized_datasets['validation'].with_format("torch")
test_dataset = tokenized_datasets['test'].with_format("torch")

# -------------------------------
# 4. Load Pretrained DistilBERT Model
# -------------------------------
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
model.to(device)

# -------------------------------
# 5. Define Evaluation Metrics
# -------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# -------------------------------
# 6. Optuna Hyperparameter Optimization
# -------------------------------
def objective(trial):
    """Function for Optuna to optimize hyperparameters."""
    
    # Sample hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 3e-5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_int("epochs", 3, 5)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        report_to="none",
        load_best_model_at_end=True,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()
    
    # Evaluate the model
    eval_result = trainer.evaluate()
    
    return eval_result["eval_accuracy"]  # Optimize for accuracy

# Run Optuna
study = optuna.create_study(direction="maximize")  # Maximize accuracy
study.optimize(objective, n_trials=10)  # Run 10 trials

# Print best parameters
print(f"Best Hyperparameters: {study.best_params}")

# -------------------------------
# 7. Train with Best Hyperparameters
# -------------------------------
best_params = study.best_params  # Get best hyperparameters

training_args = TrainingArguments(
    output_dir="./best_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_params["learning_rate"],
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"],
    num_train_epochs=best_params["epochs"],
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none",
    load_best_model_at_end=True,
)

# Define final Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train with best parameters
trainer.train()

# -------------------------------
# 8. Evaluate & Save Model
# -------------------------------
results = trainer.evaluate()
print(results)

# Save the final model
model.save_pretrained('./liar-fake-news-detector')
tokenizer.save_pretrained('./liar-fake-news-detector')

# -------------------------------
# 9. Visualize Optuna Optimization Process
# -------------------------------
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.show()

optuna.visualization.matplotlib.plot_param_importances(study)
plt.show()

