# src/train_scripts/train_fewshot_sentiment.py

import pandas as pd
import numpy as np
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import evaluate

def prepare_sentiment_dataset(csv_file: str):
    """
    Reads the CSV file and creates a Hugging Face Dataset object 
    for sentiment analysis.
    csv_file: path to the CSV file with columns:
        Text, Label
    """
    df = pd.read_csv(csv_file)

    # Normalize labels if necessary (e.g., lowercase)
    df["label"] = df["label"].str.lower()

    # Map labels to integers if they are not already
    label_mapping = {"negative": 0, "positive": 1}
    df["label"] = df["label"].map(label_mapping)

    dataset = Dataset.from_pandas(df)
    return dataset

def train_fewshot_sentiment(
    model_name,
    csv_file="data/train/fewshot_sentiment_sample.csv",
    output_dir="models/fewshot_bert-base-turkish-sentiment-cased",
    max_train_samples=None,
    num_train_epochs=3,
    batch_size=2,
    learning_rate=5e-5
):
    """
    Trains (fine-tunes) a sentiment analysis model with a small (few-shot) dataset.
    
    model_name: The Hugging Face model repo or local path 
    csv_file:   Path to your CSV dataset
    output_dir: Directory where the model checkpoints will be saved
    """
    
    # 1) Veri setini hazırla
    dataset = prepare_sentiment_dataset(csv_file)

    if max_train_samples:
        dataset = dataset.select(range(max_train_samples))
    
    # 2) Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 3) Tokenize fonksiyonu
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)
    
    # 4) Dataset'i tokenize et
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    # 5) Data collator (padding için)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 6) Metric tanımla
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # 7) Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_steps=100,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )
    
    # 8) Trainer oluştur
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,  # Basitlik için eğitim seti üzerinde değerlendirme
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # 9) Eğitimi başlat
    trainer.train()
    
    # 10) Modeli kaydet
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"[INFO] Sentiment model saved to {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    # Örnek kullanım:
    output_model_path = train_fewshot_sentiment(
        model_name="savasy/bert-base-turkish-sentiment-cased",  # Türkçe destekleyen bir BERT modeli
        csv_file="data/train/fewshot_sentiment_sample.csv",      # Sentiment veri setinizin yolu
        output_dir="models/fewshot_bert-base-turkish-sentiment-cased",
        max_train_samples=None,  # Few-shot için örnek sayısını belirleyebilirsiniz (örneğin 50)
        num_train_epochs=3,
        batch_size=2,
        learning_rate=5e-5
    )
    print(f"Finished training. Model path: {output_model_path}")
