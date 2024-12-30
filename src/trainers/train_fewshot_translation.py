# train_fewshot_translation.py

import pandas as pd
import numpy as np
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

def prepare_dataset(csv_file: str):
    """
    Reads the CSV file and creates a Hugging Face Dataset object 
    for English -> Turkish translation.
    csv_file: path to the CSV file with columns:
        English Title, English Summary, Turkish Title, Turkish Summary
    """
    df = pd.read_csv(csv_file)

    # Birleştir: English Title + English Summary -> source_text
    #            Turkish Title + Turkish Summary -> target_text
    df["source_text"] = df["English Title"] + " " + df["English Summary"]
    df["target_text"] = df["Turkish Title"] + " " + df["Turkish Summary"]

    # Minimum örnek olsun diye dataframe’i direkt dataset’e çeviriyoruz
    dataset = Dataset.from_pandas(df)
    return dataset

def train_fewshot_translation(
    model_name="Helsinki-NLP/opus-mt-en-tr",
    csv_file="data/fewshot_sample.csv",
    output_dir="models/fewshot_en_tr",
    max_train_samples=None,
    num_train_epochs=3,
    batch_size=2
):
    """
    Trains (fine-tunes) a translation model with a small (few-shot) dataset.
    
    model_name: The Hugging Face model repo or local path 
    csv_file:   Path to your CSV dataset
    output_dir: Directory where the model checkpoints will be saved
    """

    # 1) Veri setini hazırla
    dataset = prepare_dataset(csv_file)

    if max_train_samples:
        dataset = dataset.select(range(max_train_samples))
    
    # 2) Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 3) Tokenize fonksiyonu
    def preprocess_function(examples):
        # 'source_text' -> model input
        # 'target_text' -> model output
        inputs = [ex for ex in examples["source_text"]]
        targets = [ex for ex in examples["target_text"]]
        
        model_inputs = tokenizer(
            inputs, 
            max_length=128, 
            truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=128, 
                truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 4) Dataset'i tokenize et
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 5) Data collator (Seq2Seq için special bir collator)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Bu örnekte tüm veri train olarak kabul ediyoruz (Few-shot)
    train_dataset = tokenized_dataset

    # 6) Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="no",  # Küçük veri seti, eval yok
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=100,
        logging_steps=5,
        predict_with_generate=True
    )

    # 7) Trainer oluştur
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 8) Eğitimi başlat
    trainer.train()

    # 9) Modeli kaydet
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"[INFO] Model saved to {output_dir}")

    return output_dir

if __name__ == "__main__":
    # Örnek kullanım:
    output_model_path = train_fewshot_translation(
        model_name="Helsinki-NLP/opus-mt-tc-big-en-tr",
        csv_file="data/train/fewshot_sample.csv",  # Sizin CSV yolunuz
        output_dir="models/fewshot_opus-mt-tc-big-en-tr",
        max_train_samples=None,  # Verinin tamamını kullan
        num_train_epochs=3,
        batch_size=1
    )
    print(f"Finished training. Model path: {output_model_path}")
