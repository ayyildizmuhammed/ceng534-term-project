import pandas as pd
import evaluate
from transformers import pipeline

def load_sentiment_pipeline(model_path):
    """
    Loads a text-classification pipeline for sentiment analysis.
    model_path can be:
      - A pretrained model on HF Hub (e.g. "savasy/bert-base-turkish-sentiment-cased")
      - A local path to a fine-tuned model (e.g. "models/fewshot_turkish_sentiment")
    """
    return pipeline("text-classification", model=model_path, tokenizer=model_path)

def main():
    # 1) Test set CSV => "text,label" (e.g. "positive"/"negative")
    test_file = "data/test/test_sentiment.csv"
    test_df = pd.read_csv(test_file)

    # Örnek: text,label
    # "Bu ürünü gerçekten çok seviyorum!", "positive"
    # "Bu film çok sıkıcıydı.", "negative"

    # 2) Map string labels ("positive"/"negative") to integer (1/0) for evaluate
    label_map = {"positive": 1, "negative": 0}
    test_df["label_int"] = test_df["label"].map(label_map)
    references = test_df["label_int"].tolist()

    # 3) Model paths
    # old_model_path => pretrained or older model
    # new_model_path => few-shot fine-tuned model
    old_model_path = "savasy/bert-base-turkish-sentiment-cased"  # örnek
    new_model_path = "models/fewshot_bert-base-turkish-sentiment-cased"          # fine-tuned path

    # 4) Load pipelines
    old_pipe = load_sentiment_pipeline(old_model_path)
    new_pipe = load_sentiment_pipeline(new_model_path)

    # 5) Evaluate kütüphanesiyle metrikler
    acc_metric = evaluate.load("accuracy")
    # prec_metric = evaluate.load("precision")
    # rec_metric = evaluate.load("recall")
    # f1_metric = evaluate.load("f1")

    # 6) Predictions
    old_preds = []
    new_preds = []

    possible_label_map = {
        "LABEL_0": 0, 
        "LABEL_1": 1,
        "positive": 1,
        "negative": 0,
        "POSITIVE": 1,
        "NEGATIVE": 0
    }

    for i, row in test_df.iterrows():
        text = row["text"]

        # Old model output
        old_out = old_pipe(text)[0]  # e.g. {'label': 'positive', 'score': 0.98}
        old_label_str = old_out["label"]
        old_label_int = possible_label_map.get(old_label_str, 0)  # default 0 if not found
        old_preds.append(old_label_int)

        # New model output
        new_out = new_pipe(text)[0]
        new_label_str = new_out["label"]
        new_label_int = possible_label_map.get(new_label_str, 0)
        new_preds.append(new_label_int)

    # 7) Metriği compute edecek fonksiyon
    def compute_metrics(predictions, references):
        # (1) Accuracy
        acc = acc_metric.compute(predictions=predictions, references=references)["accuracy"]
        # (2) Precision
        # prec = prec_metric.compute(predictions=predictions, references=references, average="binary")["precision"]
        # # (3) Recall
        # rec = rec_metric.compute(predictions=predictions, references=references, average="binary")["recall"]
        # # (4) F1
        # f1 = f1_metric.compute(predictions=predictions, references=references, average="binary")["f1"]
        return acc

    if len(set(references)) < 2:
        print("Test set has only one class. F1 won't be meaningful.")
        # Yukarıdaki kısımları yakalayarak minimal bir rapor verebilirsiniz.
        # Burada basitçe exit yapıyoruz
        return

    # 8) Old model metrikleri
    old_acc = compute_metrics(old_preds, references)
    # 9) New model metrikleri
    new_acc = compute_metrics(new_preds, references)

    # 10) Rapor
    print("=== OLD MODEL (Pretrained/Old) ===")
    print(f"Accuracy : {old_acc:.4f}")
    # print(f"Precision: {old_prec:.4f}")
    # print(f"Recall   : {old_rec:.4f}")
    # print(f"F1       : {old_f1:.4f}")

    print("\n=== NEW MODEL (Few-Shot Fine-tuned) ===")
    print(f"Accuracy : {new_acc:.4f}")
    # print(f"Precision: {new_prec:.4f}")
    # print(f"Recall   : {new_rec:.4f}")
    # print(f"F1       : {new_f1:.4f}")

if __name__ == "__main__":
    main()
