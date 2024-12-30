import pandas as pd
from transformers import pipeline
import sacrebleu
import evaluate

def load_model(model_path):
    return pipeline(
        task="translation", 
        model=model_path, 
        tokenizer=model_path
    )

def main():
    test_df = pd.read_csv("data/test/test_translation_it.csv")
    test_df["source_text"] = test_df["src_text"]
    test_df["target_text"] = test_df["dst_text"]
    
    old_model_path = "Helsinki-NLP/opus-mt-tc-big-itc-tr"   # huggingface
    new_model_path = "models/fewshot_opus-mt-tc-big-itc-tr"

    old_model = load_model(old_model_path)
    new_model = load_model(new_model_path)

    old_model_translations = []
    new_model_translations = []
    references = []  # Gerçek Türkçe metinler

    for i, row in test_df.iterrows():
        english_text = row["source_text"]   # Kaynak metin
        turkish_ref = row["target_text"]    # Referans

        # Eski model çevirisi
        old_result = old_model(english_text)[0]["translation_text"]
        old_model_translations.append(old_result)

        # Yeni (fine-tuned) model çevirisi
        new_result = new_model(english_text)[0]["translation_text"]
        new_model_translations.append(new_result)

        # Referans
        references.append(turkish_ref)

    bleu_metric = evaluate.load("bleu")
    chrf_metric = evaluate.load("chrf")
    meteor_metric = evaluate.load("meteor")
 
    references_for_evaluate = [[r] for r in references]

    old_bleu = bleu_metric.compute(predictions=old_model_translations, references=references_for_evaluate)
    old_meteor = meteor_metric.compute(predictions=old_model_translations, references=references_for_evaluate)

    new_bleu = bleu_metric.compute(predictions=new_model_translations, references=references_for_evaluate)
    new_meteor = meteor_metric.compute(predictions=new_model_translations, references=references_for_evaluate)

    old_model_bleu_sacre = sacrebleu.corpus_bleu(old_model_translations, [references])
    new_model_bleu_sacre = sacrebleu.corpus_bleu(new_model_translations, [references])

    print("=== OLD MODEL (Pretrained) SCORES ===")
    print(f"BLEU (huggingface evaluate): {old_bleu['bleu']:.4f}")
    print(f"METEOR: {old_meteor['meteor']:.4f}")
    print(f"BLEU (sacrebleu) => {old_model_bleu_sacre.score:.2f}")

    print("\n=== NEW MODEL (Few-Shot Fine-tuned) SCORES ===")
    print(f"BLEU (huggingface evaluate): {new_bleu['bleu']:.4f}")
    print(f"METEOR: {new_meteor['meteor']:.4f}")
    print(f"BLEU (sacrebleu) => {new_model_bleu_sacre.score:.2f}")

if __name__ == "__main__":
    main()
