# compare_translation.py

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
    # 1) Test seti CSV
    test_df = pd.read_csv("data/test/test_translation.csv")
    test_df["source_text"] = test_df["English Title"] + " " + test_df["English Summary"]
    test_df["target_text"] = test_df["Turkish Title"] + " " + test_df["Turkish Summary"]
    
    # Kolon adlarının "English", "Turkish" (yani reference) olduğunu varsayıyoruz
    # ya da "English Title"/"Turkish Title" olabilir, projenizde nasıl tanımladıysanız ona göre düzenleyin.

    # 2) İki model: Eski (pretrained) ve yeni (fine-tuned)
    old_model_path = "Helsinki-NLP/opus-mt-tc-big-en-tr"   # huggingface
    new_model_path = "models/fewshot_opus-mt-tc-big-en-tr"

    old_model = load_model(old_model_path)
    new_model = load_model(new_model_path)

    # 3) Çevirileri alıp listelerde saklayacağız
    old_model_translations = []
    new_model_translations = []
    references = []  # Gerçek Türkçe metinler

    # 4) Her satır için inference
    for i, row in test_df.iterrows():
        english_text = row["source_text"]  # Kaynak metin
        turkish_ref = row["target_text"]   # Referans
        
        # Eski model çevirisi
        old_result = old_model(english_text)[0]["translation_text"]
        old_model_translations.append(old_result)

        # Yeni model çevirisi
        new_result = new_model(english_text)[0]["translation_text"]
        new_model_translations.append(new_result)

        # Referans
        references.append(turkish_ref)

    # 5) BLEU skorunu hesaplama
    # sacrebleu.corpus_bleu, parametre olarak "hipotezler" ve [["ref1", "ref2", ...]] formatında referans listesi alır.
    # Tek referans setiniz varsa, nested list şekilde veriyoruz.
    old_model_bleu = sacrebleu.corpus_bleu(old_model_translations, [references])
    new_model_bleu = sacrebleu.corpus_bleu(new_model_translations, [references])
    
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=old_model_translations, references=[[r] for r in references])
    print(results)  # {'bleu': 0.32, 'precisions': [...], 'brevity_penalty':..., 'length_ratio':..., 'translation_length':..., 'reference_length':...}

    print(f"Old Model BLEU: {old_model_bleu.score:.2f}")
    print(f"New Model BLEU: {new_model_bleu.score:.2f}")

if __name__ == "__main__":
    main()
