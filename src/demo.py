# src/main.py

from config import CONFIG
from collectors.rss_collector import RSSCollector
from translators.translator import Translator
from analyzers.sentiment_analyzer import SentimentAnalyzer
from analyzers.keyword_analyzer import KeywordAnalyzer
from preprocessors.text_preprocessor import TextPreprocessor
from utils.result_persister import save_results_to_csv, save_results_to_json
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="logs/experiment.log",  # Log dosyası
    filemode="w"
)

def main():
    """
    Main function:
    1) Collecting RSS news (Step 1).
    2) Preprocessing the text (cleaning step).
    3) Counting Turkey-related keywords for each language (optional step).
    4) Translating to Turkish & NER (Step 2).
    5) Performing Sentiment Analysis on the translated text (Step 3).
    6) Save experiment results to CSV/JSON.
    """

    # Initialize Preprocessor with config
    text_preprocessor = TextPreprocessor(
        chars_to_remove=CONFIG["PREPROCESSOR"]["CHARS_TO_REMOVE"]
    )

    # Step 1: Collect all news from RSS feeds
    all_news_data = []
    for feed_info in CONFIG["RSS_FEEDS"]:
        rss_url = feed_info["url"]
        language = feed_info["language"]

        print(f"\n[INFO] Collecting news from: {rss_url} (Language: {language})")

        collector = RSSCollector(rss_url=rss_url, language=language)
        news_data = collector.fetch_news()
        print(f"[INFO] Fetched {len(news_data)} news entries.")

        # Combine into a single list
        all_news_data.extend(news_data)

    # Step 2: Preprocess each news text
    for item in all_news_data:
        original_news_text = item.get("news_text", "")
        cleaned_text = text_preprocessor.clean_text(original_news_text)
        item["news_text"] = cleaned_text

    # Step 3: Keyword analyzer
    keyword_analyzer = KeywordAnalyzer(keyword_data_path=CONFIG["TURKEY_KEYWORDS_PATH"])
    
    # Step 4: Translator & NER
    translator = Translator(
        translation_models=CONFIG["TRANSLATION_MODELS"],
        ner_model=CONFIG["NER_MODEL"]
    )

    # Step 5: Sentiment Analyzer
    sentiment_analyzer = SentimentAnalyzer(
        model_name=CONFIG["SENTIMENT_MODEL"]
    )

    # We'll just show an example on the last 5 items
    experiment_results = []
    print("\n[INFO] Starting translation, entity recognition, and sentiment analysis...\n")
    for i, item in enumerate(all_news_data[-5:]):
        source_lang = item["language"]
        original_text = item["news_text"]

        print(f"--- News Item {i+1} ({source_lang}) ---")
        print(f"Original Text (Cleaned): {original_text}")

        turkey_score = keyword_analyzer.count_turkey_keywords(original_text, source_lang)

        translated_text = None
        entities = []
        sentiment_result = None

        # if source_lang in CONFIG["TRANSLATION_MODELS"]:
        translated_text, entities = translator.translate_and_identify(
            text=original_text,
            source_lang=source_lang
        )

        print(f"Translated Text (Turkish): {translated_text}")
        print("Entities found:")
        for ent in entities:
            print(f"  - {ent['entity']} ({ent['type']}) [score: {ent['score']:.2f}]")

        sentiment_result = sentiment_analyzer.analyze_sentiment(translated_text)
        if sentiment_result:
            print(f"Sentiment: {sentiment_result['label']} | Score: {sentiment_result['score']:.2f}")
        else:
            print("No sentiment result.")
        # else:
        #     print("No translation model available for this language. Skipping translation & sentiment analysis...")

        # Step 6: Prepare experiment result dict
        entities_str = ", ".join(f"{e['entity']}({e['type']})" for e in entities)

        result_dict = {
            "original_text": item.get("description", ""),
            "cleaned_text": original_text,
            "source_lang": source_lang,
            "turkey_score": turkey_score,  # Kaç keyword bulduk?
            "translated_text": translated_text,
            "entities": entities_str,  # CSV'de virgüllü string
            "sentiment_label": sentiment_result["label"] if sentiment_result else None,
            "sentiment_score": sentiment_result["score"] if sentiment_result else None
        }
        experiment_results.append(result_dict)

        print()

    # Kaydetme adımı
    save_results_to_csv(experiment_results, filename="data/experiment_results.csv")
    save_results_to_json(experiment_results, filename="data/experiment_results.json")


if __name__ == "__main__":
    main()
