CONFIG = {
    "RSS_FEEDS": [
        {
            "url": "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
            "language": "English"
        },
        {
            "url": "https://www.theguardian.com/world/turkey/rss",
            "language": "English"
        },
        {
            "url": "https://www.tagesschau.de/xml/rss2",
            "language": "German"
        },
        {
            "url": "https://www.lemonde.fr/turquie/rss_full.xml",
            "language": "French"
        },
        {
            "url": "https://www.france24.com/en/europe/rss",
            "language": "French"
        },
        {
            "url": "https://www.ansa.it/sito/notizie/mondo/mondo_rss.xml ",
            "language": "Italian"
        },
        {
            "url": "https://www.repubblica.it/rss/esteri/rss2.0.xml",
            "language": "Italian"
        }  
    ],
    "TRANSLATION_MODELS": {
        "English": "models/fewshot_opus-mt-tc-big-en-tr",
        "Italian":"Helsinki-NLP/opus-mt-tc-big-itc-tr",
        "French->English": "Helsinki-NLP/opus-mt-fr-en",
        "German->English": "Helsinki-NLP/opus-mt-de-en"
        # or local paths, e.g. "English": "/path/to/local/en-tr-model"
    },
    # Model to be used for Named Entity Recognition (NER)
    "NER_MODEL": "savasy/bert-base-turkish-ner-cased",
    # Model to be used for sentiment analysis
    "SENTIMENT_MODEL": "models/fewshot_bert-base-turkish-sentiment-cased",
    # "SENTIMENT_MODEL": "savasy/bert-base-turkish-sentiment-cased",
    "FEWSHOT_SENTIMENT_MODEL": "models/fewshot_bert-base-turkish-sentiment-cased",
    

    "PREPROCESSOR": {
        "CHARS_TO_REMOVE": ["%", "…", "\n", "\r", "“", "”"]
    },
    "TURKEY_KEYWORDS_PATH": "data/keywords/turkey_keywords.csv",
    
    "BART_BASE_MODEL": "facebook/bart-large"
}
