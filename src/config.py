CONFIG = {
    "RSS_FEEDS": [
        {
            "url": "https://www.tagesschau.de/xml/rss2",
            "language": "German"
        }
    ],
    "RSS_FEEDS_RECOVERY": [
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
        "English": "Helsinki-NLP/opus-mt-tc-big-en-tr",
        # "French": "Helsinki-NLP/opus-mt-fr-tr",
        "German": "Helsinki-NLP/opus-mt-tc-big-gmq-tr"
        # or local paths, e.g. "English": "/path/to/local/en-tr-model"
    },
    # Model to be used for Named Entity Recognition (NER)
    "NER_MODEL": "savasy/bert-base-turkish-ner-cased",
    # Model to be used for sentiment analysis
    "SENTIMENT_MODEL": "savasy/bert-base-turkish-sentiment-cased",

    "PREPROCESSOR": {
        "CHARS_TO_REMOVE": ["%", "…", "\n", "\r", "“", "”"]
    },
    "TURKEY_KEYWORDS": {
        "English": ["Turkey", "Turkish", "Ankara", "Istanbul", "Erdogan"],
        "French": ["Turquie", "Turque", "Ankara", "Istanbul", "Erdogan"],
        "German": ["Türkei", "Ankara", "Istanbul", "Erdogan"],
        "Italian": ["Turchia", "Ankara", "Istanbul", "Erdogan"],
    }
}
