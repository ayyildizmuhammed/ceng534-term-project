# CENG534 DL4NLP Term Project

A multi-language RSS pipeline that collects news data, cleans and translates them into Turkish, performs Named Entity Recognition (NER), and applies sentiment analysis on the translated text.

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Project Structure](#project-structure)  
5. [Usage](#usage)  
6. [Configuration (config.py)](#configuration-configpy)  
7. [Todo & Possible Improvements](#todo--possible-improvements)  
8. [License](#license)

---

## Overview

This project is a pipeline that fetches multi-language RSS news, translates them into Turkish, optionally detects Turkey-related keywords, and applies sentiment analysis to the translated text. The main steps are:

1. **RSSCollector** – Fetches news data (in various languages) from specified RSS feeds.  
2. **TextPreprocessor** – Cleans unnecessary or special characters (%, …, “, etc.) in the news text.  
3. **KeywordAnalyzer** *(optional)* – Detects or counts certain keywords (e.g., Turkey-related terms).  
4. **Translator** – Translates multi-language content into Turkish (e.g., using MBART or similar) and performs Named Entity Recognition (NER).  
5. **SentimentAnalyzer** – Classifies the translated Turkish text into positive or negative sentiment.  
6. **ResultPersister** – Saves the final results to CSV/JSON files.

Additionally, there are **few-shot fine-tuning** scripts (train files) for models, so you can refine MBART (for translation) or your Turkish sentiment model with a small dataset, then **compare** them if needed (in separate scripts).

---

## Features

- **RSSCollector**: Collects RSS data from multiple sources in different languages.  
- **Preprocessing**: Removes undesirable characters from the collected news text.  
- **(Optional) Keyword Analysis**: Checks or counts specified keywords (e.g., Turkey-related).  
- **Translation + NER**: Translates text into Turkish and detects named entities.  
- **Sentiment Analysis**: Classifies the final Turkish text into positive/negative.  
- **Result Storage**: Logs outcome into CSV/JSON for further analysis.

---

## Installation

1. **Create and Activate a Virtual Environment**  
   ```bash
   python3 -m venv ceng534-env
   source ceng534-env/bin/activate
   ```

2. **Install Dependencies**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *Ensure that `transformers`, `datasets`, `evaluate`, `feedparser`, etc., are listed in `requirements.txt`.*

3. **(Optional) Fine-Tuning Scripts**  
   - If you plan to run the **few-shot** scripts for translation or sentiment, check the `train_*.py` files and adjust the dataset paths accordingly (e.g., `data/fewshot_*.csv`).

---

## Project Structure

```plaintext
.
├── README.md
├── data
│   ├── test
│   │   ├── test_sentiment.csv
│   │   └── test_translation.csv
│   ├── fewshot_sentiment.csv
│   ├── fewshot_translation.csv
├── logs
│   └── experiment.log
├── models
│   ├── fewshot_opus-mt-tc-big-en-tr
│   ├── mbart_fewshot_tr
│   └── bert_turkish_sentiment_fewshot
├── requirements.txt
└── src
    ├── analyzers
    │   ├── keyword_analyzer.py
    │   └── sentiment_analyzer.py
    ├── collectors
    │   └── rss_collector.py
    ├── config.py
    ├── main.py
    ├── preprocessors
    │   └── text_preprocessor.py
    ├── translators
    │   └── translator.py
    ├── trainers
    │   ├── train_fewshot_sentiment.py
    │   ├── train_fewshot_translation.py
    ├── utils
    │   ├── result_persister.py
    └── ...
```

- **collectors/**: Contains `rss_collector.py` for news retrieval.  
- **preprocessors/**: Includes scripts to clean or preprocess text.  
- **analyzers/**: Houses `sentiment_analyzer.py` and possibly `keyword_analyzer.py`.  
- **translators/**: Scripts related to translation (MBART, NER integration, etc.).  
- **trainers/**: Fine-tuning scripts for translation or sentiment.  
- **utils/**: Helper utilities for saving results, logging, etc.  
- **main.py**: Demonstrates the entire pipeline.  
- **config.py**: RSS feed definitions, model paths, any other configuration needed.

---

## Usage

1. **Run the Main Pipeline**  
   ```bash
   python src/main.py
   ```
   - This will sequentially:  
     - Collect news from RSS (RSSCollector)  
     - Clean the text (TextPreprocessor)  
     - (Optionally) analyze keywords (KeywordAnalyzer)  
     - Translate & NER (Translator)  
     - Perform Sentiment Analysis (SentimentAnalyzer)  
     - Save results (ResultPersister)  
   - Final results will appear in `data/experiment_results.csv` and `.json`.

2. **Few-Shot Fine-Tuning (Optional)**  
   - For MBART-based translation:
     ```bash
     python src/trainers/train_mbart_fewshot_translation.py
     ```
     This produces a new model under `models/mbart_fewshot_tr`.
   - For Turkish sentiment analysis:
     ```bash
     python src/trainers/train_turkish_sentiment.py
     ```
     This produces a new model under `models/bert_turkish_sentiment_fewshot`.
   - Afterwards, update `config.py` or your pipeline to use these newly fine-tuned models.

---

## Configuration (config.py)

Below is an example snippet of `CONFIG` in `config.py`:

```python
CONFIG = {
    "RSS_FEEDS": [
        {"url": "https://feeds.bbci.co.uk/news/world/europe/rss.xml", "language": "English"},
        {"url": "https://www.lemonde.fr/rss/une.xml", "language": "French"}
        # More feeds...
    ],
    "PREPROCESSOR": {
        "CHARS_TO_REMOVE": ["%", "…", "\n", "\r", "“", "”"]
    },
    "TURKEY_KEYWORDS_PATH": "data/keywords.txt",
    "TRANSLATION_MODELS": {
        "English": "Helsinki-NLP/opus-mt-tc-big-en-tr",
        # ...
    },
    "NER_MODEL": "savasy/bert-base-turkish-ner-cased",
    "SENTIMENT_MODEL": "savasy/bert-base-turkish-sentiment-cased"
}
```

Adjust these settings to point to your chosen translation/sentiment models or your fine-tuned versions.

> **Note**: You can modify or remove sections as needed. Make sure to update any references to your actual script names, model paths, or data files to keep everything consistent.
