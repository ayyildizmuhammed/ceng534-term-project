import logging
import re
import pandas as pd

class KeywordAnalyzer:
    """
    KeywordAnalyzer class is responsible for counting the occurrences of specific keywords in a given text.
    """

    def __init__(self, keyword_data_path: str):
        """
        Constructor to initialize the KeywordAnalyzer with a list of keywords to search for.

        :param keywords: A list of keywords to search for in the text
        """
    
        #read via pandas
        self.keywords = pd.read_csv(keyword_data_path)

    def count_turkey_keywords(self, text: str, source_lang: str) -> int:
        """
        Counts how many times any of the given keywords appear in the text (case-insensitive).
        Returns the total match count.
        """
        lang_keywords = self.keywords[source_lang]
        if len(lang_keywords) == 0:
            logging.warning(f"No keywords found for language: {source_lang}")
            return 0

        if not text:
            logging.warning("Empty text received for keyword analysis")
            return 0

        text_lower = text.lower()
        total_count = 0

        for kw in lang_keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            matches = re.findall(pattern, text_lower)
            total_count += len(matches)

        return total_count
