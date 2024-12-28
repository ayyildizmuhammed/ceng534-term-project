import logging
import re


class KeywordAnalyzer:
    """
    KeywordAnalyzer class is responsible for counting the occurrences of specific keywords in a given text.
    """

    def __init__(self, keywords: str):
        """
        Constructor to initialize the KeywordAnalyzer with a list of keywords to search for.

        :param keywords: A list of keywords to search for in the text
        """
        if not keywords:
            logging.error("No keywords provided for keyword analysis.")
            raise ValueError("No keywords provided for keyword analysis.")

        self.keywords = keywords

    def count_turkey_keywords(self, text: str, source_lang: str) -> int:
        """
        Counts how many times any of the given keywords appear in the text (case-insensitive).
        Returns the total match count.
        """
        keywords = self.keywords.get(source_lang, [])
        if len(keywords) == 0:
            logging.warning(f"No keywords found for language: {source_lang}")
            return 0

        if not text or not self.keywords:
            logging.warning("Empty text or keywords received for keyword analysis")
            return 0

        text_lower = text.lower()
        total_count = 0

        for kw in self.keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            matches = re.findall(pattern, text_lower)
            total_count += len(matches)

        return total_count
