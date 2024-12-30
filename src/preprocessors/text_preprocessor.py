# src/preprocessors/text_preprocessor.py
import re


class TextPreprocessor:
    """
    TextPreprocessor class is responsible for cleaning the news text
    by removing unwanted characters or applying other transformations.
    """

    def __init__(self, chars_to_remove=None):
        """
        :param chars_to_remove: A list of characters that should be removed from the text.
        """
        if chars_to_remove is None:
            chars_to_remove = []
        self.chars_to_remove = chars_to_remove

    def clean_text(self, text: str) -> str:
        """
        Removes unwanted characters from the text. Returns the cleaned string.
        
        :param text: The original text to be cleaned
        :return: The cleaned text
        """
        # If text is None or empty, return it as is
        if not text:
            return text

        cleaned = text
        for char in self.chars_to_remove:
            cleaned = cleaned.replace(char, "")

        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned
