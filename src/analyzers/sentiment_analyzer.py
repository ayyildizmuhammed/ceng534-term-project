import logging
from transformers import pipeline

class SentimentAnalyzer:
    """
    SentimentAnalyzer class is responsible for classifying 
    Turkish text into positive or negative sentiment.
    """

    def __init__(self, model_name: str):
        """
        Constructor for the SentimentAnalyzer.
        
        :param model_name: The Hugging Face model name or local path for sentiment analysis.
        """
        self.model_name = model_name
        # Create a sentiment-analysis pipeline
        try:
            self.sentiment_pipeline = pipeline(
                task="sentiment-analysis",
                model=self.model_name
            )
        except Exception as e:
            logging.error(f"Failed to load sentiment model '{self.model_name}'. Error: {str(e)}")
            self.sentiment_pipeline = None

    def analyze_sentiment(self, text: str):
        """
        Analyzes the sentiment of the given text and returns a dictionary
        containing the 'label' and 'score'.
        
        :param text: The text to be analyzed for sentiment
        :return: A dictionary with two keys:
                 'label' - 'POSITIVE' or 'NEGATIVE'
                 'score' - The confidence score of the classification
                 If the pipeline is not loaded or text is empty, returns None.
        """
        if not self.sentiment_pipeline:
            logging.error("Sentiment pipeline is not initialized.")
            return None

        if not text:
            logging.warning("Empty text received for sentiment analysis.")
            return None

        # Perform sentiment classification
        results = self.sentiment_pipeline(text)

        # The pipeline returns a list of dicts, for example:
        # [{'label': 'NEGATIVE', 'score': 0.999}]

        if results and len(results) > 0:
            return {
                "label": results[0]["label"],
                "score": results[0]["score"]
            }
        else:
            return None
