import logging
from transformers import pipeline

class Translator:
    """
    Translator class handles:
    1) Translating text from a source language to Turkish using Hugging Face models.
    2) Identifying named entities in the translated Turkish text.
    """

    def __init__(self, translation_models: dict, ner_model: str):
        """
        Constructor for the Translator.
        
        :param translation_models: A dictionary mapping source languages to their respective model names/paths
        :param ner_model: The Hugging Face model name/path for Turkish NER
        """
        self.translation_models = translation_models
        self.ner_model = ner_model

    def translate_and_identify(self, text: str, source_lang: str):
        """
        Translates the given text from source_lang to Turkish, then 
        performs Named Entity Recognition (NER) on the translated text.
        
        :param text: The text to be translated
        :param source_lang: The language of the source text (e.g. English, French, German)
        :return: A tuple (translated_text, entities) where:
                 - translated_text is the text translated into Turkish
                 - entities is a list of detected named entities in the translated text
        """
        if not text:
            logging.warning("Empty text received for translation.")
            return "", []

        # 1) Select the translation model based on the source language
        model_name_or_path = self.translation_models.get(source_lang)
        if not model_name_or_path:
            logging.error(f"No translation model found for language: {source_lang}")
            return text, []

        # 2) Create a translation pipeline for the specified model
        translation_pipeline = pipeline(
            task="translation", 
            model=model_name_or_path
        )

        # 3) Translate the text into Turkish
        translation_result = translation_pipeline(text)
        translated_text = translation_result[0]["translation_text"]

        # 4) Create an NER pipeline for Turkish
        ner_pipeline = pipeline(
            task="token-classification",
            model=self.ner_model,
            aggregation_strategy="simple"  # Combine tokens into full entity spans
        )

        # 5) Perform NER on the translated text
        ner_results = ner_pipeline(translated_text)
        
        # 6) Extract relevant entity info
        entities = []
        for entity_item in ner_results:
            entities.append({
                "entity": entity_item["word"],
                "type": entity_item["entity_group"],
                "score": entity_item["score"]
            })

        return translated_text, entities
