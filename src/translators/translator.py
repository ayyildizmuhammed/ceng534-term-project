import logging
from transformers import pipeline


class Translator:
    """
    Translator class handles:
    1) Translating text from a source language to Turkish using Hugging Face models.
    2) Identifying named entities in the translated Turkish text.
    """

    def __init__(self, translation_models: dict, ner_model: str, pivot_lang: str = "English"):
        """
        Constructor for the Translator.

        :param translation_models: A dictionary mapping source languages to their respective model names/paths.
               Example:
               {
                 "English": "Helsinki-NLP/opus-mt-en-tr",
                 "French": "Helsinki-NLP/opus-mt-fr-tr",
                 "German": None  # means we don't have a direct de->tr model
               }
        :param ner_model: The Hugging Face model name/path for Turkish NER
        :param pivot_lang: The language (e.g. "English") we will pivot to if direct model not found
        """
        self.translation_models = translation_models
        self.ner_model = ner_model
        self.pivot_lang = pivot_lang

        # We'll assume we have an English->Turkish model and pivot_lang->English model, etc.
        # For example:
        #   - "French->English" might be "Helsinki-NLP/opus-mt-fr-en"
        #   - "English->Turkish" might be "Helsinki-NLP/opus-mt-en-tr"
        # This info can be stored in translation_models dictionary or in a separate structure.

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

        # 1) Check if there's a direct model for source_lang -> Turkish
        direct_model_name = self.translation_models.get(source_lang)

        # If we don't have a direct model for this language, we do pivot
        if direct_model_name:
            # Direct translation pipeline
            translation_pipeline_1 = pipeline(
                task="translation",
                model=direct_model_name
            )
            # Translate directly into Turkish
            translation_result = translation_pipeline_1(text)
            translated_text = translation_result[0]["translation_text"]

        else:
            # 2) Pivot approach:
            #    source_lang -> pivot_lang -> Turkish
            # We need two pipelines: (a) source_lang->pivot_lang (b) pivot_lang->Turkish

            # 2a) Source_lang -> pivot_lang
            pivot_model_name = self.translation_models.get(f"{source_lang}->{self.pivot_lang}")
            if not pivot_model_name:
                logging.error(f"No direct or pivot model found for {source_lang} -> {self.pivot_lang}.")
                return text, []

            pivot_pipeline = pipeline(
                task="translation",
                model=pivot_model_name
            )
            pivot_result = pivot_pipeline(text)
            pivot_text = pivot_result[0]["translation_text"]

            # 2b) Pivot_lang -> Turkish
            pivot_to_tr_model_name = self.translation_models.get(f"{self.pivot_lang}")
            if not pivot_to_tr_model_name:
                logging.error(f"No direct or pivot model found for {self.pivot_lang} -> Turkish.")
                return pivot_text, []

            pivot_to_tr_pipeline = pipeline(
                task="translation",
                model=pivot_to_tr_model_name
            )
            final_result = pivot_to_tr_pipeline(pivot_text)
            translated_text = final_result[0]["translation_text"]

        # 3) Create an NER pipeline for Turkish
        ner_pipeline = pipeline(
            task="token-classification",
            model=self.ner_model,
            aggregation_strategy="simple"  # Combine tokens into full entity spans
        )

        # 4) Perform NER on the translated text
        ner_results = ner_pipeline(translated_text)

        # 5) Extract relevant entity info (avoid duplicates, if desired)
        entities = []
        seen_words = set()
        for entity_item in ner_results:
            word = entity_item["word"]
            if word not in seen_words:
                seen_words.add(word)
                entities.append({
                    "entity": word,
                    "type": entity_item["entity_group"],
                    "score": entity_item["score"]
                })

        return translated_text, entities
