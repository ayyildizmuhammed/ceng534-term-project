# src/bart/bart_handler.py

import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class BartPromptModel:
    """
    A single model (facebook/bart-large) to handle:
    1) Zero-shot sentiment analysis (via minimal prompt).
    2) Zero-shot translation (via minimal prompt).
    3) Prompt-based few-shot in-context learning for either task.
    """

    def __init__(self, model_name="facebook/bart-large"):
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as e:
            logging.error(f"Error loading BART model ({model_name}): {str(e)}")
            self.tokenizer = None
            self.model = None

    def _generate(self, prompt_text: str, max_length=128, **generate_kwargs):
        """
        Internal helper to run the model.generate() on a given prompt text.
        """
        if not self.model or not self.tokenizer:
            logging.error("BartPromptModel not initialized.")
            return None

        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        output_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            **generate_kwargs
        )
        print("Output IDs:", output_ids)
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

    ############################################################################
    # Zero-Shot Sentiment
    ############################################################################
    def run_zero_shot_sentiment(self, text: str, labels=None, max_length=64):
        """
        Minimal prompt approach for sentiment classification.
        - We list possible labels (e.g. Positive/Negative) 
        - Ask the model to pick one.
        *Bart-large doesn't have a classification head, so we do everything by prompt.*
        """
        if labels is None:
            labels = ["positive", "negative", "neutral"]

        # Basit bir prompt: "Classify the sentiment as Positive or Negative"
        # "Text: xxxxxx, Sentiment:"
        prompt = "Classify the sentiment of the following text. Possible labels: "
        prompt += ", ".join(labels) + "\n"
        prompt += f"Text: {text}\nSentiment:"
        
        raw_output = self._generate(prompt, max_length=max_length)
        print("Raw Output:", raw_output)
        
        # Naif ayrıştırma: Çıktıda "Sentiment: X" gibi bir şey olabilir
        # Biz "raw_output"un direk son kısmını döndürelim
        # Veya "Output:" anahtar kelimesi vs. yok, mecburen ham text alacağız.
        # Basitçe "raw_output" return edelim.
        return raw_output.strip()

    ############################################################################
    # Zero-Shot Translation (English->Turkish)
    ############################################################################
    def run_zero_shot_translation(self, english_text: str, max_length=128):
        """
        Minimal prompt approach for translation.
        """
        prompt = f"Translate the following sentence from English to Turkish:\n" \
                 f"English: {english_text}\nTurkish:"

        raw_output = self._generate(prompt, max_length=max_length)
        print("Raw Output:", raw_output)
        return raw_output.strip()

    ############################################################################
    # Prompt-based Few-Shot (In-Context)
    ############################################################################
    def run_prompt_fewshot(self, examples, new_input: str, max_length=128, **generate_kwargs):
        """
        A more general prompt-based approach with few-shot examples.
        examples = [
          {
            "instruction": "English->Turkish Translation",
            "input_text": "Hello, how are you?",
            "output_text": "Merhaba, nasılsın?"
          },
          ...
        ]
        new_input = "I love cats."
        We'll build a meta-prompt and generate the next output.
        """
        if not examples:
            logging.warning("No examples provided for few-shot prompting.")
            return self._generate(new_input, max_length=max_length, **generate_kwargs)

        # Build a big prompt
        # Example format:
        # English->Turkish Translation
        # Input: Hello, how are you?
        # Output: Merhaba, nasılsın?
        #
        # English->Turkish Translation
        # Input: This is a test
        # Output: Bu bir test
        #
        # English->Turkish Translation
        # Input: I love cats.
        # Output:

        prompt_text = ""
        for ex in examples:
            instruction = ex.get("instruction", "")
            inp = ex.get("input_text", "")
            out = ex.get("output_text", "")
            prompt_text += f"{instruction}\nInput: {inp}\nOutput: {out}\n\n"

        # Add new input
        last_instruction = examples[-1].get("instruction", "Example")
        prompt_text += f"{last_instruction}\nInput: {new_input}\nOutput:"

        raw_output = self._generate(prompt_text, max_length=max_length, **generate_kwargs)
        
        # Belki "Output:" ifadesine göre kırpıyoruz
        if "Output:" in raw_output:
            raw_output = raw_output.split("Output:")[-1].strip()
        return raw_output.strip()
