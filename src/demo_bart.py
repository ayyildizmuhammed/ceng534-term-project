# src/main.py

from config import CONFIG
from bart.bart_handler import BartPromptModel

def main():
    bart_model_config = CONFIG["BART_MODEL"]
    model_name = bart_model_config["name_or_path"] 

    # Initialize
    bart = BartPromptModel(model_name=model_name)

    # ################################################################
    # # 1) Zero-Shot Sentiment
    # ################################################################
    # text = "I really like this product, it's amazing!"
    # zero_shot_sent = bart.run_zero_shot_sentiment(text, labels=["positive", "negative", "neutral"])
    # print("=== Zero-Shot Sentiment ===")
    # print("Text:", text)
    # print("Model Output:", zero_shot_sent)

    ################################################################
    # 2) Zero-Shot Translation (English->Turkish)
    ################################################################
    en_text = "Hello, how are you doing today?"
    zero_shot_trans = bart.run_zero_shot_translation(en_text)
    print("\n=== Zero-Shot Translation ===")
    print("English:", en_text)
    print("Turkish:", zero_shot_trans)

    # ################################################################
    # # 3) Few-Shot (in-context) Prompt - Translation
    # ################################################################
    # examples_translation = [
    #     {
    #         "instruction": "English->Turkish Translation",
    #         "input_text": "Hello, how are you?",
    #         "output_text": "Merhaba, nasılsın?"
    #     },
    #     {
    #         "instruction": "English->Turkish Translation",
    #         "input_text": "This is awesome!",
    #         "output_text": "Bu harika!"
    #     }
    # ]
    # new_input = "I love cats."
    # few_shot_trans = bart.run_prompt_fewshot(examples_translation, new_input)
    # print("\n=== Few-Shot Prompt-based Translation ===")
    # print("Input:", new_input)
    # print("Output:", few_shot_trans)

    # ################################################################
    # # 4) Few-Shot (in-context) Prompt - Sentiment
    # ################################################################
    # examples_sentiment = [
    #     {
    #         "instruction": "Sentiment Classification (Positive or Negative)",
    #         "input_text": "This movie was fantastic!",
    #         "output_text": "Positive"
    #     },
    #     {
    #         "instruction": "Sentiment Classification (Positive or Negative)",
    #         "input_text": "I hate this food.",
    #         "output_text": "Negative"
    #     }
    # ]
    # new_input_sent = "The weather is okay, but not great."
    # few_shot_sent = bart.run_prompt_fewshot(examples_sentiment, new_input_sent)
    # print("\n=== Few-Shot Prompt-based Sentiment ===")
    # print("Input:", new_input_sent)
    # print("Output:", few_shot_sent)

if __name__ == "__main__":
    main()
