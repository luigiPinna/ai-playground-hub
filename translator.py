from transformers import pipeline


class Translator:
    def __init__(self):
        # Init pipeline for translations IT->EN and EN->IT
        self.it_to_en = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-it-en"
        )
        self.en_to_it = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-it"
        )

    def translate_it_to_en(self, text):
        result = self.it_to_en(text)
        return {
            'original': text,
            'translated': result[0]['translation_text'],
            'direction': 'IT -> EN'
        }

    def translate_en_to_it(self, text):
        result = self.en_to_it(text)
        return {
            'original': text,
            'translated': result[0]['translation_text'],
            'direction': 'EN -> IT'
        }

    def translate_batch_it_to_en(self, texts):
        """Translate a list of texts from It to En"""
        results = self.it_to_en(texts)
        return [
            {
                'original': text,
                'translated': result['translation_text'],
                'direction': 'IT -> EN'
            }
            for text, result in zip(texts, results)
        ]


if __name__ == "__main__":
    translator = Translator()

    print("=== Test Translation===")

    # Test IT -> EN
    italian_text = "Ciao, come stai? Spero tu stia bene."
    result = translator.translate_it_to_en(italian_text)
    print(f"Original: {result['original']}")
    print(f"Translated: {result['translated']}")
    print(f"Direction: {result['direction']}")
    print("-" * 50)

    # Test EN -> IT
    english_text = "Hello, how are you? I hope you are doing well."
    result = translator.translate_en_to_it(english_text)
    print(f"Original: {result['original']}")
    print(f"Translated: {result['translated']}")
    print(f"Direction: {result['direction']}")
    print("-" * 50)

    # Test IT -> EN long sentence
    italian_text = "Provo una frase più lunga. Quella casa è bella!"
    result = translator.translate_it_to_en(italian_text)
    print(f"Original: {result['original']}")
    print(f"Translated: {result['translated']}")
    print(f"Direction: {result['direction']}")
    print("-" * 50)