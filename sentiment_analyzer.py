from transformers import pipeline


class SentimentAnalyzer:
    def __init__(self):
        # Let's use a simpler, more reliable model that works well with Italian text
        print("Loading sentiment analysis model... (this might take a moment)")
        self.classifier = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        print("Model loaded! Ready to analyze some feelings 😊")

    def analyze(self, text):
        """
        Analyzes the sentiment of the given text.
        Returns a nice dictionary with the results.
        """
        result = self.classifier(text)

        # Convert numeric labels to human-readable format
        sentiment_map = {
            'LABEL_1': 'Very Negative',
            'LABEL_2': 'Negative',
            'LABEL_3': 'Neutral',
            'LABEL_4': 'Positive',
            'LABEL_5': 'Very Positive'
        }

        raw_label = result[0]['label']
        human_label = sentiment_map.get(raw_label, raw_label)

        return {
            'text': text,
            'sentiment': human_label,
            'confidence': round(result[0]['score'], 3),
            'raw_label': raw_label  # Keep the original for debugging
        }

    def analyze_batch(self, texts):
        """
        Analyzes a whole bunch of texts at once.
        More efficient than calling analyze() multiple times.
        """
        print(f"Analyzing {len(texts)} texts in batch...")
        results = self.classifier(texts)

        sentiment_map = {
            'LABEL_1': 'Very Negative',
            'LABEL_2': 'Negative',
            'LABEL_3': 'Neutral',
            'LABEL_4': 'Positive',
            'LABEL_5': 'Very Positive'
        }

        return [
            {
                'text': text,
                'sentiment': sentiment_map.get(result['label'], result['label']),
                'confidence': round(result['score'], 3),
                'raw_label': result['label']
            }
            for text, result in zip(texts, results)
        ]


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    # Let's test with some Italian texts - from super positive to pretty negative
    test_texts = [
        "Questo prodotto è assolutamente fantastico, lo consiglio a tutti!",
        "Meh, non è male ma neanche eccezionale",
        "Che schifo, ho sprecato i miei soldi",
        "Ottimo rapporto qualità prezzo"
    ]

    print("\n=== Sentiment Analysis Test Results ===")
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\n📝 Text: '{result['text']}'")
        print(f"😊 Sentiment: {result['sentiment']}")
        print(f"🎯 Confidence: {result['confidence']}")
        print("-" * 60)