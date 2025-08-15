from sentiment_analyzer import SentimentAnalyzer
from translator import Translator
from image_generator import ImageGenerator


def test_sentiment():
    print("🔍 Testing Sentiment Analysis...")
    analyzer = SentimentAnalyzer()

    texts = [
        "Adoro questo nuovo framework di AI!",
        "Il codice è buggatissimo, non funziona niente",
        "È un progetto interessante ma ha dei limiti"
    ]

    for text in texts:
        result = analyzer.analyze(text)
        print(f"'{result['text']}' -> {result['sentiment']} ({result['confidence']})")
    print()


def test_translation():
    print("🌍 Testing Translation...")
    translator = Translator()

    # IT -> EN
    italian_texts = [
        "Hugging Face è una piattaforma fantastica per l'AI",
        "Sto imparando il machine learning"
    ]

    for text in italian_texts:
        result = translator.translate_it_to_en(text)
        print(f"IT: {result['original']}")
        print(f"EN: {result['translated']}")
        print()


def test_image_generation():
    print("🎨 Testing Image Generation...")
    generator = ImageGenerator()

    prompts = [
        "A professional developer working on AI, digital art style",
        "Abstract representation of machine learning, colorful"
    ]

    for prompt in prompts:
        result = generator.generate(prompt, num_images=1)
        print(f"Generated: {result['prompt']}")
        print(f"Saved to: {result['saved_paths'][0]}")
        print()


def main():
    print("=== AI Playground Hub ===")
    print("Scegli quale funzione testare:")
    print("1. Sentiment Analysis")
    print("2. Translation")
    print("3. Image Generation")
    print("4. Test tutto")
    print("0. Esci")

    choice = input("\nInserisci la tua scelta (0-4): ")
    # menu
    if choice == "1":
        test_sentiment()
    elif choice == "2":
        test_translation()
    elif choice == "3":
        test_image_generation()
    elif choice == "4":
        test_sentiment()
        test_translation()
        test_image_generation()
    elif choice == "0":
        print("Ciao!")
        return
    else:
        print("Scelta non valida")
        main()


if __name__ == "__main__":
    main()