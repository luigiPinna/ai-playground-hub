# AI Playground Hub

A hands-on playground for experimenting with Hugging Face models: sentiment analysis, translation, and image generation.

## Features

- **Sentiment Analysis**: Multilingual sentiment detection (Italian/English optimized)
- **Translation**: Bidirectional IT ↔ EN translation using Helsinki-NLP models
- **Image Generation**: Text-to-image generation powered by Stable Diffusion v1.5

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Models Used

- **Sentiment**: `cardiffnlp/twitter-xlm-roberta-base-sentiment` - Robust multilingual model
- **Translation**: `Helsinki-NLP/opus-mt-it-en` & `Helsinki-NLP/opus-mt-en-it` - High-quality IT/EN pairs
- **Images**: `runwayml/stable-diffusion-v1-5` - Reliable and fast image generation

## Hardware Optimization

- **MacBook M1/M2**: Leverages Metal Performance Shaders for accelerated inference
- **Other systems**: Falls back to CPU with decent performance

## Project Structure

```
├── sentiment_analyzer.py    # Sentiment analysis functionality
├── translator.py           # Text translation module  
├── image_generator.py      # Stable Diffusion image generation
├── main.py                 # Interactive menu for testing
└── generated_images/       # Output directory for generated images
```

## Notes

- All models are open source and free to use
- First run will download models (~2GB total)
- Generated images are saved locally and excluded from git
- Optimized for both development experimentation and portfolio demonstration