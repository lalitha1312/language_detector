# language_detector

A simple NLP application that detects the language of any input text using a multilingual transformer model from Hugging Face.

 Features:

- 🔎 Detects over 50+ languages from a single sentence
- 🤖 Powered by `xlm-roberta` model from Hugging Face
- 🌐 Gradio-based web interface for instant interaction
- 📊 Displays full language names with confidence score
- 🔗 Can be deployed on Hugging Face Spaces

 Model Used:

- **Model:** [`papluca/xlm-roberta-base-language-detection`](https://huggingface.co/papluca/xlm-roberta-base-language-detection)
- **Pipeline:** `text-classification` from `transformers`

 Requirements:

Install the following dependencies:
```bash
pip install transformers torch gradio
