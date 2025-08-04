from transformers import pipeline
import gradio as gr

# 🌍 Language code to name mapping
lang_map = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian",
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "kn": "Kannada",
    "ar": "Arabic", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ru": "Russian",
    "pt": "Portuguese", "bn": "Bengali", "ur": "Urdu", "tr": "Turkish", "fa": "Persian",
    "sv": "Swedish", "pl": "Polish", "nl": "Dutch", "id": "Indonesian", "he": "Hebrew",
    "uk": "Ukrainian", "ro": "Romanian", "cs": "Czech", "da": "Danish", "fi": "Finnish"
}

# 🔍 Load language detection model
lang_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")

# 🧠 Main detection function
def detect_language(text):
    result = lang_detector(text)[0]
    code = result["label"].replace("__label__", "")
    confidence = round(result["score"] * 100, 2)
    language = lang_map.get(code, code.upper())  # fallback to code if unknown
    return f"Detected Language: {language} ({confidence}%)"

# 🌐 Gradio Interface
gr.Interface(
    fn=detect_language,
    inputs=gr.Textbox(lines=2, placeholder="Type in any language...", label="Your Text"),
    outputs=gr.Textbox(label="Detected Language"),
    title="🌍 Language Detector App",
    description="Enter any sentence, and this app will identify its language using a multilingual transformer model.",
).launch(share=True)
