import speech_recognition as sr
from langdetect import detect
from gtts import gTTS
import os
import platform
from transformers import pipeline
from deep_translator import GoogleTranslator  # ✅ Replaced googletrans

# Initialize emotion classifier
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Recognize speech using microphone
def recognize_speech(language="en-US"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"🎙️ Speak now ({language})...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language=language)
            print(f"✅ Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            print("❌ Could not understand the audio.")
            return None

# Detect language from text
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# ✅ Translate using deep_translator instead of googletrans
def translate_text(text, dest_lang):
    try:
        translated = GoogleTranslator(source='auto', target=dest_lang).translate(text)
        return translated
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return None

# Convert text to speech
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    if platform.system() == "Windows":
        os.system("start output.mp3")
    elif platform.system() == "Darwin":
        os.system("afplay output.mp3")
    else:
        os.system("mpg123 output.mp3")

# Detect emotion from text
def detect_emotion(text):
    try:
        result = emotion_classifier(text)
        label = result[0][0]['label']
        score = result[0][0]['score']
        return f"{label.capitalize()} ({score * 100:.1f}%)"
    except Exception as e:
        return f"Error detecting emotion: {e}"

# Main workflow
def main():
    input_lang = input("Enter the language you'll speak ('bn','hi','en','es'): ").strip()
    text = recognize_speech(language=input_lang)

    if text:
        src_lang = detect_language(text)
        print(f"🌐 Detected Language: {src_lang}")

        emotion = detect_emotion(text)
        print(f"😊 Emotion: {emotion}")

        dest_lang = input("Enter target language code ('en', 'fr', 'hi', 'bn'): ").strip()
        translated = translate_text(text, dest_lang)
        if translated:
            print(f"📝 Translated Text: {translated}")
            text_to_speech(translated, lang=dest_lang)

# Entry point
if __name__ == "__main__":
    main()
