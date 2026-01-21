import speech_recognition as sr
from langdetect import detect
from gtts import gTTS
import os
import platform
from transformers import pipeline
from deep_translator import GoogleTranslator

# Initialize emotion classifier
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Recognize speech using microphone
def recognize_speech(language="en-US"):
    recognizer = sr.Recognizer()
    # Note: SpeechRecognition library handles the audio backend automatically
    # It will use sounddevice if pyaudio is not available
    with sr.Microphone() as source:
        print(f"🎙️ Speak now ({language})...")
        print("⏳ Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language=language)
            print(f"✅ Recognized Text: {text}")
            return text
        except sr.UnknownValueError:
            print("❌ Could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"❌ Could not request results; {e}")
            return None

# Detect language from text
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Translate using deep_translator
def translate_text(text, dest_lang):
    try:
        translated = GoogleTranslator(source='auto', target=dest_lang).translate(text)
        return translated
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        return None

# Convert text to speech
def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
        print(f"🔊 Playing audio...")
        if platform.system() == "Windows":
            os.system("start output.mp3")
        elif platform.system() == "Darwin":
            os.system("afplay output.mp3")
        else:
            os.system("mpg123 output.mp3")
    except Exception as e:
        print(f"❌ Text-to-speech failed: {e}")

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
    print("=" * 50)
    print("🎙️  Multilingual Speech Emotion Recognition")
    print("=" * 50)
    
    print("\n📝 Supported languages:")
    print("   en - English")
    print("   hi - Hindi")
    print("   bn - Bengali")
    print("   es - Spanish")
    print("   fr - French")
    print("   de - German")
    print()
    
    input_lang = input("Enter the language you'll speak (e.g., 'en', 'hi', 'bn', 'es'): ").strip()
    
    # Map language codes to Google Speech Recognition format
    lang_map = {
        'en': 'en-US',
        'hi': 'hi-IN',
        'bn': 'bn-IN',
        'es': 'es-ES',
        'fr': 'fr-FR',
        'de': 'de-DE'
    }
    
    recognition_lang = lang_map.get(input_lang, 'en-US')
    text = recognize_speech(language=recognition_lang)

    if text:
        src_lang = detect_language(text)
        print(f"🌐 Detected Language: {src_lang}")

        emotion = detect_emotion(text)
        print(f"😊 Emotion: {emotion}")

        dest_lang = input("\nEnter target language code for translation (e.g., 'en', 'fr', 'hi', 'bn'): ").strip()
        translated = translate_text(text, dest_lang)
        if translated:
            print(f"📝 Translated Text: {translated}")
            text_to_speech(translated, lang=dest_lang)
        else:
            print("❌ Translation failed. Please try again.")
    else:
        print("❌ No speech detected. Please try again.")

# Entry point
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")