"""
Enhanced Speech Translation with Voice Cloning
Records user's voice, translates, and outputs in their own voice
"""

import sounddevice as sd
from scipy.io.wavfile import write
import requests
from langdetect import detect
from gtts import gTTS
import os
import platform
from transformers import pipeline
from deep_translator import GoogleTranslator
import json
from simple_voice_cloning import clone_and_speak

# Initialize emotion classifier
print("Loading emotion classifier...")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Record audio using sounddevice
def record_audio(duration=5, sample_rate=22050):  # 22050 Hz is better for voice cloning
    print(f"🎙️ Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording complete!")
    return audio, sample_rate

# Save audio to file
def save_audio(audio, sample_rate, filename="temp_audio.wav"):
    write(filename, sample_rate, audio)
    return filename

# Recognize speech using Google Speech API
def recognize_speech_from_file(audio_file, language="en-US"):
    try:
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        
        url = "http://www.google.com/speech-api/v2/recognize"
        params = {
            'output': 'json',
            'lang': language,
            'key': 'AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw'
        }
        headers = {'Content-Type': 'audio/l16; rate=22050;'}
        
        response = requests.post(url, params=params, headers=headers, data=audio_data)
        
        for line in response.content.decode('utf-8').strip().split('\n'):
            if line:
                try:
                    result = json.loads(line)
                    if 'result' in result and result['result']:
                        transcript = result['result'][0]['alternative'][0]['transcript']
                        print(f"✅ Recognized Text: {transcript}")
                        return transcript
                except:
                    continue
        
        print("❌ Could not understand the audio.")
        return None
        
    except Exception as e:
        print(f"❌ Speech recognition failed: {e}")
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

# Detect emotion from text
def detect_emotion(text):
    try:
        result = emotion_classifier(text)
        label = result[0][0]['label']
        score = result[0][0]['score']
        return label, score
    except Exception as e:
        return "neutral", 0.0

# Play audio file
def play_audio(audio_file):
    """Play audio file using platform-specific command"""
    print(f"🔊 Playing audio...")
    try:
        if platform.system() == "Windows":
            os.system(f"start {audio_file}")
        elif platform.system() == "Darwin":
            os.system(f"afplay {audio_file}")
        else:
            os.system(f"mpg123 {audio_file}")
    except Exception as e:
        print(f"❌ Could not play audio: {e}")

# Map language codes for TTS
def map_language_code(lang_code):
    """Map 2-letter codes to full language names for XTTS"""
    lang_map = {
        'en': 'en',
        'es': 'es',
        'fr': 'fr',
        'de': 'de',
        'it': 'it',
        'pt': 'pt',
        'pl': 'pl',
        'tr': 'tr',
        'ru': 'ru',
        'nl': 'nl',
        'cs': 'cs',
        'ar': 'ar',
        'zh': 'zh-cn',
        'ja': 'ja',
        'hi': 'hi',  # Hindi support depends on model
        'ko': 'ko'
    }
    return lang_map.get(lang_code, 'en')

# Main workflow with voice cloning
def main():
    print("=" * 70)
    print("🎤 Multilingual Speech Translation with Voice Cloning")
    print("   Your voice, any language!")
    print("=" * 70)
    
    print("\n📝 Supported languages:")
    print("   en - English    | hi - Hindi     | bn - Bengali")
    print("   es - Spanish    | fr - French    | de - German")
    print("   it - Italian    | pt - Portuguese| ar - Arabic")
    print("   zh - Chinese    | ja - Japanese  | ko - Korean")
    print()
    
    input_lang = input("Enter the language you'll speak (e.g., 'en', 'hi', 'es'): ").strip()
    
    # Map language codes to Google Speech Recognition format
    lang_map = {
        'en': 'en-US', 'hi': 'hi-IN', 'bn': 'bn-IN',
        'es': 'es-ES', 'fr': 'fr-FR', 'de': 'de-DE',
        'it': 'it-IT', 'pt': 'pt-PT', 'ar': 'ar-SA',
        'zh': 'zh-CN', 'ja': 'ja-JP', 'ko': 'ko-KR'
    }
    
    recognition_lang = lang_map.get(input_lang, 'en-US')
    
    # Record audio
    duration = int(input("Enter recording duration in seconds (default 5): ") or "5")
    audio, sample_rate = record_audio(duration=duration, sample_rate=22050)
    
    # Save audio to file (this becomes our voice reference!)
    audio_file = save_audio(audio, sample_rate, filename="user_voice_reference.wav")
    
    # Recognize speech
    text = recognize_speech_from_file(audio_file, language=recognition_lang)

    if text:
        src_lang = detect_language(text)
        print(f"🌐 Detected Language: {src_lang}")

        # Detect emotion
        emotion_label, emotion_score = detect_emotion(text)
        print(f"😊 Emotion: {emotion_label.capitalize()} ({emotion_score * 100:.1f}%)")

        # Translate text
        dest_lang = input("\nEnter target language code for translation (e.g., 'en', 'fr', 'hi', 'es'): ").strip()
        translated = translate_text(text, dest_lang)
        
        if translated:
            print(f"📝 Translated Text: {translated}")
            
            # Ask about voice cloning
            print("\n" + "=" * 70)
            print("🎭 Output Options:")
            print("=" * 70)
            print("1. Voice Cloning + Emotion (YOUR voice with emotional expression)")
            print("2. Voice Cloning Only (YOUR voice, neutral)")
            print("3. Standard TTS (robotic voice)")
            
            choice = input("\nChoose option (1/2/3, default=1): ").strip() or "1"
            
            if choice in ['1', '2']:
                print("\n🎤 Using emotional voice cloning...")
                
                try:
                    target_lang_code = dest_lang
                    use_emotion = (choice == '1')
                    
                    # Clone voice and generate speech
                    output_file = clone_and_speak(
                        text=translated,
                        reference_audio=audio_file,
                        language=target_lang_code,
                        emotion=emotion_label if use_emotion else 'neutral',
                        output_file="cloned_voice_output.wav"
                    )
                    
                    play_audio(output_file)
                    print(f"\n✅ Voice cloned audio saved as: {output_file}")
                    
                except Exception as e:
                    print(f"\n❌ Voice cloning failed: {e}")
                    print("   Falling back to standard TTS...")
                    tts = gTTS(text=translated, lang=dest_lang)
                    tts.save("output.mp3")
                    play_audio("output.mp3")
            
            else:
                # Use standard TTS
                print("\n🔊 Generating standard speech...")
                tts = gTTS(text=translated, lang=dest_lang)
                tts.save("output.mp3")
                play_audio("output.mp3")
            
            # Clean up reference audio if not needed
            cleanup = input("\n🗑️  Delete recorded voice reference? (y/n, default=n): ").strip().lower()
            if cleanup == 'y' and os.path.exists(audio_file):
                os.remove(audio_file)
                print("✅ Reference audio deleted.")
        else:
            print("❌ Translation failed. Please try again.")
    else:
        print("❌ No speech detected. Please try again.")
        if os.path.exists(audio_file):
            os.remove(audio_file)

# Entry point
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()