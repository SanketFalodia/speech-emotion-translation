# 🎙️ Multilingual Speech Emotion Recognition & Translation

This project is an **AI-powered voice assistant** that can recognize speech, detect the language, analyze emotions, translate spoken words into multiple languages, and generate speech output.  

It integrates **speech recognition, emotion detection, and text-to-speech synthesis** to create an intelligent, interactive experience.

---

## ✨ Features
- 🎤 **Speech Recognition** – Convert speech to text using Google Speech Recognition.  
- 🌐 **Language Detection** – Automatically detect the spoken language.  
- 🔄 **Translation** – Translate speech into multiple languages using `deep-translator`.  
- 😊 **Emotion Detection** – Analyze emotions from speech text using Hugging Face Transformers.  
- 🔊 **Text-to-Speech (TTS)** – Generate natural-sounding speech with Google TTS.  
- 🖥️ **Cross-Platform Support** – Works on **Windows, macOS, and Linux**.

---

## 🛠️ Tech Stack
- **Python 3**
- [speechrecognition](https://pypi.org/project/SpeechRecognition/) – for speech-to-text  
- [langdetect](https://pypi.org/project/langdetect/) – for language detection  
- [gTTS](https://pypi.org/project/gTTS/) – for text-to-speech conversion  
- [transformers](https://huggingface.co/transformers/) – for emotion detection  
- [deep-translator](https://pypi.org/project/deep-translator/) – for translation  

---

## 🚀 How It Works
1. The user speaks into the microphone.  
2. Speech is converted into text.  
3. The language of the text is detected.  
4. The text is analyzed for emotions.  
5. The text is translated into a user-selected language.  
6. The translated text is spoken back to the user.  

---

## 📦 Installation
```bash
git clone https://github.com/your-username/speech-emotion-translation.git
cd speech-emotion-translation
pip install -r requirements.txt
