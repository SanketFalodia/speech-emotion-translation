"""
Emotion-Preserving Speech Translation Pipeline
Main orchestration for hackathon demo
"""

import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf

# Audio emotion detection
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch

# STT
from faster_whisper import WhisperModel

# Prosody extraction
import parselmouth
from parselmouth.praat import call

# Translation
from deep_translator import GoogleTranslator

# TTS
from TTS.api import TTS as CoquiTTS

# Noise reduction
import noisereduce as nr

# Import config
import sys
sys.path.append('.')
from config import *


class EmotionPreservingPipeline:
    """Main pipeline for emotion-preserving speech translation"""
    
    def __init__(self, use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        print("🚀 Initializing Emotion-Preserving Pipeline...")
        
        # Load models
        self._load_stt()
        self._load_emotion_detector()
        self._load_tts()
        
        print("✅ Pipeline ready!")
        
    def _load_stt(self):
        """Load speech-to-text model"""
        print("📝 Loading STT model...")
        self.stt_model = WhisperModel(
            STT_MODEL, 
            device=STT_DEVICE,
            compute_type=STT_COMPUTE_TYPE
        )
        
    def _load_emotion_detector(self):
        """Load audio-based emotion detection model"""
        print("😊 Loading emotion detection model...")
        self.emotion_processor = Wav2Vec2Processor.from_pretrained(EMOTION_MODEL)
        self.emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_MODEL)
        self.emotion_model.to(self.device)
        
    def _load_tts(self):
        """Load text-to-speech model"""
        print("🔊 Loading TTS model...")
        try:
            self.tts = CoquiTTS(TTS_MODEL, gpu=False)
        except Exception as e:
            print(f"⚠️ Coqui TTS failed: {e}")
            print("📱 Falling back to gTTS...")
            self.tts = None
            
    def preprocess_audio(self, audio_data, sample_rate):
        """
        Apply noise reduction and normalization
        Returns: cleaned audio, sample_rate
        """
        if NOISE_REDUCTION['enabled']:
            audio_data = nr.reduce_noise(
                y=audio_data, 
                sr=sample_rate,
                stationary=NOISE_REDUCTION['stationary'],
                prop_decrease=NOISE_REDUCTION['prop_decrease']
            )
        
        # Normalize
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data, sample_rate
    
    def detect_emotion_and_intensity(self, audio_data, sample_rate):
        """
        Detect emotion and intensity from audio
        Returns: (emotion_label, intensity_score)
        """
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import librosa
            audio_data = librosa.resample(
                y=audio_data, 
                orig_sr=sample_rate, 
                target_sr=16000
            )
            sample_rate = 16000
        
        # Process audio
        inputs = self.emotion_processor(
            audio_data, 
            sampling_rate=sample_rate,
            return_tensors="pt", 
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            logits = self.emotion_model(**inputs).logits
        
        # Get emotion and intensity
        scores = torch.nn.functional.softmax(logits, dim=-1)
        emotion_id = torch.argmax(scores).item()
        intensity = scores[0][emotion_id].item()
        
        emotion = EMOTION_LABELS[emotion_id]
        
        return emotion, intensity
    
    def extract_prosody(self, audio_data, sample_rate):
        """
        Extract prosody features: pitch, energy, speaking rate
        Returns: dict with prosody parameters
        """
        # Save to temp file for Praat
        temp_file = "temp_prosody.wav"
        sf.write(temp_file, audio_data, sample_rate)
        
        # Load with Parselmouth
        sound = parselmouth.Sound(temp_file)
        
        # Extract pitch
        pitch = call(sound, "To Pitch", 
                    0.0, 
                    PROSODY_CONFIG['pitch']['min'], 
                    PROSODY_CONFIG['pitch']['max'])
        
        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        pitch_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        
        # Extract intensity/energy
        intensity = call(sound, "To Intensity", 
                        PROSODY_CONFIG['intensity']['min'], 
                        0.0)
        mean_intensity = call(intensity, "Get mean", 0, 0, "energy")
        intensity_std = call(intensity, "Get standard deviation", 0, 0)
        
        # Get duration for speaking rate
        duration = call(sound, "Get total duration")
        
        # Clean up
        Path(temp_file).unlink(missing_ok=True)
        
        return {
            'mean_pitch': mean_pitch,
            'pitch_std': pitch_std,
            'mean_intensity': mean_intensity,
            'intensity_std': intensity_std,
            'duration': duration,
            'speaking_rate': 1.0 / duration if duration > 0 else 1.0
        }
    
    def speech_to_text(self, audio_data, sample_rate, language='en'):
        """
        Convert speech to text
        Returns: transcribed text
        """
        # Save to temp file
        temp_file = "temp_stt.wav"
        sf.write(temp_file, audio_data, sample_rate)
        
        # Transcribe
        segments, info = self.stt_model.transcribe(
            temp_file, 
            language=language,
            beam_size=5
        )
        
        # Combine segments
        text = " ".join([segment.text for segment in segments])
        
        # Clean up
        Path(temp_file).unlink(missing_ok=True)
        
        return text.strip()
    
    def translate_text(self, text, target_lang):
        """
        Translate text to target language
        Returns: translated text
        """
        try:
            translator = GoogleTranslator(source='auto', target=target_lang)
            translated = translator.translate(text)
            return translated
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return text  # Return original if translation fails
    
    def apply_emotion_to_prosody(self, base_prosody, emotion):
        """
        Adjust prosody parameters based on emotion
        Returns: modified prosody dict
        """
        if emotion not in EMOTION_PROSODY_MAP:
            emotion = 'neutral'
        
        emotion_params = EMOTION_PROSODY_MAP[emotion]
        
        modified_prosody = {
            'pitch_factor': emotion_params['pitch_factor'],
            'speed_factor': emotion_params['speed_factor'],
            'energy_factor': emotion_params['energy_factor'],
            'target_pitch': base_prosody['mean_pitch'] * emotion_params['pitch_factor'],
            'target_energy': base_prosody['mean_intensity'] * emotion_params['energy_factor']
        }
        
        return modified_prosody
    
    def synthesize_speech(self, text, target_lang, emotion, prosody_params):
        """
        Generate speech with emotion and prosody control
        Returns: audio_array, sample_rate
        """
        if self.tts is None:
            # Fallback to gTTS (no emotion control)
            from gtts import gTTS
            temp_file = "temp_tts.mp3"
            tts = gTTS(text=text, lang=target_lang)
            tts.save(temp_file)
            
            audio_data, sr = sf.read(temp_file)
            Path(temp_file).unlink(missing_ok=True)
            
            return audio_data, sr
        
        # Use Coqui TTS with emotion control
        try:
            # Get speed from prosody
            speed = prosody_params['speed_factor']
            
            # Generate
            wav = self.tts.tts(
                text=text,
                language=target_lang,
                speed=speed
            )
            
            return np.array(wav), OUTPUT_SAMPLE_RATE
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return np.array([]), OUTPUT_SAMPLE_RATE
    
    async def process_async(self, audio_file, source_lang, target_lang):
        """
        Process audio with parallel execution
        Returns: results dict with all outputs and metrics
        """
        start_time = time.time()
        results = {'latency': {}}
        
        # Load audio
        audio_data, sample_rate = sf.read(audio_file)
        results['latency']['load'] = time.time() - start_time
        
        # Preprocess
        step_time = time.time()
        audio_data, sample_rate = self.preprocess_audio(audio_data, sample_rate)
        results['latency']['preprocess'] = time.time() - step_time
        
        # PARALLEL: STT + Emotion + Prosody
        step_time = time.time()
        
        loop = asyncio.get_event_loop()
        
        stt_task = loop.run_in_executor(
            self.executor, 
            self.speech_to_text, 
            audio_data, sample_rate, source_lang
        )
        
        emotion_task = loop.run_in_executor(
            self.executor,
            self.detect_emotion_and_intensity,
            audio_data, sample_rate
        )
        
        prosody_task = loop.run_in_executor(
            self.executor,
            self.extract_prosody,
            audio_data, sample_rate
        )
        
        # Wait for all parallel tasks
        text, (emotion, intensity), prosody = await asyncio.gather(
            stt_task, emotion_task, prosody_task
        )
        
        results['latency']['parallel_processing'] = time.time() - step_time
        
        # Store results
        results['original_text'] = text
        results['emotion'] = emotion
        results['intensity'] = intensity
        results['prosody_original'] = prosody
        
        # Translate
        step_time = time.time()
        translated_text = self.translate_text(text, target_lang)
        results['translated_text'] = translated_text
        results['latency']['translation'] = time.time() - step_time
        
        # Apply emotion to prosody
        prosody_modified = self.apply_emotion_to_prosody(prosody, emotion)
        results['prosody_modified'] = prosody_modified
        
        # Synthesize with emotion
        step_time = time.time()
        output_audio, output_sr = self.synthesize_speech(
            translated_text, 
            target_lang, 
            emotion, 
            prosody_modified
        )
        results['latency']['tts'] = time.time() - step_time
        
        # Save output
        output_file = f"{AUDIO_OUTPUT_DIR}/output_{int(time.time())}.wav"
        Path(AUDIO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        sf.write(output_file, output_audio, output_sr)
        results['output_file'] = output_file
        
        # Total latency
        results['latency']['total'] = time.time() - start_time
        
        return results
    
    def process(self, audio_file, source_lang, target_lang):
        """
        Synchronous wrapper for async processing
        """
        return asyncio.run(self.process_async(audio_file, source_lang, target_lang))


# ============================================================================
# DEMO USAGE
# ============================================================================
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = EmotionPreservingPipeline()
    
    # Process a test file
    results = pipeline.process(
        audio_file="test_audio.wav",
        source_lang="en",
        target_lang="hi"
    )
    
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    print(f"Original Text: {results['original_text']}")
    print(f"Translated Text: {results['translated_text']}")
    print(f"Emotion: {results['emotion']} (Intensity: {results['intensity']:.2f})")
    print(f"\n⏱️ Latency Breakdown:")
    for step, latency in results['latency'].items():
        print(f"  {step}: {latency:.3f}s")
    print(f"\n✅ Output saved to: {results['output_file']}")
    print(f"🎯 Total Latency: {results['latency']['total']:.2f}s")