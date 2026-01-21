"""
Windows-compatible Emotional Text-to-Speech Module
Uses librosa instead of pyrubberband for better Windows compatibility
"""

from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.effects import normalize
import librosa
import soundfile as sf


class EmotionalTTS:
    """Text-to-Speech with emotional modulation (Windows-compatible)"""
    
    EMOTION_PARAMS = {
        'joy': {
            'pitch_shift': 2.0,
            'speed_factor': 1.15,
            'volume_boost': 3.0
        },
        'sadness': {
            'pitch_shift': -2.5,
            'speed_factor': 0.85,
            'volume_boost': -2.0
        },
        'anger': {
            'pitch_shift': 1.5,
            'speed_factor': 1.25,
            'volume_boost': 5.0
        },
        'fear': {
            'pitch_shift': 3.0,
            'speed_factor': 1.3,
            'volume_boost': 1.0
        },
        'surprise': {
            'pitch_shift': 3.5,
            'speed_factor': 1.2,
            'volume_boost': 4.0
        },
        'disgust': {
            'pitch_shift': -1.0,
            'speed_factor': 0.95,
            'volume_boost': 0.0
        },
        'neutral': {
            'pitch_shift': 0.0,
            'speed_factor': 1.0,
            'volume_boost': 0.0
        }
    }
    
    def __init__(self):
        self.temp_files = []
    
    def generate_emotional_speech(self, text, lang='en', emotion='neutral'):
        """Generate speech with emotional characteristics"""
        emotion = emotion.lower()
        params = self.EMOTION_PARAMS.get(emotion, self.EMOTION_PARAMS['neutral'])
        
        print(f"🎭 Applying {emotion.upper()} emotion profile:")
        print(f"   • Pitch shift: {params['pitch_shift']:+.1f} semitones")
        print(f"   • Speed: {params['speed_factor']:.2f}x")
        print(f"   • Volume: {params['volume_boost']:+.1f} dB")
        
        # Generate base audio
        temp_base = "temp_base_tts.mp3"
        tts = gTTS(text=text, lang=lang)
        tts.save(temp_base)
        self.temp_files.append(temp_base)
        
        # Convert to WAV
        audio = AudioSegment.from_mp3(temp_base)
        temp_wav = "temp_processing.wav"
        audio.export(temp_wav, format="wav")
        self.temp_files.append(temp_wav)
        
        # Apply effects using librosa
        modified_audio = self._apply_emotion_effects(
            temp_wav,
            pitch_shift=params['pitch_shift'],
            speed_factor=params['speed_factor'],
            volume_boost=params['volume_boost']
        )
        
        # Export final audio
        output_file = "emotional_output.mp3"
        modified_audio.export(output_file, format="mp3")
        
        return output_file
    
    def _apply_emotion_effects(self, audio_file, pitch_shift, speed_factor, volume_boost):
        """Apply pitch, speed, and volume modifications using librosa"""
        
        # Load audio with librosa
        y, sr = librosa.load(audio_file, sr=None)
        
        # Apply pitch shift
        if pitch_shift != 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
        
        # Apply time stretch (speed change)
        if speed_factor != 1.0:
            y = librosa.effects.time_stretch(y, rate=speed_factor)
        
        # Save to temporary file
        temp_modified = "temp_modified.wav"
        sf.write(temp_modified, y, sr)
        self.temp_files.append(temp_modified)
        
        # Load with pydub for volume adjustment
        audio = AudioSegment.from_wav(temp_modified)
        
        # Apply volume boost
        if volume_boost != 0:
            audio = audio + volume_boost
        
        # Normalize to prevent clipping
        audio = normalize(audio)
        
        return audio
    
    def cleanup(self):
        """Remove temporary files"""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        self.temp_files = []


def text_to_emotional_speech(text, lang='en', emotion='neutral'):
    """Convenience function for generating emotional speech"""
    tts = EmotionalTTS()
    try:
        output_file = tts.generate_emotional_speech(text, lang, emotion)
        return output_file
    finally:
        tts.cleanup()