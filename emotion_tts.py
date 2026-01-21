"""
Emotional Text-to-Speech Module
Modifies audio properties (pitch, speed, volume) based on detected emotions
"""

from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.effects import speedup, normalize
import numpy as np
from scipy.io import wavfile
import pyrubberband as pyrb


class EmotionalTTS:
    """
    Text-to-Speech with emotional modulation
    """
    
    # Emotion-based audio parameters
    EMOTION_PARAMS = {
        'joy': {
            'pitch_shift': 2.0,      # Higher pitch
            'speed_factor': 1.15,     # Slightly faster
            'volume_boost': 3.0       # Louder
        },
        'sadness': {
            'pitch_shift': -2.5,      # Lower pitch
            'speed_factor': 0.85,     # Slower
            'volume_boost': -2.0      # Softer
        },
        'anger': {
            'pitch_shift': 1.5,       # Slightly higher pitch
            'speed_factor': 1.25,     # Faster
            'volume_boost': 5.0       # Much louder
        },
        'fear': {
            'pitch_shift': 3.0,       # High pitch
            'speed_factor': 1.3,      # Fast, rushed
            'volume_boost': 1.0       # Slightly louder
        },
        'surprise': {
            'pitch_shift': 3.5,       # Very high pitch
            'speed_factor': 1.2,      # Faster
            'volume_boost': 4.0       # Louder
        },
        'disgust': {
            'pitch_shift': -1.0,      # Slightly lower
            'speed_factor': 0.95,     # Slightly slower
            'volume_boost': 0.0       # Normal volume
        },
        'neutral': {
            'pitch_shift': 0.0,       # No change
            'speed_factor': 1.0,      # Normal speed
            'volume_boost': 0.0       # Normal volume
        }
    }
    
    def __init__(self):
        self.temp_files = []
    
    def generate_emotional_speech(self, text, lang='en', emotion='neutral'):
        """
        Generate speech with emotional characteristics
        
        Args:
            text: Text to convert to speech
            lang: Language code
            emotion: Detected emotion (lowercase)
        
        Returns:
            Path to the generated audio file
        """
        emotion = emotion.lower()
        
        # Get emotion parameters (default to neutral if not found)
        params = self.EMOTION_PARAMS.get(emotion, self.EMOTION_PARAMS['neutral'])
        
        print(f"🎭 Applying {emotion.upper()} emotion profile:")
        print(f"   • Pitch shift: {params['pitch_shift']:+.1f} semitones")
        print(f"   • Speed: {params['speed_factor']:.2f}x")
        print(f"   • Volume: {params['volume_boost']:+.1f} dB")
        
        # Generate base audio with gTTS
        temp_base = "temp_base_tts.mp3"
        tts = gTTS(text=text, lang=lang)
        tts.save(temp_base)
        self.temp_files.append(temp_base)
        
        # Convert to WAV for processing
        audio = AudioSegment.from_mp3(temp_base)
        temp_wav = "temp_processing.wav"
        audio.export(temp_wav, format="wav")
        self.temp_files.append(temp_wav)
        
        # Apply emotional modifications
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
        """
        Apply pitch, speed, and volume modifications to audio
        """
        # Load audio
        audio = AudioSegment.from_wav(audio_file)
        
        # Apply volume adjustment
        if volume_boost != 0:
            audio = audio + volume_boost
        
        # Apply speed change
        if speed_factor != 1.0:
            # Export to temporary file for speed adjustment
            temp_speed = "temp_speed.wav"
            audio.export(temp_speed, format="wav")
            self.temp_files.append(temp_speed)
            
            # Read with scipy
            sample_rate, samples = wavfile.read(temp_speed)
            
            # Change speed using time stretching
            if speed_factor > 1.0:
                # Speed up
                stretched = pyrb.time_stretch(samples.astype(np.float32), sample_rate, speed_factor)
            else:
                # Slow down
                stretched = pyrb.time_stretch(samples.astype(np.float32), sample_rate, speed_factor)
            
            # Write back
            temp_stretched = "temp_stretched.wav"
            wavfile.write(temp_stretched, sample_rate, stretched.astype(np.int16))
            self.temp_files.append(temp_stretched)
            
            audio = AudioSegment.from_wav(temp_stretched)
        
        # Apply pitch shift
        if pitch_shift != 0:
            temp_pitch = "temp_pitch_input.wav"
            audio.export(temp_pitch, format="wav")
            self.temp_files.append(temp_pitch)
            
            # Read audio
            sample_rate, samples = wavfile.read(temp_pitch)
            
            # Apply pitch shift
            shifted = pyrb.pitch_shift(samples.astype(np.float32), sample_rate, pitch_shift)
            
            # Write pitched audio
            temp_pitched = "temp_pitched.wav"
            wavfile.write(temp_pitched, sample_rate, shifted.astype(np.int16))
            self.temp_files.append(temp_pitched)
            
            audio = AudioSegment.from_wav(temp_pitched)
        
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
    """
    Convenience function for generating emotional speech
    
    Args:
        text: Text to convert to speech
        lang: Language code
        emotion: Detected emotion
    
    Returns:
        Path to the generated audio file
    """
    tts = EmotionalTTS()
    try:
        output_file = tts.generate_emotional_speech(text, lang, emotion)
        return output_file
    finally:
        tts.cleanup()


# Test the module
if __name__ == "__main__":
    print("Testing Emotional TTS...")
    
    test_text = "Hello! This is a test of emotional speech synthesis."
    
    for emotion in ['joy', 'sadness', 'anger', 'neutral']:
        print(f"\n{'='*50}")
        print(f"Testing {emotion.upper()} emotion")
        print('='*50)
        
        output = text_to_emotional_speech(test_text, 'en', emotion)
        print(f"✅ Generated: {output}")
        
        # Play the audio (platform-specific)
        import platform
        if platform.system() == "Windows":
            os.system(f"start {output}")
        elif platform.system() == "Darwin":
            os.system(f"afplay {output}")
        else:
            os.system(f"mpg123 {output}")
        
        input("Press Enter to continue to next emotion...")