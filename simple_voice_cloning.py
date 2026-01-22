"""
Improved Voice Cloning with Emotional Expression
More conservative voice matching with better emotion modulation
"""

import os
import numpy as np
import librosa
import soundfile as sf
from gtts import gTTS


class EmotionalVoiceCloner:
    """
    Voice cloning with emotional expression - improved algorithm
    """
    
    # More subtle emotion parameters
    EMOTION_PARAMS = {
        'joy': {
            'pitch_adjust': 3.0,       # Noticeably higher
            'speed_adjust': 1.20,      # Faster
            'energy_adjust': 1.4,      # Louder
            'vibrato': 0.05            # Add slight vibrato
        },
        'sadness': {
            'pitch_adjust': -3.0,      # Lower
            'speed_adjust': 0.80,      # Slower
            'energy_adjust': 0.65,     # Quieter
            'vibrato': 0.0
        },
        'anger': {
            'pitch_adjust': 2.0,       # Slightly higher
            'speed_adjust': 1.30,      # Much faster
            'energy_adjust': 1.6,      # Much louder
            'vibrato': 0.02
        },
        'fear': {
            'pitch_adjust': 4.0,       # Higher, shaky
            'speed_adjust': 1.35,      # Fast, rushed
            'energy_adjust': 1.3,      # Louder
            'vibrato': 0.08            # More vibrato (shaky)
        },
        'surprise': {
            'pitch_adjust': 5.0,       # Much higher
            'speed_adjust': 1.25,      # Faster
            'energy_adjust': 1.5,      # Louder
            'vibrato': 0.03
        },
        'disgust': {
            'pitch_adjust': -2.0,      # Slightly lower
            'speed_adjust': 0.90,      # Slower
            'energy_adjust': 0.9,      # Slightly quieter
            'vibrato': 0.0
        },
        'neutral': {
            'pitch_adjust': 0.0,
            'speed_adjust': 1.0,
            'energy_adjust': 1.0,
            'vibrato': 0.0
        }
    }
    
    def __init__(self):
        self.temp_files = []
    
    def analyze_voice_characteristics(self, reference_audio):
        """
        Analyze voice with better pitch detection
        """
        print(f"🔍 Analyzing voice from: {reference_audio}")
        
        try:
            # Load audio
            y, sr = librosa.load(reference_audio, sr=None)
            
            # Use YIN algorithm for better pitch detection
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
            
            # Filter out zeros and get median pitch (more robust than mean)
            valid_f0 = f0[f0 > 0]
            if len(valid_f0) > 0:
                avg_pitch = float(np.median(valid_f0))
            else:
                # Fallback: use old method
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(float(pitch))
                avg_pitch = float(np.median(pitch_values)) if pitch_values else 180.0
            
            # Clamp pitch to reasonable range (80 Hz - 400 Hz covers most human voices)
            avg_pitch = float(np.clip(avg_pitch, 80, 400))
            
            # Determine if voice is male or female
            if avg_pitch < 165:
                voice_type = "male"
                gTTS_base = 140.0  # gTTS sounds more male around here
            else:
                voice_type = "female"
                gTTS_base = 220.0  # gTTS sounds more female around here
            
            characteristics = {
                'pitch_hz': avg_pitch,
                'voice_type': voice_type,
                'gTTS_base': gTTS_base
            }
            
            print(f"   • Detected voice type: {voice_type.upper()}")
            print(f"   • Average pitch: {avg_pitch:.1f} Hz")
            
            return characteristics
            
        except Exception as e:
            print(f"   ⚠️ Error analyzing voice: {e}")
            return {
                'pitch_hz': 180.0,
                'voice_type': 'neutral',
                'gTTS_base': 180.0
            }
    
    def add_vibrato(self, y, sr, amount=0.05):
        """Add vibrato effect for emotional expression"""
        if amount == 0:
            return y
        
        # Create vibrato (5-6 Hz modulation)
        vibrato_freq = 5.5
        t = np.arange(len(y)) / sr
        vibrato = 1 + amount * np.sin(2 * np.pi * vibrato_freq * t)
        
        return y * vibrato
    
    def synthesize_with_emotion_and_voice(self, text, lang, voice_chars, emotion='neutral', output_file="emotional_cloned.wav"):
        """
        Generate speech with CONSERVATIVE voice matching and STRONG emotion
        """
        emotion = emotion.lower()
        emotion_params = self.EMOTION_PARAMS.get(emotion, self.EMOTION_PARAMS['neutral'])
        
        print(f"\n🎨 Synthesizing speech:")
        print(f"   🎭 Emotion: {emotion.upper()}")
        print(f"   🎤 Voice type: {voice_chars['voice_type']}")
        
        try:
            # Generate base speech
            temp_tts = "temp_base_speech.mp3"
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(temp_tts)
            self.temp_files.append(temp_tts)
            
            # Load generated speech
            y, sr = librosa.load(temp_tts, sr=None)
            
            # STEP 1: CONSERVATIVE voice matching (limit to ±4 semitones max)
            target_pitch = float(voice_chars['pitch_hz'])
            gTTS_base = float(voice_chars['gTTS_base'])
            
            if target_pitch > 0 and gTTS_base > 0:
                voice_pitch_shift = float(12.0 * np.log2(target_pitch / gTTS_base))
                # LIMIT voice matching to ±4 semitones to avoid extreme changes
                voice_pitch_shift = float(np.clip(voice_pitch_shift, -4.0, 4.0))
            else:
                voice_pitch_shift = 0.0
            
            # STEP 2: Add FULL emotional pitch (this is what makes it expressive!)
            emotion_pitch = float(emotion_params['pitch_adjust'])
            total_pitch_shift = float(voice_pitch_shift + emotion_pitch)
            
            # Overall safety limit: ±8 semitones total
            total_pitch_shift = float(np.clip(total_pitch_shift, -8.0, 8.0))
            
            print(f"   • Voice matching: {voice_pitch_shift:+.1f} semitones (limited)")
            print(f"   • Emotion effect: {emotion_pitch:+.1f} semitones")
            print(f"   • Total pitch: {total_pitch_shift:+.1f} semitones")
            
            # Apply pitch shift
            if abs(total_pitch_shift) > 0.3:
                y = librosa.effects.pitch_shift(y, sr=sr, n_steps=total_pitch_shift)
            
            # STEP 3: Emotion speed (no voice matching for tempo)
            emotion_speed = float(emotion_params['speed_adjust'])
            total_speed = float(np.clip(emotion_speed, 0.7, 1.4))
            
            print(f"   • Speed adjustment: {total_speed:.2f}x")
            
            if abs(total_speed - 1.0) > 0.05:
                y = librosa.effects.time_stretch(y, rate=total_speed)
            
            # STEP 4: Emotion volume (simple, no voice matching)
            emotion_energy = float(emotion_params['energy_adjust'])
            total_volume = float(np.clip(emotion_energy, 0.5, 2.0))
            
            print(f"   • Volume adjustment: {total_volume:.2f}x")
            
            y = y * total_volume
            
            # Prevent clipping
            max_val = float(np.abs(y).max())
            if max_val > 1.0:
                y = y / max_val * 0.95
            
            # STEP 5: Add vibrato for certain emotions
            vibrato_amount = float(emotion_params['vibrato'])
            if vibrato_amount > 0:
                print(f"   • Adding vibrato: {vibrato_amount:.2f}")
                y = self.add_vibrato(y, sr, vibrato_amount)
            
            # Save output
            sf.write(output_file, y, sr)
            print(f"\n✅ Emotional voice-cloned audio saved!")
            print(f"📁 File: {output_file}")
            
            return output_file
            
        except Exception as e:
            print(f"   ❌ Error during synthesis: {e}")
            raise
    
    def clone_voice(self, text, reference_audio, lang='en', emotion='neutral', output_file="cloned_voice.wav"):
        """
        Main method to clone voice with emotion
        """
        print("\n" + "="*70)
        print("🎤 Emotional Voice Cloning (Improved)")
        print("="*70)
        
        # Analyze reference voice
        voice_chars = self.analyze_voice_characteristics(reference_audio)
        
        # Synthesize with characteristics + emotion
        result = self.synthesize_with_emotion_and_voice(text, lang, voice_chars, emotion, output_file)
        
        return result
    
    def cleanup(self):
        """Remove temporary files"""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        self.temp_files = []


def clone_and_speak(text, reference_audio, language='en', emotion='neutral', output_file="cloned_output.wav"):
    """
    Convenience function for emotional voice cloning
    
    Args:
        text: Text to speak
        reference_audio: Path to reference audio file
        language: Language code
        emotion: Emotion to apply (joy, sadness, anger, fear, surprise, disgust, neutral)
        output_file: Output file path
    
    Returns:
        Path to generated audio file
    """
    cloner = EmotionalVoiceCloner()
    try:
        output = cloner.clone_voice(text, reference_audio, language, emotion, output_file)
        return output
    finally:
        cloner.cleanup()


# Test the module
if __name__ == "__main__":
    print("=" * 70)
    print("🎤 Emotional Voice Cloning Test (Improved)")
    print("="*70)
    
    reference = input("\nEnter path to reference audio (WAV file): ").strip()
    
    if os.path.exists(reference):
        test_text = input("Enter text to speak: ").strip() or "Hello! This is a test."
        
        print("\n🎭 Available emotions:")
        print("   joy, sadness, anger, fear, surprise, disgust, neutral")
        emotion = input("Choose emotion (default=neutral): ").strip() or "neutral"
        
        output = clone_and_speak(test_text, reference, "en", emotion)
        
        # Play audio
        import platform
        print("\n🔊 Playing cloned audio...")
        if platform.system() == "Windows":
            os.system(f"start {output}")
        elif platform.system() == "Darwin":
            os.system(f"afplay {output}")
        else:
            os.system(f"mpg123 {output}")
    else:
        print(f"❌ Reference audio not found: {reference}")