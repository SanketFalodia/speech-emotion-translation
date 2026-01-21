"""
Configuration file for Emotion-Preserving Speech Translation System
"""

# ============================================================================
# AUDIO SETTINGS
# ============================================================================
SAMPLE_RATE = 16000  # 16kHz for Whisper
CHANNELS = 1  # Mono
CHUNK_DURATION = 5  # seconds for recording
BUFFER_SIZE = 1024

# ============================================================================
# MODEL PATHS & SETTINGS
# ============================================================================

# Speech-to-Text
STT_MODEL = "base"  # Options: tiny, base, small, medium, large-v3
STT_DEVICE = "cpu"  # cpu or cuda
STT_COMPUTE_TYPE = "int8"  # int8 for CPU optimization

# Emotion Detection
EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Text-to-Speech
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"  # Coqui TTS
# Alternative: "en_US-lessac-medium" for piper-tts (faster but less emotional)

# ============================================================================
# LANGUAGE SETTINGS
# ============================================================================
SUPPORTED_LANGUAGES = {
    'en': {'name': 'English', 'whisper': 'en', 'tts': 'en'},
    'hi': {'name': 'Hindi', 'whisper': 'hi', 'tts': 'hi'},
    'bn': {'name': 'Bengali', 'whisper': 'bn', 'tts': 'bn'},
    'es': {'name': 'Spanish', 'whisper': 'es', 'tts': 'es'},
    'fr': {'name': 'French', 'whisper': 'fr', 'tts': 'fr'},
    'de': {'name': 'German', 'whisper': 'de', 'tts': 'de'},
}

# ============================================================================
# PROSODY SETTINGS
# ============================================================================
PROSODY_CONFIG = {
    'pitch': {
        'min': 75,  # Hz
        'max': 600,  # Hz
        'time_step': 0.01  # seconds
    },
    'intensity': {
        'min': 50,  # dB
        'time_step': 0.01
    }
}

# Emotion to Prosody Mapping (adjustments as multipliers)
EMOTION_PROSODY_MAP = {
    'happy': {
        'pitch_factor': 1.15,  # 15% higher
        'speed_factor': 1.10,  # 10% faster
        'energy_factor': 1.20,  # 20% more energy
    },
    'sad': {
        'pitch_factor': 0.85,  # 15% lower
        'speed_factor': 0.80,  # 20% slower
        'energy_factor': 0.70,  # 30% less energy
    },
    'angry': {
        'pitch_factor': 1.20,  # 20% higher
        'speed_factor': 1.25,  # 25% faster
        'energy_factor': 1.30,  # 30% more energy
    },
    'fear': {
        'pitch_factor': 1.10,
        'speed_factor': 1.15,
        'energy_factor': 0.90,
    },
    'disgust': {
        'pitch_factor': 0.95,
        'speed_factor': 0.95,
        'energy_factor': 0.85,
    },
    'surprise': {
        'pitch_factor': 1.25,
        'speed_factor': 1.20,
        'energy_factor': 1.15,
    },
    'neutral': {
        'pitch_factor': 1.0,
        'speed_factor': 1.0,
        'energy_factor': 1.0,
    }
}

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
MAX_WORKERS = 4  # For parallel processing
LATENCY_TARGET = 2.5  # seconds
ENABLE_ASYNC = True  # Use async processing

# ============================================================================
# UI SETTINGS
# ============================================================================
UI_CONFIG = {
    'theme': 'default',
    'title': '🎙️ Emotion-Preserving Speech Translation',
    'description': '''
    Translate speech while preserving emotional intent and speaking style.
    Real-time emotion detection and prosody-aware synthesis.
    ''',
    'enable_queue': True,
    'show_progress': True
}

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================
OUTPUT_DIR = 'outputs'
AUDIO_OUTPUT_DIR = f'{OUTPUT_DIR}/audio'
LOG_DIR = f'{OUTPUT_DIR}/logs'
COMPARISON_DIR = f'{OUTPUT_DIR}/comparisons'

# Audio format
OUTPUT_FORMAT = 'wav'
OUTPUT_SAMPLE_RATE = 22050  # Standard for TTS

# ============================================================================
# NOISE REDUCTION SETTINGS
# ============================================================================
NOISE_REDUCTION = {
    'enabled': True,
    'stationary': True,  # For consistent background noise
    'prop_decrease': 1.0  # Reduction strength (0.0-1.0)
}

# ============================================================================
# DEBUGGING & LOGGING
# ============================================================================
DEBUG = True
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
SAVE_INTERMEDIATE_OUTPUTS = True  # Save STT, prosody data, etc.

# Performance monitoring
TRACK_LATENCY = True
LATENCY_LOG_FILE = f'{LOG_DIR}/latency.csv'

# ============================================================================
# HACKATHON DEMO SETTINGS
# ============================================================================
DEMO_CONFIG = {
    'show_technical_panel': True,
    'show_comparison': True,
    'show_metrics': True,
    'enable_before_after': True,
    'show_latency_timer': True
}

# ============================================================================
# CONSTRAINTS (Hackathon Requirements)
# ============================================================================
CONSTRAINTS = {
    'max_latency': 2.5,  # seconds
    'min_continuous_speech': 30,  # seconds
    'max_continuous_speech': 60,  # seconds
    'min_emotions': 7,  # emotion categories
    'min_prosody_controls': 2,  # pitch + speed minimum
    'cpu_only': True  # Must work on CPU
}