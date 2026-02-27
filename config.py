# ── Global Configuration ──
import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env into os.environ

# Whisper STT settings
WHISPER_MODEL = "base.en"          # Model: tiny.en, base.en, small.en, medium.en
WHISPER_COMPUTE_TYPE = "int8"      # Quantization: int8 (fastest), float16, float32
WHISPER_DEVICE = "cpu"             # Device: "cpu" or "cuda"

# Audio capture settings
SAMPLE_RATE = 16000                # Whisper expects 16kHz mono audio
CHANNELS = 1                      # Mono
BLOCK_DURATION_MS = 30             # Duration of each audio block in ms
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)  # Samples per block

# Silence detection (Voice Activity Detection)
SILENCE_THRESHOLD = 0.01          # RMS amplitude below this = silence (tune this)
SILENCE_DURATION = 1.5            # Seconds of silence before finalizing utterance
MAX_RECORDING_DURATION = 30       # Max seconds per utterance (safety cap)
MIN_UTTERANCE_DURATION = 0.3      # Ignore utterances shorter than this (noise filter)

# ── Groq API settings ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Set in .env file
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast, free-tier model
GROQ_TEMPERATURE = 0.0            # Deterministic output for reliable JSON
GROQ_MAX_TOKENS = 1024            # Max tokens in response

# ── MCP Server settings ──
MCP_SERVER_NAME = "desktop-automation"

# ── Vision / ML settings ──
RF_MODEL_TYPE = "rf"                # "rf" for Random Forest, "svm" for SVM
RF_CONFIDENCE = 0.4                 # Min confidence to report a detection
SCREENSHOT_DIR = "screenshots"      # Folder to save screenshots
