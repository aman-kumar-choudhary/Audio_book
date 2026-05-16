# OPENROUTER_API_KEY  = os.getenv('OPENROUTER_API_KEY', '')
# OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'google/gemini-3.1-flash-lite-preview')

# # Flask secret key
# SECRET_KEY = os.getenv('SECRET_KEY', 'change-me-in-production')

# ALL_MODELS = [
    
#     {
#         "id": "google/gemini-3.1-flash-lite-preview",
#         "name": "google/gemini-3.1-flash-lite-preview",
#         "provider": "openrouter",
#         "free": True,
#     }
# ]

# # Default model = first free OpenRouter model.
# # DEFAULT_LLM_MODEL = "gemini-2.5-flash"
# DEFAULT_LLM_MODEL = "google/gemini-3.1-flash-lite-preview"





import os
import torch
from dotenv import load_dotenv

load_dotenv()

# ─── Directory Layout ─────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
TEMP_FOLDER   = os.path.join(BASE_DIR, 'temp')

for _folder in (UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER):
    os.makedirs(_folder, exist_ok=True)

# ─── MongoDB ──────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MONGO_DB  = os.getenv('MONGO_DB',  'audiobook_pipeline')

# ─── API Keys ─────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'   # llm_cleaner.py appends /chat/completions

# Flask
SECRET_KEY = os.getenv('SECRET_KEY', 'change-me-in-production')

# ─── Available LLM Models (OpenRouter only) ───────────────────────────────────
ALL_MODELS = [
    {
        'id':       'google/gemini-2.0-flash-exp:free',
        'name':     'Gemini 2.0 Flash (Free)',
        'provider': 'openrouter',
        'free':     True,
    },
    {
        'id':       'meta-llama/llama-3.3-70b-instruct:free',
        'name':     'Llama 3.3 70B Instruct (Free)',
        'provider': 'openrouter',
        'free':     True,
    },
    {
        'id':       'mistralai/mistral-7b-instruct:free',
        'name':     'Mistral 7B Instruct (Free)',
        'provider': 'openrouter',
        'free':     True,
    },
]

DEFAULT_LLM_MODEL = ALL_MODELS[0]['id']

# ─── GPU Detection ────────────────────────────────────────────────────────────
def _detect_gpu() -> bool:
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  GPU detected : {name}")
            print(f"  CUDA version : {torch.version.cuda}")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['KOKORO_USE_GPU']        = '1'
            return True
        print("  No CUDA GPU — CPU mode")
        os.environ['KOKORO_USE_GPU'] = '0'
        return False
    except Exception as exc:
        print(f"  GPU check failed ({exc}) — CPU mode")
        os.environ['KOKORO_USE_GPU'] = '0'
        return False

USE_GPU = _detect_gpu()
print(f"  TTS acceleration : {'GPU (NVIDIA CUDA)' if USE_GPU else 'CPU'}")

# ─── Kokoro-FastAPI ───────────────────────────────────────────────────────────.
# GPU mode listens on 8880, CPU mode on 8880 (same port, different startup).
KOKORO_API_URL = os.getenv('KOKORO_API_URL', 'http://localhost:8880/web')
KOKORO_CPU_URL = os.getenv('KOKORO_CPU_URL', 'http://localhost:8880')

KOKORO_HEALTH_URL = f'{KOKORO_API_URL}/health'

print(f"  Kokoro URL : {KOKORO_API_URL}")

# ─── Calibre CLI ──────────────────────────────────────────────────────────────
CALIBRE_CMD = os.getenv('CALIBRE_CMD', 'ebook-convert')

# ─── Upload Limits ────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', '200'))
MAX_UPLOAD_BYTES   = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# ─── Kokoro Voice Mappings ────────────────────────────────────────────────────
KOKORO_DEFAULT_VOICES: dict[str, str] = {
    'default':        'af_bella',
    'female':         'af_heart',
    'male':           'am_adam',
    'british_female': 'bf_emma',
    'british_male':   'bm_george',
}

KOKORO_AVAILABLE_VOICES: list[dict] = [
    {'id': 'af_bella',  'name': 'Bella (American Female)',  'category': 'american_female'},
    {'id': 'af_heart',  'name': 'Heart (American Female)',  'category': 'american_female'},
    {'id': 'af_sarah',  'name': 'Sarah (American Female)',  'category': 'american_female'},
    {'id': 'af_sky',    'name': 'Sky (American Female)',    'category': 'american_female'},
    {'id': 'am_adam',   'name': 'Adam (American Male)',     'category': 'american_male'},
    {'id': 'am_echo',   'name': 'Echo (American Male)',     'category': 'american_male'},
    {'id': 'am_liam',   'name': 'Liam (American Male)',     'category': 'american_male'},
    {'id': 'bf_emma',   'name': 'Emma (British Female)',    'category': 'british_female'},
    {'id': 'bf_alice',  'name': 'Alice (British Female)',   'category': 'british_female'},
    {'id': 'bm_george', 'name': 'George (British Male)',    'category': 'british_male'},
    {'id': 'bm_daniel', 'name': 'Daniel (British Male)',    'category': 'british_male'},
]

# ─── Parallel TTS Workers ─────────────────────────────────────────────────────
TTS_MAX_WORKERS = int(os.getenv('TTS_MAX_WORKERS', '8' if USE_GPU else '4'))
print(f"  TTS workers : {TTS_MAX_WORKERS}")