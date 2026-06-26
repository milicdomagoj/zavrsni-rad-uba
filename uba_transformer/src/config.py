import os

CERT_DIR = r"C:\Users\Domagoj\Desktop\r4.2"
ANSWERS_DIR = r"C:\Users\Domagoj\Desktop\answers"

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_DIR, "data_cache")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
for _d in (CACHE_DIR, RESULTS_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

DAILY_CACHE = os.path.join(CACHE_DIR, "daily_features.parquet")

DATE_FMT = "%m/%d/%Y %H:%M:%S"
DAY_FMT = "%m/%d/%Y"
CHUNK_SIZE = 2_000_000

WORK_START, WORK_END = 8, 18
INTERNAL_DOMAIN = "dtaa.com"
SUSPICIOUS_SITES = ["wikileaks", "job-hunt", "monster", "careerbuilder",
                    "indeed", "simplyhired", "keylog", "hackmageddon", "pastebin"]

SEQ_LENGTHS = [30, 50, 100, 200]
STRIDE = 5
SEQ_LENGTH_DEFAULT = 50

SEED = 42
TEST_FRAC = 0.20
VAL_FRAC = 0.20
MAX_EPOCHS = 200
BATCH_SIZE = 128
USE_MIXED_PRECISION = True

OPTUNA_TRIALS = 30
OPTUNA_EPOCHS = 25
