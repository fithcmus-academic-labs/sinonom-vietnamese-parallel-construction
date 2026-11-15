"""
System Configuration for Hán Nôm - Quốc ngữ Parallel Corpus Pipeline
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DICTS_DIR = DATA_DIR / "dicts"
INTERIM_DIR = DATA_DIR / "interim"

def _resolve_dir(candidates, default):
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
    for c in candidates:
        if c.is_dir():
            return c
    return default

OUTPUT_DIR = _resolve_dir(
    [BASE_DIR / "output", Path.cwd() / "output", BASE_DIR.parent / "output"],
    BASE_DIR / "output",
)
DATASETS_MIDTERM_DIR = _resolve_dir(
    [
        BASE_DIR / "datasets-midterm",
        Path.cwd() / "datasets-midterm",
        BASE_DIR.parent / "datasets-midterm",
    ],
    BASE_DIR / "datasets-midterm",
)
PDF_DIR = _resolve_dir(
    [
        BASE_DIR / "datasets-midterm" / "pdf",
        BASE_DIR / "pdf",
        BASE_DIR / "input",
        BASE_DIR.parent / "datasets-midterm" / "pdf",
        BASE_DIR.parent / "pdf",
        BASE_DIR.parent / "input",
        Path.cwd() / "datasets-midterm" / "pdf",
        Path.cwd() / "pdf",
        Path.cwd() / "input",
    ],
    DATASETS_MIDTERM_DIR / "pdf",
)

# Files
HANNOM_DICT_PKL = DICTS_DIR / "hannom_dict.pkl"
HANNOM_DICT_JSON = DICTS_DIR / "hannom_dict.json"

# Thresholds & Rules
MIN_SIMILARITY_THRESHOLD = 0.95  # Yêu cầu >= 95%
RENDER_DPI = 300
LAPLACIAN_BLUR_THRESHOLD = 50.0

# Excel Colors (ARGB)
COLOR_BLACK = "FF000000"
COLOR_RED = "FFFF0000"

# KimHanNom FIT HCMUS Cloud API Configuration
KIMHANNOM_BASE_URL = os.getenv("KIMHANNOM_BASE_URL", "https://kimhannom.fit.hcmus.edu.vn")
KIMHANNOM_EMAIL = os.getenv("KIMHANNOM_EMAIL", "")
KIMHANNOM_PASSWORD = os.getenv("KIMHANNOM_PASSWORD", "")
KIMHANNOM_UPLOAD_ENDPOINT = "/api/web/clc-sinonom/image-upload"
KIMHANNOM_OCR_ENDPOINT = "/api/web/clc-sinonom/image-ocr"
KIMHANNOM_TRANSLIT_ENDPOINT = "/api/web/clc-sinonom/sinonom-transliteration"


# Target Books in Midterm Datasets
BOOKS = {
    "NOM_Y": {
        "title": "Ca Trù Thể Cách, Nôm_Y",
        "pdf_filename": "Ca Trù Thể Cách, Nôm_Y.pdf",
        "csv_filename": "Ca Trù Thể Cách, Nôm_Y.csv",
        "xlsx_filename": "Ca Trù Thể Cách, Nôm_Y.xlsx",
        "layout_type": "T2b",
    },
    "AB_160": {
        "title": "Ca Trù Thê Cách, AB 160, Nôm_Y",
        "pdf_filename": "Ca Trù Thê Cách, AB 160, Nôm_Y.pdf",
        "csv_filename": "Ca Trù Thê Cách, AB 160, Nôm_Y.csv",
        "xlsx_filename": "Ca Trù Thê Cách, AB 160, Nôm_Y.xlsx",
        "layout_type": "T2b",
    },
    "AB_431": {
        "title": "Ca Trù Thế Cách, AB 431, Nôm_Y",
        "pdf_filename": "Ca Trù Thế Cách, AB 431, Nôm_Y.pdf",
        "csv_filename": "Ca Trù Thế Cách, AB 431, Nôm_Y.csv",
        "xlsx_filename": "Ca Trù Thế Cách, AB 431, Nôm_Y.xlsx",
        "layout_type": "T2b",
    },
    "AB_456": {
        "title": "Ca Trù Thế Cách, AB 456, Nôm_Y",
        "pdf_filename": "Ca Trù Thế Cách, AB 456, Nôm_Y.pdf",
        "csv_filename": "Ca Trù Thế Cách, AB 456, Nôm_Y.csv",
        "xlsx_filename": "Ca Trù Thế Cách, AB 456, Nôm_Y.xlsx",
        "layout_type": "T2b",
    }
}
