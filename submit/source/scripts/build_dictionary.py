import os
import sys
import re
import json
import pickle
import unicodedata

sys.stdout.reconfigure(encoding='utf-8')

DICT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'dicts')
UNIHAN_FILE = os.path.join(DICT_DIR, 'Unihan_Readings.txt')
NOMVIET_FILE = os.path.join(DICT_DIR, 'nom_viet.dict.yaml')
OUT_JSON = os.path.join(DICT_DIR, 'hannom_dict.json')
OUT_PKL = os.path.join(DICT_DIR, 'hannom_dict.pkl')

def normalize_text(text: str) -> str:
    """Normalize text using NFC."""
    return unicodedata.normalize('NFC', text.strip())

def extract_vietnamese_words(text: str) -> list[str]:
    """Extract all valid Vietnamese words from a string."""
    clean = re.sub(r'\(.*?\)', ' ', text)
    clean = re.sub(r'\[.*?\]', ' ', clean)
    clean = re.sub(r'\d+\.', ' ', clean)
    words = re.findall(r'[a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+', clean)
    return [normalize_text(w.lower()) for w in words if len(w) > 0]

def build():
    print("Building Hán Nôm Dictionary Database...")
    char_to_readings = {} # char -> set of readings
    reading_to_chars = {} # reading -> set of chars
    compound_words = {}   # multi-word phrase -> nom phrase

    def add_entry(char: str, reading: str):
        char = normalize_text(char)
        reading = normalize_text(reading.lower())
        if not char or not reading:
            return
        if len(char) == 1:
            if char not in char_to_readings:
                char_to_readings[char] = set()
            char_to_readings[char].add(reading)
            
            if reading not in reading_to_chars:
                reading_to_chars[reading] = set()
            reading_to_chars[reading].add(char)
        else:
            compound_words[reading] = char

    # 1. Parse Unihan_Readings.txt
    if os.path.exists(UNIHAN_FILE):
        print(f"Parsing Unihan from {UNIHAN_FILE}...")
        unihan_count = 0
        with open(UNIHAN_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if 'kVietnamese' in line and not line.startswith('#'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        try:
                            code_hex = parts[0].replace('U+', '')
                            char = chr(int(code_hex, 16))
                            readings = parts[2].split()
                            for r in readings:
                                for w in extract_vietnamese_words(r):
                                    add_entry(char, w)
                                    unihan_count += 1
                        except Exception:
                            continue
        print(f"  Added {unihan_count} readings from Unihan.")
    else:
        print("Unihan_Readings.txt not found, skipping.")

    # 2. Parse nom_viet.dict.yaml
    if os.path.exists(NOMVIET_FILE):
        print(f"Parsing Nom Na Viet from {NOMVIET_FILE}...")
        passed_dots = False
        nom_count = 0
        with open(NOMVIET_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '...':
                    passed_dots = True
                    continue
                if not passed_dots or not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    nom_text = normalize_text(parts[0])
                    # From comments
                    if len(parts) >= 4:
                        comment = parts[3]
                        comment_words = extract_vietnamese_words(comment)
                        for w in comment_words:
                            add_entry(nom_text, w)
                            nom_count += 1
                    # Also if text is 1 char and code is simple single word
                    code = parts[1].strip()
                    # extract code word
                    code_words = extract_vietnamese_words(code)
                    for w in code_words:
                        add_entry(nom_text, w)
                        nom_count += 1
        print(f"  Added {nom_count} readings from nom_viet.dict.yaml.")
    else:
        print("nom_viet.dict.yaml not found, skipping.")

    # 3. Add common Hán Nôm synonyms and project specific baseline corrections
    baseline_pairs = [
        ('歌', 'ca'), ('籌', 'trù'), ('體', 'thể'), ('格', 'cách'),
        ('各', 'các'), ('調', 'điệu'), ('唱', 'hát'), ('術', 'về'),
        ('妸', 'ả'), ('姩', 'đào'), ('教', 'giáo'), ('敎', 'giáo'),
        ('香', 'hương'), ('皇', 'hoàng'), ('圖', 'đồ'), ('鞏', 'củng'),
        ('固', 'cố'), ('帝', 'đế'), ('道', 'đạo'), ('遐', 'hà'),
        ('昌', 'xương'), ('管', 'quản'), ('甲', 'giáp'), ('僻', 'lỏi'),
        ('登', 'giảng'), ('上', 'cách'), ('朱', 'cho'), ('沛', 'phải'),
        ('法', 'phép'), ('意', 'ấy'), ('樂', 'nhạc'), ('坐', 'ngồi'),
        ('難', 'nghe'), ('沒', 'một'), ('嗵', 'tiếng'), ('持', 'chãy'),
        ('埃', 'ai'), ('寬', 'khoan'), ('日', 'nhặt'), ('脂', 'đêm'),
        ('冬', 'đông'), ('長', 'trường'), ('聽', 'gió'), ('律', 'lọt'),
        ('味', 'mùi'), ('祝', 'chúc'), ('聖', 'thánh'), ('君', 'quân'),
        ('開', 'mở'), ('塘', 'đường'), ('治', 'trị'), ('雁', 'nhạn'),
        ('飛', 'phi'), ('紫', 'tử'), ('塞', 'tái'), ('舍', 'xá'),
        ('吏', 'lại'), ('回', 'hồi'), ('歸', 'quy')
    ]
    for char, reading in baseline_pairs:
        add_entry(char, reading)

    # Convert sets to sorted lists for JSON serialization
    serialized_char_to_readings = {k: sorted(list(v)) for k, v in char_to_readings.items()}
    serialized_reading_to_chars = {k: sorted(list(v)) for k, v in reading_to_chars.items()}

    print(f"\nFinal Dictionary Stats:")
    print(f"  Total unique Hán Nôm characters: {len(serialized_char_to_readings)}")
    print(f"  Total unique Vietnamese readings: {len(serialized_reading_to_chars)}")
    print(f"  Compound words / phrases: {len(compound_words)}")

    # Save to JSON
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({
            'char_to_readings': serialized_char_to_readings,
            'reading_to_chars': serialized_reading_to_chars,
            'compound_words': compound_words
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON to {OUT_JSON} ({os.path.getsize(OUT_JSON)} bytes)")

    # Save to PKL
    with open(OUT_PKL, 'wb') as f:
        pickle.dump({
            'char_to_readings': char_to_readings,
            'reading_to_chars': reading_to_chars,
            'compound_words': compound_words
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved Pickle to {OUT_PKL} ({os.path.getsize(OUT_PKL)} bytes)")

if __name__ == '__main__':
    build()
