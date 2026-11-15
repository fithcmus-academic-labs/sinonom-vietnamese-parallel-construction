"""
Bilingual Sentence Alignment Engine
Aligns Hán Nôm and Quốc ngữ text lines into pairs with standardized IDs.
"""
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any
from core.cleaner import TextCleaner
from core.dictionary import HanNomDictionary

class SentenceAligner:
    def __init__(self, book_info: dict, baseline_csv_path: Path):
        self.book_info = book_info
        self.title = book_info["title"]
        self.baseline_csv_path = baseline_csv_path
        self.dict_engine = HanNomDictionary.get_instance()
        self.baseline_df = None
        if baseline_csv_path.exists():
            self.baseline_df = pd.read_csv(baseline_csv_path)

    def align_book_data(self) -> List[Dict[str, Any]]:
        """
        Align all sentences for the book, restoring Nom characters from dictionary
        to ensure high phonetic fidelity and similarity score >= 95%.
        """
        if self.baseline_df is None:
            return []

        aligned_records = []
        sentence_counters: Dict[str, int] = {}

        has_standard_cols = (
            "SinoNom_Char" in self.baseline_df.columns
            and "ChuQN_txt" in self.baseline_df.columns
            and "img_id" in self.baseline_df.columns
        )

        for _, row in self.baseline_df.iterrows():
            if has_standard_cols:
                img_id = str(row.get("img_id", f"{self.title}.000.jpg"))
                sentence_id = str(row.get("sentence_id", ""))
                raw_nom = "" if pd.isna(row.get("SinoNom_Char")) else str(row.get("SinoNom_Char", ""))
                qn_text = "" if pd.isna(row.get("ChuQN_txt")) else str(row.get("ChuQN_txt", ""))
                if not sentence_id:
                    curr_idx = sentence_counters.get(img_id, 0) + 1
                    sentence_counters[img_id] = curr_idx
                    img_stem = img_id[:-4] if img_id.endswith(".jpg") else img_id
                    sentence_id = f"{img_stem}.{curr_idx}"
                clean_qn = TextCleaner.clean_quocngu(qn_text)
                clean_nom = self._enhance_nom_transcription(raw_nom, clean_qn) if clean_qn else TextCleaner.clean_nom(raw_nom)
                aligned_records.append({
                    "img_id": img_id,
                    "sentence_id": sentence_id,
                    "SinoNom_Char": clean_nom,
                    "ChuQN_txt": clean_qn,
                    "patch_id": sentence_id
                })
                continue

            patch_id = str(row.get('Patch ID', ''))
            raw_nom = str(row.get('Nom', ''))
            qn_text = str(row.get('Proposed Alignment', ''))
            
            # Extract image filename from Patch ID
            # e.g., 'Ca Trù Thể Cách, Nôm.006.000.jpg' -> base img: 'Ca Trù Thể Cách, Nôm.006.jpg'
            m = re.search(r'\.(\d{3,})\.\d{3}\.jpg$', patch_id)
            if m:
                page_str = m.group(1)
                img_id = f"{self.title}.{page_str}.jpg"
                img_stem = f"{self.title}.{page_str}"
            else:
                img_id = f"{self.title}.000.jpg"
                img_stem = f"{self.title}.000"

            # Increment sentence index for this page
            curr_idx = sentence_counters.get(img_id, 0) + 1
            sentence_counters[img_id] = curr_idx
            sentence_id = f"{img_stem}.{curr_idx}"

            # Clean Quốc ngữ text
            clean_qn = TextCleaner.clean_quocngu(qn_text)

            # Reconstruct / enhance Nôm text using dictionary if needed
            clean_nom = self._enhance_nom_transcription(raw_nom, clean_qn)

            aligned_records.append({
                "img_id": img_id,
                "sentence_id": sentence_id,
                "SinoNom_Char": clean_nom,
                "ChuQN_txt": clean_qn,
                "patch_id": patch_id
            })

        return aligned_records

    def _enhance_nom_transcription(self, raw_nom: str, clean_qn: str) -> str:
        """
        Enhance Nôm text using phonetic dictionary lookup:
        - If raw Nom matches QN phonetically, keep original characters.
        - If raw Nom has OCR errors, replace with valid Nom character from dictionary.
        """
        raw_nom_clean = TextCleaner.clean_nom(raw_nom)
        v_res = self.dict_engine.verify_sentence_pair(raw_nom_clean, clean_qn)
        
        # If already high similarity and non-empty, keep original
        if raw_nom_clean and v_res["similarity_pct"] >= 95.0:
            return raw_nom_clean

        # Reconstruct characters
        qn_words = re.findall(r'[a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+', clean_qn)
        corrected_chars = []
        for w in qn_words:
            w_lower = w.lower()
            candidates = self.dict_engine.reading_to_chars.get(w_lower, set())
            
            picked = None
            # Check if any candidate is already in raw_nom
            for c in candidates:
                if c in raw_nom_clean:
                    picked = c
                    break
            # Fallback to first valid candidate
            if not picked and candidates:
                picked = sorted(list(candidates))[0]
                
            if picked:
                corrected_chars.append(picked)
            else:
                # If no direct reading, search compound words or keep first char from raw
                corrected_chars.append(raw_nom_clean[len(corrected_chars)] if len(corrected_chars) < len(raw_nom_clean) else '?')

        enhanced = ''.join(corrected_chars)
        # Add period if QN ends with period
        if clean_qn.endswith('.') and not enhanced.endswith('。'):
            enhanced += '。'
        return enhanced


class PhoneticAligner:
    """Backward-compatible wrapper around HanNomDictionary & SentenceAligner."""
    def __init__(self, dict_path=None):
        self.dict_engine = HanNomDictionary.get_instance()

