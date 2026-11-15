"""
Text Normalization and PPOCRLabel File Generation Module
Follows strict formatting rules defined in Quy định đồ án.
"""
import os
import re
import json
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

class TextCleaner:
    @staticmethod
    def clean_nom(text: str) -> str:
        """
        Clean and format Hán Nôm transcription:
        - No spaces between Nôm characters.
        - Standardize Nom punctuation: comma -> 、 , period -> 。
        - Remove invalid non-Han/Nom junk symbols.
        """
        if not text:
            return ""
        # Remove annotations like [1], (1)
        s = re.sub(r'\[\d+\]', '', text)
        s = re.sub(r'\(\d+\)', '', s)
        s = re.sub(r'[()（）\[\]]', '', s)
        
        # Replace latin punctuation with standard Sino-Nom punctuation
        s = s.replace(',', '、').replace('.', '。').replace(';', '；').replace(':', '：')
        s = s.replace('?', '？').replace('!', '！')
        
        # Remove whitespace between characters
        s = re.sub(r'\s+', '', s)
        return s.strip()

    clean_sinonom = clean_nom

    @staticmethod
    def clean_quocngu(text: str) -> str:
        """
        Clean and format Quốc ngữ transcription:
        - Unicode NFC normalization.
        - Proper spacing after punctuation.
        - Remove extraneous annotation marks.
        """
        if not text:
            return ""
        s = unicodedata.normalize('NFC', text)
        # Remove annotations
        s = re.sub(r'\[\d+\]', '', s)
        s = re.sub(r'\(\d+\)', '', s)
        
        # Normalize whitespace
        s = re.sub(r'\s+', ' ', s)
        
        # Proper spacing after punctuation: . , ; : ? !
        s = re.sub(r'\s*([.,;:?!])\s*', r'\1 ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        
        # Capitalize first letter if it starts with word
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        return s

class PPOCRLabelFormatter:
    @staticmethod
    def format_label_line(folder_name: str, img_name: str, boxes_with_text: List[Dict[str, Any]]) -> str:
        """
        Format a single line for Label.txt:
        <folderName>/<imageName>\t[{"transcription":"...","points":[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]},...]
        """
        path_str = f"{folder_name}/{img_name}"
        records = []
        for item in boxes_with_text:
            records.append({
                "transcription": item.get("transcription", ""),
                "points": item.get("points", [])
            })
        json_str = json.dumps(records, ensure_ascii=False, separators=(',', ':'))
        return f"{path_str}\t{json_str}\n"

    @staticmethod
    def format_state_line(folder_name: str, img_name: str, is_completed: bool = True) -> str:
        """
        Format a single line for fileState.txt:
        <folderName>/<imageName>\t1
        """
        path_str = f"{folder_name}/{img_name}"
        state = "1" if is_completed else "0"
        return f"{path_str}\t{state}\n"

    format_filestate_line = format_state_line

    @classmethod
    def write_ppocr_dataset(cls, folder_path: Path, folder_name: str, labels_by_image: Dict[str, List[Dict[str, Any]]]):
        """
        Write Label.txt and fileState.txt into folder_path.
        """
        folder_path.mkdir(parents=True, exist_ok=True)
        label_file = folder_path / "Label.txt"
        state_file = folder_path / "fileState.txt"
        
        with open(label_file, "w", encoding="utf-8") as fl, open(state_file, "w", encoding="utf-8") as fs:
            for img_name in sorted(labels_by_image.keys()):
                boxes = labels_by_image[img_name]
                fl.write(cls.format_label_line(folder_name, img_name, boxes))
                fs.write(cls.format_state_line(folder_name, img_name, is_completed=True))
                
        print(f"Generated PPOCRLabel dataset in {folder_path}: {len(labels_by_image)} images.")
