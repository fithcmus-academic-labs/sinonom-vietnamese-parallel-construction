"""
Exporter Module
Exports aligned parallel corpus to CSV and styled color-coded XLSX.
Packages final submission ZIP structure according to strict project specifications.
"""
import os
import shutil
import zipfile
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from pathlib import Path
from typing import List, Dict, Any
from core.config import COLOR_BLACK, COLOR_RED


class CorpusExporter:
    REQUIRED_COLUMNS = ["img_id", "sentence_id", "SinoNom_Char", "ChuQN_txt"]

    @staticmethod
    def export_csv(records: List[Dict[str, Any]], out_csv_path: Path):
        """
        Export records to CSV using utf-8-sig encoding (BOM) to prevent font issues in Excel.
        Safely handles empty record lists by writing the 4 required header columns.
        """
        out_csv_path = Path(out_csv_path)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            df = pd.DataFrame(columns=CorpusExporter.REQUIRED_COLUMNS)
        else:
            df = pd.DataFrame(records)
            for col in CorpusExporter.REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df = df[CorpusExporter.REQUIRED_COLUMNS]
        df.to_csv(str(out_csv_path), index=False, encoding="utf-8-sig")
        print(f"Exported CSV to {out_csv_path} ({len(df)} rows).")

    @staticmethod
    def export_xlsx(records: List[Dict[str, Any]], row_evals: List[Dict[str, Any]], out_xlsx_path: Path):
        """
        Export records to styled XLSX with black/red color-coding based on alignment evaluation.
        """
        out_xlsx_path = Path(out_xlsx_path)
        out_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Parallel Corpus"

        # Headers
        headers = CorpusExporter.REQUIRED_COLUMNS
        ws.append(headers)

        header_font = Font(name="Arial", size=11, bold=True, color="FF000000")
        header_fill = PatternFill(start_color="FFE0E0E0", end_color="FFE0E0E0", fill_type="solid")
        thin_border = Border(
            left=Side(style='thin', color='FFD0D0D0'),
            right=Side(style='thin', color='FFD0D0D0'),
            top=Side(style='thin', color='FFD0D0D0'),
            bottom=Side(style='thin', color='FFD0D0D0')
        )

        for col_idx in range(1, 5):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border

        font_black = Font(name="Arial", size=11, color=COLOR_BLACK)
        font_red = Font(name="Arial", size=11, color=COLOR_RED, bold=True)

        for row_idx, (rec, eval_info) in enumerate(zip(records, row_evals), start=2):
            # Check row color: If all matched -> BLACK; If has mismatch -> RED
            is_all_matched = eval_info.get("mismatched_chars", 0) == 0
            row_font = font_black if is_all_matched else font_red

            vals = [
                rec.get("img_id", ""),
                rec.get("sentence_id", ""),
                rec.get("SinoNom_Char", ""),
                rec.get("ChuQN_txt", "")
            ]
            for col_idx, val in enumerate(vals, start=1):
                cell = ws.cell(row=row_idx, column=col_idx, value=val)
                cell.font = row_font
                cell.border = thin_border
                if col_idx in [1, 2]:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")

        # Set column widths
        ws.column_dimensions['A'].width = 30
        ws.column_dimensions['B'].width = 35
        ws.column_dimensions['C'].width = 40
        ws.column_dimensions['D'].width = 50

        wb.save(str(out_xlsx_path))
        print(f"Exported styled XLSX to {out_xlsx_path} ({len(records)} rows).")

    @staticmethod
    def package_submission_zip(book_dir: Path, zip_dest_path: Path):
        """
        Package book directory into ZIP file.
        """
        book_dir = Path(book_dir)
        zip_dest_path = Path(zip_dest_path)
        zip_dest_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(zip_dest_path), 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(str(book_dir)):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, str(book_dir.parent))
                    zipf.write(full_path, rel_path)
        print(f"Packaged submission archive to {zip_dest_path} ({os.path.getsize(zip_dest_path)} bytes).")
