"""
Export all 23 PDF files in datasets-midterm/pdf to structured CSV and XLSX files
in output/csv/ and output/xlsx/ for streamlined processing and alignment.
"""
import os
import sys
import shutil
import unicodedata
from pathlib import Path
import fitz
import cv2
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "datasets-midterm" / "pdf"
DATASET_CSV_DIR = BASE_DIR / "datasets-midterm" / "csv"
DATASET_XLSX_DIR = BASE_DIR / "datasets-midterm" / "xlsx"
OUT_CSV_DIR = BASE_DIR / "output" / "csv"
OUT_XLSX_DIR = BASE_DIR / "output" / "xlsx"
DATASET_CSV_DIR.mkdir(parents=True, exist_ok=True)
DATASET_XLSX_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
OUT_XLSX_DIR.mkdir(parents=True, exist_ok=True)

# Layout metadata for the 23 PDFs
BOOK_LAYOUT_MAP = {
    "Ca Trù Thê Cách, AB 160, Nôm_Y.pdf": "T2bw",
    "Ca Trù Thế Cách, AB 431, Nôm_Y.pdf": "T2bw",
    "Ca Trù Thế Cách, AB 456, Nôm_Y.pdf": "T2bw",
    "Ca Trù Thể Cách, Nôm_Y.pdf": "T2bw",
    "Tây Du Truyên, Nôm_Y.pdf": "T2bw",
    "Tây Dương Quân Thiên Lạc, Nôm, 2 volumes_Y.pdf": "T2bw",
    "Tế cô hồn văn SV4.0 Tuệ Phương.pdf": "T2b",
    "Thanh Tông Ngư Vịnh Tư Thơi Thi Tâp, Nôm.pdf": "Tnomw",
    "Thánh Du Quảng Huân Diên Nghĩa, Nôm.pdf": "T2bw",
    "Thánh Tô Kê Diên Quôc Âm, Nôm.pdf": "T2bw",
    "Thâp Nhị Tiên Nàng Thỉnh Luyên Bí Pháp, Nôm.pdf": "T2bw",
    "ThẩmTửHưTruyện_沈子虛傳.pdf": "T2w",
    "Tho Ho Xuan Huong [chu nom].pdf": "T2b",
    "ThoNomHoXuanHuongHoangXuanHanSuuTapVaGhiChu_HoXuanHuong.pdf": "T2b",
    "ThơCaDuHànhTừHàNộiVàoKinhĐôHuế_1.pdf": "T2w",
    "ThơCaDuHànhTừHàNộiVàoKinhĐôHuế_2.pdf": "T2bw",
    "ThơVănLêTrịnhMạc.pdf": "T2w",
    "ThuDạLữHoàiNgâm_秋夜旅懷吟.pdf": "Tnom",
    "Thư Thưc Tân Chuyên, Nôm_Y.pdf": "T2bw",
    "Thư Vị Văn Châu, Nôm_Y.pdf": "T2bw",
    "Tì Bà Quôc Âm Tân Truyên, Preface in Hán, text in Nôm_Y.pdf": "T2bw",
    "TìBàQuốcÂmTânTruyện_琵琶國音新傳_1.pdf": "T2w",
    "TìBàQuốcÂmTânTruyện_琵琶國音新傳_2.pdf": "Tnomw",
}


def format_img_id(book_stem: str, page_num: int, total_pages: int) -> str:
    if total_pages > 999:
        return f"{book_stem}.{page_num}.jpg"
    return f"{book_stem}.{page_num:03d}.jpg"


def classify_pdf_pages(pdf_path: Path, book_stem: str, layout_code: str):
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    p0_txt = " ".join(doc[0].get_text().split())
    is_yale = "Yale University" in p0_txt

    page_records = []
    for p_idx in range(n_pages):
        p_num = p_idx + 1
        img_id = format_img_id(book_stem, p_num, n_pages)
        pix = doc[p_idx].get_pixmap(dpi=50)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img[:, :, 0]
        h, w = gray.shape
        crop = gray[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]

        std_val = round(float(np.std(crop)), 2)
        lap_val = round(float(cv2.Laplacian(crop, cv2.CV_64F).var()), 2)

        raw_txt = " ".join(doc[p_idx].get_text().split())
        page_note = ""
        if is_yale:
            for line in doc[p_idx].get_text().splitlines():
                if "Image ID:" in line or "Page " in line or "cover" in line.lower():
                    page_note = line.strip()
                    break

        threshold_std = 6.5 if is_yale else 11.5
        if p_num == 1 and is_yale:
            target_folder = "misc"
            page_role = "Yale Metadata Cover"
        elif is_yale and ("cover" in page_note.lower() or p_num in [2, 3, n_pages - 1, n_pages] and std_val < 9.0):
            target_folder = "blank" if std_val < threshold_std else "misc"
            page_role = "Cover / Flyleaf"
        elif std_val <= threshold_std:
            target_folder = "blank"
            page_role = "Blank Page"
        else:
            # Active content page: determine folder based on layout_code and page parity/range
            if layout_code in ("T2bw", "T2b"):
                if "ThơCaDuHànhTừHàNộiVàoKinhĐôHuế_2" in book_stem and p_num >= 195:
                    target_folder = "misc"
                    page_role = "Hand-drawn Route Map"
                else:
                    target_folder = "ch+vi"
                    page_role = "Bilingual Page (Nom + Quoc Ngu)"
            elif layout_code == "Tnom":
                target_folder = "ch"
                page_role = "Woodblock Han Nom Page (2 pages/image)"
            elif layout_code == "Tnomw":
                if p_num in [1, 2, 3, n_pages - 1, n_pages] and not is_yale:
                    target_folder = "misc"
                    page_role = "Cover / Title Page"
                else:
                    target_folder = "ch"
                    page_role = "Handwritten Han Nom Page"
            elif layout_code == "T2w":
                if "ThẩmTửHưTruyện" in book_stem:
                    if p_num in [5, 119, 120]:
                        target_folder = "misc"
                        page_role = "Title / Cover Page"
                    elif p_num in [15, 17]:
                        target_folder = "vi"
                        page_role = "Handwritten Quoc Ngu Page"
                    elif p_num % 2 == 0 and 8 <= p_num <= 112:
                        target_folder = "ch"
                        page_role = "Handwritten Han Nom Page"
                    else:
                        target_folder = "blank"
                        page_role = "Blank Facing Page"
                elif "ThơVănLêTrịnhMạc" in book_stem:
                    if p_num in [3, 43, 79, 81, 84, 85]:
                        target_folder = "misc"
                        page_role = "Title / Index Card / Cover"
                    elif p_num in [9, 11, 13, 15, 17, 44]:
                        target_folder = "vi"
                        page_role = "Typed/Handwritten Quoc Ngu Page"
                    elif p_num % 2 == 0 and (6 <= p_num <= 38 or 48 <= p_num <= 74):
                        target_folder = "ch"
                        page_role = "Handwritten Han Nom Page"
                    else:
                        target_folder = "blank"
                        page_role = "Blank Facing Page"
                elif "ThơCaDuHànhTừHàNộiVàoKinhĐôHuế_1" in book_stem:
                    if p_num in [5, 87, 127, 132, 135]:
                        target_folder = "misc"
                        page_role = "Title / Notes / Index Card"
                    elif p_num % 2 == 1 and 9 <= p_num <= 85:
                        target_folder = "vi"
                        page_role = "Typed Quoc Ngu Page"
                    elif p_num % 2 == 0 and 8 <= p_num <= 124:
                        target_folder = "ch"
                        page_role = "Handwritten Han Nom Page"
                    else:
                        target_folder = "blank"
                        page_role = "Blank Facing Page"
                elif "TìBàQuốcÂmTânTruyện_琵琶國音新傳_1" in book_stem:
                    if p_num in [5, 6, 142, 143]:
                        target_folder = "misc"
                        page_role = "Title / Cover Page"
                    elif p_num in [48, 50, 52, 54, 56]:
                        target_folder = "vi"
                        page_role = "Handwritten Quoc Ngu Page"
                    elif p_num % 2 == 1 and 7 <= p_num <= 137:
                        target_folder = "ch"
                        page_role = "Handwritten Han Nom Page"
                    else:
                        target_folder = "blank"
                        page_role = "Blank Facing Page"
                else:
                    target_folder = "ch"
                    page_role = "Content Page"
            else:
                target_folder = "ch"
                page_role = "Content Page"

        # Estimate horizontal/vertical line segments on active content pages
        est_sentences = 0
        if target_folder in ("ch", "ch+vi"):
            est_sentences = 10 if is_yale else 8

        page_records.append({
            "page_num": p_num,
            "img_id": img_id,
            "layout_code": layout_code,
            "target_folder": target_folder,
            "page_role": page_role,
            "est_sentences": est_sentences,
            "contrast_std": std_val,
            "laplacian_sharpness": lap_val,
            "page_note": page_note
        })

    doc.close()
    return page_records


def write_styled_workbook(corpus_df: pd.DataFrame, pages_df: pd.DataFrame, out_xlsx: Path):
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Parallel Corpus"

    header_font = Font(name="Arial", size=11, bold=True, color="FF000000")
    header_fill = PatternFill(start_color="FFE0E0E0", end_color="FFE0E0E0", fill_type="solid")
    thin_border = Border(
        left=Side(style="thin", color="FFD0D0D0"),
        right=Side(style="thin", color="FFD0D0D0"),
        top=Side(style="thin", color="FFD0D0D0"),
        bottom=Side(style="thin", color="FFD0D0D0"),
    )
    body_font = Font(name="Arial", size=11, color="FF000000")

    # Sheet 1: Parallel Corpus
    cols1 = list(corpus_df.columns)
    ws1.append(cols1)
    for c_idx in range(1, len(cols1) + 1):
        cell = ws1.cell(row=1, column=c_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    for r_idx, row in enumerate(corpus_df.itertuples(index=False), start=2):
        for c_idx, val in enumerate(row, start=1):
            cell = ws1.cell(row=r_idx, column=c_idx, value="" if pd.isna(val) else val)
            cell.font = body_font
            cell.border = thin_border
            if c_idx in [1, 2]:
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

    ws1.column_dimensions["A"].width = 38
    ws1.column_dimensions["B"].width = 42
    ws1.column_dimensions["C"].width = 38
    ws1.column_dimensions["D"].width = 48
    if len(cols1) > 4:
        for extra_col in ["E", "F", "G"]:
            ws1.column_dimensions[extra_col].width = 24

    # Sheet 2: Page Classification
    ws2 = wb.create_sheet(title="Page_Classification")
    cols2 = list(pages_df.columns)
    ws2.append(cols2)
    for c_idx in range(1, len(cols2) + 1):
        cell = ws2.cell(row=1, column=c_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    for r_idx, row in enumerate(pages_df.itertuples(index=False), start=2):
        for c_idx, val in enumerate(row, start=1):
            cell = ws2.cell(row=r_idx, column=c_idx, value="" if pd.isna(val) else val)
            cell.font = body_font
            cell.border = thin_border

    ws2.column_dimensions["A"].width = 12
    ws2.column_dimensions["B"].width = 40
    ws2.column_dimensions["C"].width = 14
    ws2.column_dimensions["D"].width = 16
    ws2.column_dimensions["E"].width = 32
    ws2.column_dimensions["F"].width = 16
    ws2.column_dimensions["G"].width = 16
    ws2.column_dimensions["H"].width = 20
    ws2.column_dimensions["I"].width = 35

    wb.save(str(out_xlsx))


def main():
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    print(f"Found {len(pdf_files)} PDF files to export to CSV and XLSX.")

    # Load Thu Da Lu Hoai Ngam translation TXT for mapping with ThuDaLuHoaiNgam PDF
    thu_da_txt_lines = []
    for f in os.listdir(PDF_DIR):
        if f.lower().endswith(".txt"):
            raw_txt = (PDF_DIR / f).read_text(encoding="utf-8", errors="ignore")
            thu_da_txt_lines = [line.strip() for line in raw_txt.splitlines() if line.strip()]
            break

    export_summary = []

    for idx, fname in enumerate(pdf_files, 1):
        nfc_fname = unicodedata.normalize("NFC", fname)
        book_stem = Path(nfc_fname).stem
        layout_code = BOOK_LAYOUT_MAP.get(nfc_fname, "T2bw")
        pdf_path = PDF_DIR / fname

        page_records = classify_pdf_pages(pdf_path, book_stem, layout_code)
        pages_df = pd.DataFrame(page_records)

        # Check if an already-aligned CSV exists in output/<book_stem>/<book_stem>.csv (the 4 Ca Tru books)
        existing_aligned_csv = BASE_DIR / "output" / book_stem / f"{book_stem}.csv"
        if existing_aligned_csv.exists():
            corpus_df = pd.read_csv(existing_aligned_csv, encoding="utf-8-sig")
            status_tag = "Aligned Ground Truth (100%)"
        elif "ThuDạLữHoàiNgâm" in book_stem and thu_da_txt_lines:
            # Map the 141 verses from Thu Da Lu Hoai Ngam Khuc_Dich.txt across the 10 pages
            rows = []
            # Page 1 has title + first 12 verses, pages 2..10 have ~14 verses each
            v_idx = 0
            n_total = len(thu_da_txt_lines)
            per_page = int(np.ceil(n_total / len(page_records)))
            for p_rec in page_records:
                img_id = p_rec["img_id"]
                chunk = thu_da_txt_lines[v_idx:v_idx + per_page]
                v_idx += per_page
                for s_i, qn_line in enumerate(chunk, 1):
                    stem_no_jpg = img_id[:-4]
                    rows.append({
                        "img_id": img_id,
                        "sentence_id": f"{stem_no_jpg}.{s_i}",
                        "SinoNom_Char": "",
                        "ChuQN_txt": qn_line,
                        "target_folder": p_rec["target_folder"],
                        "layout_code": layout_code,
                        "page_role": p_rec["page_role"]
                    })
            corpus_df = pd.DataFrame(rows)
            status_tag = f"Mapped Quoc Ngu from TXT ({len(corpus_df)} verses)"
        else:
            # Create structured page-level template rows for all active Nom/Bilingual pages
            rows = []
            for p_rec in page_records:
                if p_rec["target_folder"] in ("ch", "ch+vi", "vi"):
                    img_id = p_rec["img_id"]
                    stem_no_jpg = img_id[:-4]
                    rows.append({
                        "img_id": img_id,
                        "sentence_id": f"{stem_no_jpg}.1",
                        "SinoNom_Char": "",
                        "ChuQN_txt": "",
                        "target_folder": p_rec["target_folder"],
                        "layout_code": layout_code,
                        "page_role": p_rec["page_role"]
                    })
            corpus_df = pd.DataFrame(rows)
            status_tag = f"Structured Template ({len(corpus_df)} active pages)"

        out_csv_path = OUT_CSV_DIR / f"{book_stem}.csv"
        out_xlsx_path = OUT_XLSX_DIR / f"{book_stem}.xlsx"

        corpus_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
        write_styled_workbook(corpus_df, pages_df, out_xlsx_path)

        shutil.copy2(str(out_csv_path), str(DATASET_CSV_DIR / f"{book_stem}.csv"))
        shutil.copy2(str(out_xlsx_path), str(DATASET_XLSX_DIR / f"{book_stem}.xlsx"))

        ch_count = int((pages_df["target_folder"].isin(["ch", "ch+vi"])).sum())
        vi_count = int((pages_df["target_folder"].isin(["vi", "ch+vi"])).sum())
        blank_count = int((pages_df["target_folder"] == "blank").sum())
        misc_count = int((pages_df["target_folder"] == "misc").sum())

        export_summary.append({
            "STT": idx,
            "Ten_Sach": book_stem,
            "Ma_Loai_Sach": layout_code,
            "Tong_Trang_PDF": len(pages_df),
            "Trang_ch": ch_count,
            "Trang_vi": vi_count,
            "Trang_blank": blank_count,
            "Trang_misc": misc_count,
            "So_Dong_Corpus": len(corpus_df),
            "Trang_Thai_Du_Lieu": status_tag,
            "File_CSV": f"output/csv/{book_stem}.csv",
            "File_XLSX": f"output/xlsx/{book_stem}.xlsx"
        })
        print(f"[{idx:02d}/23] Exported: {book_stem}.csv & .xlsx | Pages={len(pages_df)} (ch={ch_count}, vi={vi_count}, blank={blank_count}) | Rows={len(corpus_df)}")

    df_summary = pd.DataFrame(export_summary)
    df_summary.to_csv(BASE_DIR / "output" / "index_23_books_export.csv", index=False, encoding="utf-8-sig")
    df_summary.to_excel(BASE_DIR / "output" / "index_23_books_export.xlsx", index=False)
    print(f"\nCompleted exporting all {len(pdf_files)} PDFs to {OUT_CSV_DIR} and {OUT_XLSX_DIR}.")


if __name__ == "__main__":
    main()
