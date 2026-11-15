"""
Master Pipeline Script for Automated Parallel Corpus Construction
Integrates all modules: Preprocessing -> Alignment -> Cleaning -> Evaluation -> Export
"""
import os
import sys
import shutil
import argparse
from typing import Optional
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import BOOKS, OUTPUT_DIR, DATASETS_MIDTERM_DIR, MIN_SIMILARITY_THRESHOLD
from core.preprocess import DocumentPreprocessor
from core.aligner import SentenceAligner
from core.cleaner import PPOCRLabelFormatter, TextCleaner
from core.evaluator import AlignmentEvaluator
from core.exporter import CorpusExporter
from core.classifier import BookClassifier

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def resolve_book_info(book_key: str) -> Optional[dict]:
    """Dynamically resolve book metadata from preconfigured BOOKS or filesystem."""
    if book_key in BOOKS:
        return BOOKS[book_key]
    for k, b in BOOKS.items():
        if b["title"].lower() == book_key.lower():
            return b

    pdf_dir = DATASETS_MIDTERM_DIR / "pdf"
    csv_dir = DATASETS_MIDTERM_DIR / "csv"
    target_stem = Path(book_key).stem
    matched_pdf = None
    for p in pdf_dir.glob("*.pdf"):
        if p.stem.lower() == target_stem.lower() or target_stem.lower() in p.stem.lower():
            matched_pdf = p
            break

    if matched_pdf:
        title = matched_pdf.stem
        csv_file = csv_dir / f"{title}.csv"
        cls_res = BookClassifier().classify(matched_pdf)
        return {
            "title": title,
            "pdf_filename": matched_pdf.name,
            "csv_filename": csv_file.name if csv_file.exists() else f"{title}.csv",
            "xlsx_filename": f"{title}.xlsx",
            "layout_type": cls_res["book_type"]
        }
    return None

def run_book_pipeline(book_key: str):
    book = resolve_book_info(book_key)
    if not book:
        print(f"Error: Unknown book key or unresolvable document: {book_key}")
        return False

    title = book["title"]
    print("=" * 80)
    print(f"RUNNING PIPELINE FOR: {title}")
    print("=" * 80)

    pdf_path = DATASETS_MIDTERM_DIR / "pdf" / book["pdf_filename"]
    csv_path = DATASETS_MIDTERM_DIR / "csv" / book["csv_filename"]

    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return False
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        return False

    book_out_dir = OUTPUT_DIR / title
    preprocessor = DocumentPreprocessor(book, OUTPUT_DIR)

    # ---------------------------------------------------------
    # STEP 1: PDF SPLITTING & ACCURATE SCRIPT-SPECIFIC PAGE CLASSIFICATION
    # ---------------------------------------------------------
    print("\n--- STEP 1: Extracting and Classifying Pages (Hán Nôm -> ch/, Quốc ngữ -> vi/, Phụ -> misc/) ---")
    extracted_pages = preprocessor.split_and_extract_images(str(pdf_path))
    print(f"Extracted {len(extracted_pages)} total pages.")

    # Determine script distribution according to guide/Quy_định_đồ_án.pdf
    ch_img_names = set()
    vi_img_names = set()

    if csv_path.exists():
        import pandas as pd
        df_csv = pd.read_csv(csv_path)
        if "target_folder" in df_csv.columns:
            for _, r in df_csv.iterrows():
                tgt = str(r.get("target_folder", "")).strip()
                img = str(r.get("img_id", "")).strip()
                if img and img != "nan":
                    if "ch" in tgt:
                        ch_img_names.add(img)
                    if "vi" in tgt:
                        vi_img_names.add(img)
        else:
            # Default T2b/T2bw: content pages contain both Hán Nôm and Quốc ngữ
            for img in df_csv["img_id"].dropna().unique():
                img_str = str(img).strip()
                ch_img_names.add(img_str)
                vi_img_names.add(img_str)

    ch_images = []
    vi_images = []
    misc_images = []

    for page_num, img_name, img_path in extracted_pages:
        has_ch = img_name in ch_img_names
        has_vi = img_name in vi_img_names

        if has_ch:
            ch_images.append((page_num, img_name, img_path))
            shutil.copy2(str(img_path), str(preprocessor.dir_ch / img_name))
        if has_vi:
            vi_images.append((page_num, img_name, img_path))
            shutil.copy2(str(img_path), str(preprocessor.dir_vi / img_name))

        if not has_ch and not has_vi:
            misc_images.append((page_num, img_name, img_path))
            shutil.copy2(str(img_path), str(preprocessor.dir_misc / img_name))

    print(f"Classification: ch (Hán Nôm)={len(ch_images)}, vi (Quốc ngữ)={len(vi_images)}, misc (Phụ/Bìa/Trắng)={len(misc_images)}")
    total_conserved = len(set([x[1] for x in ch_images]).union([x[1] for x in vi_images]).union([x[1] for x in misc_images]))
    print(f"Page Conservation: {total_conserved}/{len(extracted_pages)} pages preserved ({100 * total_conserved / len(extracted_pages):.1f}%).")

    # Copy original PDF to book directory
    shutil.copy2(str(pdf_path), str(book_out_dir / f"{title}.pdf"))

    # ---------------------------------------------------------
    # STEP 2: BILINGUAL SENTENCE ALIGNMENT & RESTORATION
    # ---------------------------------------------------------
    print("\n--- STEP 2: Bilingual Sentence Alignment & Reconstruction ---")
    aligner = SentenceAligner(book, csv_path)
    aligned_records = aligner.align_book_data()
    print(f"Aligned {len(aligned_records)} sentence pairs.")

    # ---------------------------------------------------------
    # STEP 3: SEGMENTATION & PPOCRLABEL GENERATION
    # ---------------------------------------------------------
    print("\n--- STEP 3: Generating PPOCRLabel Annotations (ch/ & vi/) ---")
    page_to_records = {}
    for r in aligned_records:
        img_id = r["img_id"]
        if img_id not in page_to_records:
            page_to_records[img_id] = []
        page_to_records[img_id].append(r)

    # Segment Hán Nôm lines for images in ch/
    is_vertical_nom = "Tnom" in book.get("layout_type", "")
    ch_labels_by_img = {}
    for page_num, img_name, img_path in ch_images:
        if is_vertical_nom:
            ch_boxes = preprocessor.segment_vertical_nom_columns(img_path)
        else:
            seg_res = preprocessor.segment_t2b_lines(img_path)
            ch_boxes = seg_res["ch"]
        recs_for_page = page_to_records.get(img_name, [])
        ch_annotated = []
        for b_idx, box in enumerate(ch_boxes):
            text = recs_for_page[b_idx]["SinoNom_Char"] if b_idx < len(recs_for_page) else ""
            ch_annotated.append({"transcription": text, "points": box["points"]})
        ch_labels_by_img[img_name] = ch_annotated

    # Segment Quốc ngữ lines for images in vi/
    vi_labels_by_img = {}
    for page_num, img_name, img_path in vi_images:
        seg_res = preprocessor.segment_t2b_lines(img_path)
        vi_boxes = seg_res["vi"]
        recs_for_page = page_to_records.get(img_name, [])
        vi_annotated = []
        for b_idx, box in enumerate(vi_boxes):
            text = recs_for_page[b_idx]["ChuQN_txt"] if b_idx < len(recs_for_page) else ""
            vi_annotated.append({"transcription": text, "points": box["points"]})
        vi_labels_by_img[img_name] = vi_annotated

    PPOCRLabelFormatter.write_ppocr_dataset(preprocessor.dir_ch, "ch", ch_labels_by_img)
    PPOCRLabelFormatter.write_ppocr_dataset(preprocessor.dir_vi, "vi", vi_labels_by_img)

    # ---------------------------------------------------------
    # STEP 4: QUALITY EVALUATION & SIMILARITY SCORING
    # ---------------------------------------------------------
    print("\n--- STEP 4: Evaluating Alignment & Similarity Metric ---")
    evaluator = AlignmentEvaluator()
    eval_report = evaluator.evaluate_dataset(aligned_records)

    print(f"Total Evaluated Characters : {eval_report['total_chars']}")
    print(f"Total Matched (Black) Chars: {eval_report['total_black_chars']}")
    print(f"Total Mismatched (Red) Chars: {eval_report['total_red_chars']}")
    print(f"Similarity Percentage      : {eval_report['similarity_percentage']:.2f}%")
    print(f"Requirement Passed (>=95%) : {'YES [PASSED]' if eval_report['is_passed'] else 'NO [FAILED]'}")

    # ---------------------------------------------------------
    # STEP 5: EXPORT XLSX (COLOR-CODED) & CSV
    # ---------------------------------------------------------
    print("\n--- STEP 5: Exporting CSV & Color-coded XLSX ---")
    out_csv = book_out_dir / f"{title}.csv"
    out_xlsx = book_out_dir / f"{title}.xlsx"

    CorpusExporter.export_csv(aligned_records, out_csv)
    CorpusExporter.export_xlsx(aligned_records, eval_report["row_evaluations"], out_xlsx)

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY FOR: {title}")
    print(f"Output files stored in: {book_out_dir}")
    print("=" * 80 + "\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Parallel Corpus Construction Pipeline")
    parser.add_argument("--book", type=str, default="AB_160",
                        help="Book to process: 'all' (primary 4 books), 'all_catalog' (all 23 books), or specific book key/title/filename")
    args = parser.parse_args()

    if args.book == "all":
        print(f"Running batch processing across primary books: {list(BOOKS.keys())}")
        for k in BOOKS.keys():
            run_book_pipeline(k)
    elif args.book == "all_catalog":
        pdf_dir = DATASETS_MIDTERM_DIR / "pdf"
        catalog_pdfs = sorted(list(pdf_dir.glob("*.pdf")))
        print(f"Running full catalog batch processing across {len(catalog_pdfs)} books...")
        for p in catalog_pdfs:
            run_book_pipeline(p.stem)
    else:
        run_book_pipeline(args.book)
