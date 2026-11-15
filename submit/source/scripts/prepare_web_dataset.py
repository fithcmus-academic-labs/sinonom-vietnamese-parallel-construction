"""
Prepare All 4 Books Dataset for Web Studio
Copies scan images and generates complete JSON data for:
1. NOM_Y (101 pages)
2. AB_431 (41 pages)
3. AB_160 (45 pages)
4. AB_456 (45 pages)
"""
import os
import sys
import json
import shutil
import csv
import re
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "output"
WEB_STUDIO_DIR = BASE_DIR / "web-studio"
PUBLIC_DIR = WEB_STUDIO_DIR / "public"
SERVICES_DIR = WEB_STUDIO_DIR / "src" / "services"

from core.config import BOOKS
from core.evaluator import AlignmentEvaluator

evaluator = AlignmentEvaluator()

def process_all_books():
    all_books_data = {}

    for book_key, book_info in BOOKS.items():
        title = book_info["title"]
        pdf_name = book_info["pdf_filename"]
        book_dir = OUTPUT_DIR / title
        extracted_dir = book_dir / "extracted_all"
        ch_label_path = book_dir / "ch" / "Label.txt"
        vi_label_path = book_dir / "vi" / "Label.txt"
        csv_path = book_dir / f"{title}.csv"
        pdf_path = book_dir / pdf_name

        print(f"\n==========================================")
        print(f"PROCESSING BOOK: {book_key} -> {title}")
        print(f"==========================================")

        # 1. Target public image folder
        book_public_dir = PUBLIC_DIR / "dataset" / book_key
        book_public_dir.mkdir(parents=True, exist_ok=True)

        extracted_images = sorted(list(extracted_dir.glob("*.jpg")))
        total_pages = len(extracted_images)
        print(f"Found {total_pages} extracted page images.")

        page_sizes = {}
        for img_path in extracted_images:
            # Parse page number from filename e.g. "Ca Trù Thể Cách, Nôm_Y.005.jpg"
            m = re.search(r'\.(\d+)\.jpg$', img_path.name)
            p_num = int(m.group(1)) if m else int(re.search(r'(\d+)', img_path.stem).group(1))
            
            # Copy as both page_XXX.jpg and original filename
            dest_page = book_public_dir / f"page_{p_num:03d}.jpg"
            dest_orig = book_public_dir / img_path.name
            
            if not dest_page.exists() or dest_page.stat().st_size != img_path.stat().st_size:
                shutil.copy2(img_path, dest_page)
            if not dest_orig.exists() or dest_orig.stat().st_size != img_path.stat().st_size:
                shutil.copy2(img_path, dest_orig)

            # Read dimensions
            try:
                with Image.open(img_path) as im:
                    page_sizes[p_num] = im.size # (width, height)
            except Exception:
                page_sizes[p_num] = (1505, 2000)

        # 2. Parse Label.txt for CH and VI
        boxes_by_page = {p: [] for p in range(1, total_pages + 1)}
        all_boxes = []

        def parse_labels(label_file, region_type):
            if not label_file.exists():
                return
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    file_rel = parts[0]
                    box_list_json = parts[1]
                    if box_list_json == '[]':
                        continue
                    
                    m = re.search(r'\.(\d+)\.jpg$', file_rel)
                    if not m:
                        continue
                    p_num = int(m.group(1))
                    
                    try:
                        box_entries = json.loads(box_list_json)
                    except Exception as e:
                        print(f"JSON parse error on {file_rel}: {e}")
                        continue

                    img_filename = f"{title}.{p_num:03d}.jpg"
                    for idx, b_entry in enumerate(box_entries):
                        pts = b_entry.get("points", [])
                        txt = b_entry.get("transcription", "")
                        if len(pts) != 4:
                            continue
                        
                        box_obj = {
                            "id": f"box_{region_type}_{p_num}_{idx + 1}",
                            "img_id": img_filename,
                            "sentence_id": f"{title}.{p_num:03d}.{idx + 1}",
                            "region_type": region_type,
                            "transcription": txt,
                            "points": pts,
                            "confidence": 0.98,
                            "status": "matched",
                            "isModified": False
                        }
                        if p_num in boxes_by_page:
                            boxes_by_page[p_num].append(box_obj)
                        all_boxes.append(box_obj)

        parse_labels(ch_label_path, "ch")
        parse_labels(vi_label_path, "vi")
        print(f"Parsed {len(all_boxes)} bounding boxes across {len(boxes_by_page)} pages.")

        # 3. Read and evaluate CSV rows
        records = []
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                raw_rows = list(reader)

            eval_res = evaluator.evaluate_dataset(raw_rows)
            for idx, (raw_row, r_eval) in enumerate(zip(raw_rows, eval_res["row_evaluations"])):
                img_id = raw_row.get("img_id", "")
                m = re.search(r'\.(\d+)\.jpg$', img_id)
                p_num = int(m.group(1)) if m else 5
                
                # Sentence order on page
                s_id = raw_row.get("sentence_id", "")
                m_sub = re.search(r'\.(\d+)$', s_id)
                line_idx = int(m_sub.group(1)) if m_sub else (idx + 1)

                is_matched = r_eval["mismatched_chars"] == 0
                color = "BLACK" if is_matched else "RED"
                
                # Build charStatuses
                char_statuses = []
                for cs in r_eval["char_statuses"]:
                    if isinstance(cs, (list, tuple)):
                        ch_val = cs[0]
                        is_match = bool(cs[1])
                        reason_str = cs[2] if len(cs) > 2 else ""
                    else:
                        ch_val = cs.get("char", "")
                        is_match = bool(cs.get("is_matched", True))
                        reason_str = cs.get("reason", "")
                    
                    char_statuses.append({
                        "char": ch_val,
                        "isMatched": is_match,
                        "color": "BLACK" if is_match else "RED",
                        "targetQn": "",
                        "readings": [],
                        "reason": reason_str
                    })

                records.append({
                    "id": f"rec_{book_key}_{idx + 1}",
                    "img_id": img_id,
                    "sentence_id": s_id,
                    "SinoNom_Char": raw_row.get("SinoNom_Char", ""),
                    "ChuQN_txt": raw_row.get("ChuQN_txt", ""),
                    "pageNumber": p_num,
                    "lineIndex": line_idx,
                    "isMatched": is_matched,
                    "color": color,
                    "charStatuses": char_statuses,
                    "reviewStatus": "approved" if is_matched else "pending",
                    "isEdited": False
                })
        print(f"Processed {len(records)} alignment records from CSV.")

        # 4. Generate PageDocument objects (1 .. total_pages)
        pages = []
        for p_num in range(1, total_pages + 1):
            w, h = page_sizes.get(p_num, (1505, 2000))
            is_misc = p_num <= 3
            category = "misc" if is_misc else "content"
            
            p_boxes = boxes_by_page.get(p_num, [])
            ch_boxes = [b for b in p_boxes if b["region_type"] == "ch"]
            vi_boxes = [b for b in p_boxes if b["region_type"] == "vi"]
            
            p_records = [r for r in records if r["pageNumber"] == p_num]
            mismatches = sum(1 for r in p_records if not r["isMatched"])

            pages.append({
                "pageNumber": p_num,
                "pageId": f"{p_num:03d}",
                "imgName": f"{title}.{p_num:03d}.jpg",
                "imageUrl": f"/dataset/{book_key}/page_{p_num:03d}.jpg",
                "category": category,
                "reviewStatus": "approved" if mismatches == 0 else "in_progress",
                "width": w,
                "height": h,
                "detectedChRegions": len(ch_boxes),
                "detectedViRegions": len(vi_boxes),
                "mismatchedCount": mismatches
            })

        # 5. Calculate Metrics
        total_chars = sum(len(r["SinoNom_Char"]) for r in records)
        mismatched_chars = sum(
            sum(1 for cs in r["charStatuses"] if not cs["isMatched"])
            for r in records
        )
        sim_score = (1.0 - (mismatched_chars / total_chars)) * 100.0 if total_chars > 0 else 100.0
        validated = sum(1 for r in records if r["isMatched"])

        metrics = {
            "totalPages": total_pages,
            "processedPages": total_pages,
            "totalAlignedPairs": len(records),
            "totalSinoNomChars": total_chars,
            "totalVietnameseWords": sum(len(r["ChuQN_txt"].split()) for r in records),
            "validatedPairs": validated,
            "mismatchedChars": mismatched_chars,
            "pendingReviews": len(records) - validated,
            "similarityScore": round(sim_score, 2),
            "isSimilarityPassed": sim_score >= 95.0
        }

        # 6. Metadata
        pdf_size = pdf_path.stat().st_size if pdf_path.exists() else 0
        metadata = {
            "id": f"proj_{book_key.lower()}",
            "bookKey": book_key,
            "bookName": title,
            "studentIds": ["21120001", "21120002"],
            "originalPdfName": pdf_name,
            "originalPdfSize": pdf_size,
            "totalPages": total_pages,
            "contentPages": total_pages - 3,
            "miscPages": 3,
            "createdAt": "2026-10-31T23:00:00Z",
            "updatedAt": "2026-11-01T01:00:00Z"
        }

        all_books_data[book_key] = {
            "metadata": metadata,
            "metrics": metrics,
            "pages": pages,
            "records": records,
            "boxesByPage": boxes_by_page
        }

        print(f"Book {book_key} processed successfully: {total_pages} pages, {len(records)} records, similarity {sim_score:.2f}%")

    # Save to JSON
    json_path = SERVICES_DIR / "allBooksDataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_books_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved allBooksDataset.json ({json_path.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    process_all_books()
