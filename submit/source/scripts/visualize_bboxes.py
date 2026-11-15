"""
Module: scripts/visualize_bboxes.py
Description: Render 4-point polygon bounding boxes and margin guides from PPOCRLabel
annotations directly onto scanned manuscript pages for visual verification.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import OUTPUT_DIR
from core.preprocess import cv2_imread_unicode


def render_book_bboxes(book_title: str, page_numbers: Optional[List[int]] = None, max_pages: int = 5):
    book_dir = OUTPUT_DIR / book_title
    if not book_dir.exists():
        print(f"Error: Output directory not found at {book_dir}")
        return False

    ch_dir = book_dir / "ch"
    vi_dir = book_dir / "vi"
    ch_label_file = ch_dir / "Label.txt"
    vi_label_file = vi_dir / "Label.txt"

    if not ch_label_file.exists() or not vi_label_file.exists():
        print(f"Error: Missing Label.txt in ch/ or vi/ for {book_title}")
        return False

    # Load PPOCRLabel annotations
    ch_labels = {}
    for line in ch_label_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parts = line.split("\t")
            ch_labels[Path(parts[0]).name] = json.loads(parts[1])

    vi_labels = {}
    for line in vi_label_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parts = line.split("\t")
            vi_labels[Path(parts[0]).name] = json.loads(parts[1])

    save_dir = book_dir / "annotated_samples"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(list(ch_dir.glob("*.jpg")))
    if page_numbers:
        target_images = []
        for pno in page_numbers:
            match = [img for img in all_images if f".{pno:03d}.jpg" in img.name or f".{pno:04d}.jpg" in img.name]
            target_images.extend(match)
    else:
        # Default: pick pages that have bounding boxes
        pages_with_boxes = [img for img in all_images if len(ch_labels.get(img.name, [])) > 0]
        target_images = pages_with_boxes[:max_pages]

    print("=" * 80)
    print(f"RENDERING BOUNDING BOX OVERLAYS FOR: {book_title}")
    print(f"Target Pages Count: {len(target_images)}")
    print(f"Save Directory     : {save_dir}")
    print("=" * 80)

    for img_p in target_images:
        img_name = img_p.name
        img = cv2_imread_unicode(img_p)
        if img is None:
            continue

        h, w, _ = img.shape

        # Draw red vertical guide at x=260 (student notebook left margin)
        cv2.line(img, (260, 0), (260, h), (0, 0, 255), 2)
        cv2.putText(img, "Red Margin x=260", (265, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        nom_boxes = ch_labels.get(img_name, [])
        qn_boxes = vi_labels.get(img_name, [])

        # Draw Chữ Nôm boxes in green
        for idx, b in enumerate(nom_boxes, 1):
            pts = np.array(b["points"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(0, 200, 0), thickness=3)
            p1 = b["points"][0]
            txt_label = f"N{idx}"
            cv2.putText(img, txt_label, (p1[0] - 35, p1[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)

        # Draw Quốc Ngữ boxes in blue
        for idx, b in enumerate(qn_boxes, 1):
            pts = np.array(b["points"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=True, color=(255, 120, 0), thickness=3)
            p1 = b["points"][0]
            txt_label = f"Q{idx}"
            cv2.putText(img, txt_label, (p1[0] - 35, p1[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2)

        # Legend header
        cv2.rectangle(img, (w - 380, 10), (w - 20, 95), (240, 240, 240), -1)
        cv2.rectangle(img, (w - 380, 10), (w - 20, 95), (100, 100, 100), 1)
        cv2.putText(img, "Legend:", (w - 365, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "[Green] Chu Nom BBox", (w - 365, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 2)
        cv2.putText(img, "[Blue]  Quoc Ngu BBox", (w - 365, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)

        out_name = img_name.replace(".jpg", "_annotated_bbox.jpg")
        out_path = save_dir / out_name
        is_success, buf = cv2.imencode(".jpg", img)
        with open(out_path, "wb") as f:
            f.write(buf)

        print(f"  [SAVED] {out_name} | Nom Boxes: {len(nom_boxes)} | QN Boxes: {len(qn_boxes)}")

    print("=" * 80)
    print("RENDERING FINISHED SUCCESSFULLY")
    print("=" * 80)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PPOCRLabel Bounding Boxes on Scanned Pages")
    parser.add_argument("--book", type=str, default="Ca Trù Thê Cách, AB 160, Nôm_Y", help="Book title")
    parser.add_argument("--pages", type=int, nargs="*", default=None, help="Page numbers to render (e.g. --pages 5 6 7)")
    parser.add_argument("--max_pages", type=int, default=5, help="Maximum number of sample pages to render")
    args = parser.parse_args()

    render_book_bboxes(args.book, args.pages, args.max_pages)
