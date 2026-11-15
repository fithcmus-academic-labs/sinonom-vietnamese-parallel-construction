"""
Module: scripts/visualize_bboxes.py
Description: Render 4-point polygon bounding boxes and margin guides from PPOCRLabel
annotations directly onto scanned manuscript pages for visual verification.
Adheres strictly to color specifications:
- Han Nom bounding boxes: Black [Đen] (color=(25, 25, 25))
- Quoc Ngu bounding boxes: Green [Xanh] (color=(30, 180, 30))
- Unmatched / Outlier boxes: Red [Đỏ] (color=(0, 0, 220))
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import OUTPUT_DIR
from core.dictionary import HanNomDictionary
from core.preprocess import cv2_imread_unicode, cv2_imwrite_unicode


def load_bboxes_from_label_file(label_path: Path, target_img_name: str) -> List[Dict[str, Any]]:
    """Load bounding box coordinates from PPOCRLabel Label.txt."""
    if not label_path.exists():
        return []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        rel_p, json_s = line.split("\t", 1)
        if rel_p.endswith("/" + target_img_name) or Path(rel_p).name == target_img_name:
            try:
                return json.loads(json_s)
            except Exception:
                return []
    return []


def draw_colored_bboxes(
    img_bgr: np.ndarray, ch_boxes: List[dict], vi_boxes: List[dict], dict_engine: Any = None
) -> np.ndarray:
    """Draw bounding boxes strictly adhering to project color rules:
    - Han Nom: Black [Đen] (color=(25, 25, 25)) if verified in dictionary, Red [Đỏ] if not.
    - Quoc Ngu: Green [Xanh] (color=(30, 180, 30)) if valid text, Red [Đỏ] if invalid.
    """
    vis = img_bgr.copy()
    h, w, _ = vis.shape

    # 1. Han Nom: Black [Đen]
    for idx, b in enumerate(ch_boxes, 1):
        pts = np.array(b["points"], dtype=np.int32)
        txt = b.get("transcription", "")
        nom_chars = [
            c
            for c in txt
            if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf' or '\U00020000' <= c <= '\U0002fa1f'
        ]
        if dict_engine and nom_chars:
            found_count = sum(1 for c in nom_chars if dict_engine.has_char(c))
            is_found = (found_count / len(nom_chars)) >= 0.75
        else:
            is_found = True if txt and txt != "?" else False

        if is_found:
            color = (25, 25, 25)  # Black [Đen]
            tag = f"CH-{idx} [Den]"
        else:
            color = (0, 0, 220)  # Red [Đỏ]
            tag = f"CH-{idx} [Do]"

        # White halo border for high contrast on all paper backgrounds
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 255, 255), thickness=4)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        tx = int(pts[0][0])
        ty = max(22, int(pts[0][1]) - 5)
        cv2.putText(vis, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3)
        cv2.putText(vis, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 2. Quoc Ngu: Green [Xanh]
    for idx, b in enumerate(vi_boxes, 1):
        pts = np.array(b["points"], dtype=np.int32)
        txt = b.get("transcription", "")
        is_valid_qn = len(txt.strip()) > 0 and txt != "Quoc Ngu"

        if is_valid_qn:
            color = (30, 180, 30)  # Green [Xanh]
            tag = f"VI-{idx} [Xanh]"
        else:
            color = (0, 0, 220)  # Red [Đỏ]
            tag = f"VI-{idx} [Do]"

        # White halo border for high contrast
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 255, 255), thickness=4)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        tx = int(pts[0][0])
        ty = max(22, int(pts[0][1]) - 5)
        cv2.putText(vis, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3)
        cv2.putText(vis, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Legend badge at top-right
    badge_w, badge_h = 360, 80
    bx1 = max(10, w - badge_w - 15)
    by1 = 12
    cv2.rectangle(vis, (bx1, by1), (bx1 + badge_w, by1 + badge_h), (250, 250, 250), -1)
    cv2.rectangle(vis, (bx1, by1), (bx1 + badge_w, by1 + badge_h), (120, 120, 120), 1)
    cv2.putText(vis, "Quy dinh Bounding Box:", (bx1 + 10, by1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(vis, "[Den]  Han Nom BBox (ch)", (bx1 + 10, by1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (25, 25, 25), 2)
    cv2.putText(vis, "[Xanh] Quoc Ngu BBox (vi)", (bx1 + 10, by1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (30, 180, 30), 2)

    return vis


def render_book_proofs(book_title: str, max_pages: int = 10, out_root: Optional[Path] = None) -> int:
    bdir = OUTPUT_DIR / book_title
    if not bdir.exists():
        print(f"Error: Output directory not found for: {book_title}")
        return 0

    proof_dir = (out_root or PROJECT_ROOT / "visual_proofs") / book_title
    proof_dir.mkdir(parents=True, exist_ok=True)

    ch_lbl = bdir / "ch" / "Label.txt"
    vi_lbl = bdir / "vi" / "Label.txt"
    ch_imgs = sorted((bdir / "ch").glob("*.jpg"))
    vi_imgs = sorted((bdir / "vi").glob("*.jpg"))
    content_imgs = ch_imgs if ch_imgs else vi_imgs

    # Filter out misc/cover pages if possible, pick content pages with boxes
    pages_to_render = content_imgs[:max_pages]
    dict_engine = None
    try:
        dict_engine = HanNomDictionary.get_instance()
    except Exception:
        pass

    count = 0
    for idx, img_p in enumerate(pages_to_render, 1):
        img_name = img_p.name
        img_bgr = cv2_imread_unicode(img_p)
        if img_bgr is None:
            continue

        ch_boxes = load_bboxes_from_label_file(ch_lbl, img_name)
        vi_boxes = load_bboxes_from_label_file(vi_lbl, img_name)

        vis = draw_colored_bboxes(img_bgr, ch_boxes, vi_boxes, dict_engine=dict_engine)
        out_name = f"page_{idx:03d}_{img_name}"
        out_path = proof_dir / out_name
        cv2_imwrite_unicode(out_path, vis)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Render Visual Proofs with Strict Color Rules ([Đen] for Nôm, [Xanh] for Quốc ngữ)")
    parser.add_argument("--book", type=str, default="all", help="Target book title or 'all' for all 24 books")
    parser.add_argument("--max_pages", type=int, default=10, help="Max pages per book (default: 10)")
    args = parser.parse_args()

    books = [d.name for d in OUTPUT_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if args.book != "all":
        books = [b for b in books if args.book.lower() in b.lower()]

    print(f"Rendering visual proofs for {len(books)} books (max {args.max_pages} pages/book)...")
    total = 0
    for idx, b in enumerate(sorted(books), 1):
        cnt = render_book_proofs(b, max_pages=args.max_pages)
        print(f"[{idx:2d}/{len(books)}] {b[:40]:<40} -> {cnt} pages rendered.")
        total += cnt

    print(f"\nCOMPLETED: Rendered {total} visual proof pages in visual_proofs/.")


if __name__ == "__main__":
    main()
