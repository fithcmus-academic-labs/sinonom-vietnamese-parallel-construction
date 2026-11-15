import sys
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from core.config import OUTPUT_DIR

def main():
    output_dir = OUTPUT_DIR
    books = sorted([d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith("_")])

    print(f"{'Book':<38} | {'ch_b':>5} | {'vi_b':>5} | {'Max_H':>5} | {'>105':>4} | {'Avg_H':>5} | {'Max_IoU':>7}")
    print("-" * 80)

    total_ch = 0
    total_vi = 0
    total_tall = 0
    worst_iou = 0.0

    for b in books:
        b_ch_boxes = 0
        b_vi_boxes = 0
        b_heights = []
        b_max_iou = 0.0

        for folder in ["ch", "vi"]:
            lbl = b / folder / "Label.txt"
            if not lbl.exists():
                continue
            with open(lbl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "\t" not in line:
                        continue
                    rel, js = line.split("\t", 1)
                    boxes = json.loads(js)
                    if folder == "ch":
                        b_ch_boxes += len(boxes)
                    else:
                        b_vi_boxes += len(boxes)

                    polys = []
                    for bx in boxes:
                        pts = bx["points"]
                        ys = [p[1] for p in pts]
                        xs = [p[0] for p in pts]
                        h_val = max(ys) - min(ys)
                        b_heights.append(h_val)
                        polys.append([min(xs), min(ys), max(xs), max(ys)])

                    # Check IoU on page
                    if len(polys) >= 2:
                        arr = np.array(polys, dtype=float)
                        x1 = np.maximum(arr[:, None, 0], arr[None, :, 0])
                        y1 = np.maximum(arr[:, None, 1], arr[None, :, 1])
                        x2 = np.minimum(arr[:, None, 2], arr[None, :, 2])
                        y2 = np.minimum(arr[:, None, 3], arr[None, :, 3])
                        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
                        area = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
                        denom = area[:, None] + area[None, :] - inter
                        iou = np.where(denom > 0, inter / denom, 0.0)
                        np.fill_diagonal(iou, 0)
                        page_max_iou = float(iou.max())
                        if page_max_iou > b_max_iou:
                            b_max_iou = page_max_iou

        tall_count = sum(1 for h in b_heights if h > 105)
        max_h = max(b_heights) if b_heights else 0
        avg_h = np.mean(b_heights) if b_heights else 0
        total_ch += b_ch_boxes
        total_vi += b_vi_boxes
        total_tall += tall_count
        if b_max_iou > worst_iou:
            worst_iou = b_max_iou

        print(f"{b.name[:38]:<38} | {b_ch_boxes:>5} | {b_vi_boxes:>5} | {max_h:>5} | {tall_count:>4} | {avg_h:>5.1f} | {b_max_iou:>7.3f}")

    print("=" * 80)
    print(f"TOTAL BOOKS: {len(books)}")
    print(f"TOTAL CH BOXES: {total_ch}")
    print(f"TOTAL VI BOXES: {total_vi}")
    print(f"TOTAL COMBINED BOXES: {total_ch + total_vi}")
    print(f"TOTAL BOXES > 105px: {total_tall}")
    print(f"WORST IoU ACROSS ALL BOOKS: {worst_iou:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
print("=" * 80)
