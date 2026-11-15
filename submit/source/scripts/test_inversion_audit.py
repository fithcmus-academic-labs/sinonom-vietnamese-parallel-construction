import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from core.preprocess import DocumentPreprocessor, cv2_imread_unicode

output_dir = PROJECT_ROOT / "output"
books = sorted([d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith("_")])

print(f"{'Book':<40} | {'ch_rg':>6} | {'vi_rg':>6} | {'diff':>6} | {'Status'}")
print("-" * 75)

for b in books:
    ch_imgs = sorted((b / "ch").glob("*.jpg"))
    if not ch_imgs:
        continue
    # Pick a content page in the middle
    sample_img = ch_imgs[len(ch_imgs) // 2]
    img = cv2_imread_unicode(sample_img)
    if img is None:
        continue
    prep = DocumentPreprocessor({"title": b.name, "layout_type": "T2b"}, output_dir)
    res = prep.segment_t2b_lines(sample_img)
    ch_len = len(res["ch"])
    vi_len = len(res["vi"])
    if ch_len == 0 or vi_len == 0:
        print(f"{b.name[:40]:<40} | Single folder (ch: {ch_len}, vi: {vi_len})")
        continue

    def get_rg(boxes):
        vals = []
        for box in boxes:
            pts = box["points"]
            crop = img[pts[0][1]:pts[2][1], pts[0][0]:pts[1][0]]
            if crop.size == 0:
                continue
            g = crop[:, :, 1].astype(float)
            r = crop[:, :, 2].astype(float)
            dark = g < np.percentile(g, 70)
            if dark.any():
                vals.append(float((r[dark] - g[dark]).mean()))
        return float(np.mean(vals)) if vals else 0.0

    ch_rg = get_rg(res["ch"])
    vi_rg = get_rg(res["vi"])
    diff = vi_rg - ch_rg
    status = "OK" if diff > 1.5 else ("INVERTED" if diff < -1.5 else "AMBIGUOUS")
    print(f"{b.name[:40]:<40} | {ch_rg:6.1f} | {vi_rg:6.1f} | {diff:6.1f} | {status}")
