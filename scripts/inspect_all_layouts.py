import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from core.preprocess import DocumentPreprocessor, cv2_imread_unicode, cv2_imwrite_unicode

output_dir = PROJECT_ROOT / "output"
books = sorted([d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith("_")])

temp_crops_dir = PROJECT_ROOT / "temp_crops"
temp_crops_dir.mkdir(exist_ok=True)

print(f"{'Book':<38} | {'Lines':>5} | {'L0 RG':>6} | {'L1 RG':>6} | {'L2 RG':>6} | {'L3 RG':>6}")
print("-" * 75)

for b in books:
    # Look for content pages
    all_imgs = sorted((b / "ch").glob("*.jpg"))
    if not all_imgs:
        all_imgs = sorted((b / "vi").glob("*.jpg"))
    if not all_imgs:
        continue

    sample_img = all_imgs[len(all_imgs) // 2]
    img = cv2_imread_unicode(sample_img)
    if img is None:
        continue

    prep = DocumentPreprocessor({"title": b.name, "layout_type": "T2b"}, output_dir)
    # Check side by side or special cases
    fname_lower = b.name.lower()
    if "tế cô hồn" in fname_lower:
        print(f"{b.name[:38]:<38} | Side-by-side table")
        continue
    if "thonomhoxuanhuonghoangxuanhan" in fname_lower:
        print(f"{b.name[:38]:<38} | Top Nom / Bottom Quoc Ngu split")
        continue

    # Get raw refined spans
    h, w, _ = img.shape
    margin_x = prep.detect_margin_x(img)
    right_limit = int(w * 0.975)
    top_cut = int(h * 0.045)
    bot_cut = int(h * 0.96)
    main_area = img[top_cut:bot_cut, margin_x:right_limit]
    ink_mask, signed_rg = prep._build_ink_binary(main_area)

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(h * 0.018))))
    vert_lines = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, vert_kernel)
    clean_ink_mask = cv2.subtract(ink_mask, vert_lines)

    close_w = max(25, int((right_limit - margin_x) * 0.035))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 2))
    closed = cv2.morphologyEx(clean_ink_mask, cv2.MORPH_CLOSE, kernel)
    h_proj = (closed > 0).sum(axis=1)

    min_pixels = max(18, int((right_limit - margin_x) * 0.015))
    raw_spans = []
    in_line = False
    start_y = 0
    for y, val in enumerate(h_proj):
        if val >= min_pixels and not in_line:
            in_line = True
            start_y = y
        elif val < min_pixels and in_line:
            in_line = False
            if (y - start_y) >= 8:
                raw_spans.append((start_y, y))

    # Multi pass split
    max_single_line_h = max(38, int(h * 0.046))
    cur = list(raw_spans)
    for _ in range(3):
        nxt = []
        chg = False
        for sy, ey in cur:
            span_h = ey - sy
            if span_h > max_single_line_h and span_h >= 24:
                mid_s = sy + int(span_h * 0.25)
                mid_e = sy + int(span_h * 0.75)
                if mid_e > mid_s:
                    spl = mid_s + int(np.argmin(h_proj[mid_s:mid_e]))
                    if (spl - sy) >= 12 and (ey - spl) >= 12:
                        nxt.append((sy, spl))
                        nxt.append((spl, ey))
                        chg = True
                        continue
            nxt.append((sy, ey))
        cur = nxt
        if not chg:
            break

    # Calculate RG for first 4 lines
    rgs = []
    for sy, ey in cur[:4]:
        ty1 = sy + top_cut
        ty2 = ey + top_cut
        crop = img[ty1:ty2, margin_x:right_limit]
        g = crop[:, :, 1].astype(float)
        r = crop[:, :, 2].astype(float)
        dark = g < np.percentile(g, 70)
        rg = float((r[dark] - g[dark]).mean()) if dark.any() else 0.0
        rgs.append(f"{rg:6.1f}")

    rgs_str = " | ".join(rgs)
    print(f"{b.name[:38]:<38} | {len(cur):>5} | {rgs_str}")
