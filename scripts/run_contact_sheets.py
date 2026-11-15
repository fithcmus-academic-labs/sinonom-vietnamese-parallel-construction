import os
import re
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
SHEET_DIR = OUTPUT_DIR / "_contact_sheets"
AUDIT_CSV = PROJECT_ROOT / "datasets-midterm" / "audit_dataset_summary.csv"

THUMB_W, THUMB_H = 300, 420
COLS, ROWS = 6, 4
CAPTION_H = 22
JPEG_QUALITY = 85

FOLDERS = ["ch", "vi", "misc"]
COLOR = {"ch": (0, 0, 255), "vi": (255, 0, 0), "misc": (128, 128, 128)}
FLAG_COLOR = (0, 140, 255)
OUTLIER_SIGMA = 4.0
MIN_OUTLIER_DIFF = 18
TINY_AREA_FRAC = 2e-4
OVERLAP_IOU = 0.5

SHEET_DIR.mkdir(parents=True, exist_ok=True)
PAGE_RE = re.compile(r"\.(\d+)\.jpg$", re.I)

def page_no(name):
    m = PAGE_RE.search(name)
    return int(m.group(1)) if m else -1

def imread_unicode(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ok:
        buf.tofile(str(path))
    return ok

def load_labels(book, folder):
    f = OUTPUT_DIR / book / folder / "Label.txt"
    out = {}
    if not f.exists():
        return out
    for line in f.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        rel, js = line.split("\t", 1)
        try:
            out[Path(rel).name] = json.loads(js)
        except json.JSONDecodeError:
            out[Path(rel).name] = None
    return out

def load_csv_pages(book):
    f = OUTPUT_DIR / book / f"{book}.csv"
    if not f.exists():
        f = PROJECT_ROOT / "datasets-midterm" / "csv" / f"{book}.csv"
    if not f.exists():
        return set()
    try:
        df = pd.read_csv(f, encoding="utf-8-sig")
        return set(df["img_id"].dropna().astype(str))
    except Exception:
        return set()

def box_arrays(boxes):
    return [np.array(b["points"], dtype=float) for b in boxes if b.get("points") and len(b["points"]) == 4]

def max_overlap_iou(polys):
    if len(polys) < 2:
        return 0.0
    b = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys])
    x1 = np.maximum(b[:, None, 0], b[None, :, 0])
    y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2])
    y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    denom = area[:, None] + area[None, :] - inter
    iou = np.where(denom > 0, inter / denom, 0.0)
    np.fill_diagonal(iou, 0)
    return float(iou.max())

def page_flags(folder, boxes, img_w, img_h):
    flags = []
    if folder == "misc":
        return flags
    if boxes is None:
        return ["corrupt_label"]
    if len(boxes) == 0:
        return ["no_boxes"]
    polys = box_arrays(boxes)
    if len(polys) < len(boxes):
        flags.append("malformed_box")
    oob = sum(1 for p in polys if p[:, 0].min() < -2 or p[:, 1].min() < -2
              or p[:, 0].max() > img_w + 2 or p[:, 1].max() > img_h + 2)
    if oob:
        flags.append(f"out_of_bounds({oob})")
    tiny = sum(1 for p in polys
               if (p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min()) < TINY_AREA_FRAC * img_w * img_h)
    if tiny:
        flags.append(f"tiny_box({tiny})")
    if max_overlap_iou(polys) > OVERLAP_IOU:
        flags.append("overlap")
    return flags

def count_outliers(counts):
    res = {}
    for folder in ("ch", "vi"):
        items = [(k, n) for k, n in counts.items() if k[0] == folder]
        if len(items) < 5:
            continue
        ns = np.array([n for _, n in items], dtype=float)
        med, sd = np.median(ns), ns.std()
        if sd == 0:
            continue
        for k, n in items:
            if abs(n - med) > OUTLIER_SIGMA * sd and abs(n - med) >= MIN_OUTLIER_DIFF:
                res[k] = f"box_count_outlier({int(n)} vs median {int(med)})"
    return res

def render_thumb(args):
    book, folder, name, boxes = args
    img = imread_unicode(OUTPUT_DIR / book / folder / name)
    if img is None:
        cell = np.full((THUMB_H, THUMB_W, 3), 255, np.uint8)
        cv2.putText(cell, "UNREADABLE", (20, THUMB_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return {"book": book, "folder": folder, "name": name, "n": 0, "w": 0, "h": 0,
                "flags": ["unreadable_image"], "cell": cell}
    h, w = img.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
    thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    polys = box_arrays(boxes) if boxes else []
    for p in polys:
        pts = np.round(p * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(thumb, [pts], True, COLOR[folder], 1, cv2.LINE_AA)
    cell = np.full((THUMB_H, THUMB_W, 3), 255, np.uint8)
    cell[:th, :tw] = thumb
    return {"book": book, "folder": folder, "name": name, "n": len(polys), "w": w, "h": h,
            "flags": page_flags(folder, boxes, w, h), "cell": cell}

def finish_cell(t):
    flagged = bool(t["flags"])
    color = FLAG_COLOR if flagged else COLOR[t["folder"]]
    cell = cv2.copyMakeBorder(t["cell"], CAPTION_H, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, cell.shape[0] - 1), color, 4 if flagged else 2)
    label = f"p{page_no(t['name']):03d} {t['folder']}" + (f" n={t['n']}" if t["folder"] != "misc" else "")
    if flagged:
        label += " !"
    cv2.putText(cell, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return cell

def build_sheet(cells):
    ch, cw = cells[0].shape[:2]
    blank = np.full((ch, cw, 3), 235, np.uint8)
    cells = cells + [blank] * (COLS * ROWS - len(cells))
    rows = [np.hstack(cells[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
    return np.vstack(rows)

def process_book(book):
    labels = {f: load_labels(book, f) for f in FOLDERS}
    jobs = []
    ORDER = {f: i for i, f in enumerate(FOLDERS)}
    for f in FOLDERS:
        for p in sorted((OUTPUT_DIR / book / f).glob("*.jpg")):
            jobs.append((book, f, p.name, labels[f].get(p.name, [])))
    jobs.sort(key=lambda j: (page_no(j[2]), ORDER[j[1]]))
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        thumbs = list(ex.map(render_thumb, jobs))

    outl = count_outliers({(t["folder"], t["name"]): t["n"] for t in thumbs})
    csv_pages = load_csv_pages(book)
    present = {t["name"] for t in thumbs}
    for t in thumbs:
        k = (t["folder"], t["name"])
        if k in outl:
            t["flags"].append(outl[k])

    sheets, out_dir = [], SHEET_DIR / book
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sheet_*.jpg"):
        old.unlink()
    per = COLS * ROWS
    for i in range(0, len(thumbs), per):
        chunk = thumbs[i:i + per]
        fn = f"sheet_{i // per + 1:02d}.jpg"
        imwrite_unicode(out_dir / fn, build_sheet([finish_cell(t) for t in chunk]))
        sheets.append({"file": fn, "n_flagged": sum(bool(t["flags"]) for t in chunk),
                       "first": page_no(chunk[0]["name"]), "last": page_no(chunk[-1]["name"])})

    frows = [{"book": book, "page": page_no(t["name"]), "folder": t["folder"], "image": t["name"],
              "n_boxes": t["n"], "flag": "; ".join(t["flags"]), "reviewer_verdict": "", "notes": ""}
             for t in thumbs if t["flags"]]

    stats = {"book": book, "pages": len(thumbs), "boxes": sum(t["n"] for t in thumbs),
             "sheets": len(sheets), "flagged": len(frows)}
    return stats, frows, sheets

def main():
    books = sorted(p.name for p in OUTPUT_DIR.iterdir()
                   if p.is_dir() and not p.name.startswith("_") and ((p / "ch").is_dir() or (p / "vi").is_dir()))
    print(f"Generating contact sheets for {len(books)} books...")
    flag_rows = []
    book_stats = []

    for i, b in enumerate(books, 1):
        s, fr, _ = process_book(b)
        book_stats.append(s)
        flag_rows += fr
        print(f"[{i:2d}/{len(books)}] {b[:40]:<40} | Pages: {s['pages']:>4} | Sheets: {s['sheets']:>2} | Flagged: {s['flagged']:>2}")

    flags_df = pd.DataFrame(flag_rows, columns=["book", "page", "folder", "image", "n_boxes", "flag", "reviewer_verdict", "notes"])
    flags_df.to_csv(SHEET_DIR / "qc_flags.csv", index=False, encoding="utf-8-sig")
    print("=" * 80)
    print(f"TOTAL FLAGGED PAGES IN qc_flags.csv: {len(flags_df)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
