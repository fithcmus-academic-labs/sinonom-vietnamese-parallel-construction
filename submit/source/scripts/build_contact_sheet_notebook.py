"""
Builder for contact_sheet.ipynb
Human-QC contact sheets: every page of every book in output/ as a thumbnail with its
4-point bounding boxes (ch = red, vi = blue, misc = gray, no boxes), plus auto QC flags.

Usage:  python scripts/build_contact_sheet_notebook.py
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent

MD_TITLE = """# Contact Sheets for Human QC: Bounding Boxes of All Books

Reads `output/<Book>/{ch,vi,misc}/` and the PPOCRLabel `Label.txt` files (read-only) and produces:

- `output/_contact_sheets/<Book>/sheet_NN.jpg`: paginated grids of page thumbnails with 4-point boxes
  (**red** = `ch/` Han-Nom, **blue** = `vi/` Quoc ngu, **gray frame** = `misc/`, no boxes; **orange frame** = auto-flagged).
- `output/_contact_sheets/qc_flags.csv`: one row per flagged page, with empty `reviewer_verdict` / `notes` columns to fill in.
- `output/_contact_sheets/contact_sheets_index.html`: open in a browser to scroll through every book.

Set `BOOKS` in the config cell to a list of folder names to run a subset."""

C_CONFIG = '''import os, re, json, html, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from IPython.display import display, Image as IPImage

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "output").is_dir() and (PROJECT_ROOT.parent / "output").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
SHEET_DIR = OUTPUT_DIR / "_contact_sheets"
AUDIT_CSV = PROJECT_ROOT / "datasets-midterm" / "audit_dataset_summary.csv"

BOOKS = None            # None = all books, or e.g. ["Ca Trù Thể Cách, Nôm_Y"]
THUMB_W, THUMB_H = 300, 420     # max thumbnail box (aspect ratio kept)
COLS, ROWS = 6, 4               # 24 thumbnails per sheet
CAPTION_H = 22
JPEG_QUALITY = 85
MAX_INLINE_SHEETS = 12          # flagged sheets auto-displayed in the notebook

FOLDERS = ["ch", "vi", "misc"]
COLOR = {"ch": (0, 0, 255), "vi": (255, 0, 0), "misc": (128, 128, 128)}   # BGR
FLAG_COLOR = (0, 140, 255)      # orange, BGR
OUTLIER_SIGMA = 4.0
MIN_OUTLIER_DIFF = 18            # calibrated to accommodate natural poem stanza variations and title pages
TINY_AREA_FRAC = 2e-4           # box area / page area
OVERLAP_IOU = 0.5

SHEET_DIR.mkdir(parents=True, exist_ok=True)
print("project root:", PROJECT_ROOT)'''

C_DISCOVER_MD = "## 1. Discover books and load metadata"

C_DISCOVER = '''PAGE_RE = re.compile(r"\\.(\\d+)\\.jpg$", re.I)

def page_no(name):
    m = PAGE_RE.search(name)
    return int(m.group(1)) if m else -1

audit = pd.read_csv(AUDIT_CSV, encoding="utf-8-sig") if AUDIT_CSV.exists() else pd.DataFrame()
layout_by_stem = {}
if not audit.empty:
    for _, r in audit.iterrows():
        layout_by_stem[Path(str(r["Ten_File"])).stem] = r.get("Ma_Loai_Sach", "?")

all_books = sorted(p.name for p in OUTPUT_DIR.iterdir()
                   if p.is_dir() and not p.name.startswith("_") and (p / "ch").is_dir())
books = [b for b in all_books if BOOKS is None or b in BOOKS]

rows = []
for b in books:
    rows.append({"book": b, "layout": layout_by_stem.get(b, "?"),
                 **{f: len(list((OUTPUT_DIR / b / f).glob("*.jpg"))) for f in FOLDERS}})
catalog = pd.DataFrame(rows)
catalog["total"] = catalog[FOLDERS].sum(axis=1)
display(catalog)
print(len(catalog), "books,", int(catalog["total"].sum()), "page images")'''

C_LOAD_MD = "## 2. Loaders: Label.txt, images, CSV cross-check"

C_LOAD = '''def imread_unicode(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def load_labels(book, folder):
    """{image file name: [ {transcription, points}, ... ]} from <folder>/Label.txt."""
    f = OUTPUT_DIR / book / folder / "Label.txt"
    out = {}
    if not f.exists():
        return out
    for line in f.read_text(encoding="utf-8").splitlines():
        if "\\t" not in line:
            continue
        rel, js = line.split("\\t", 1)
        try:
            out[Path(rel).name] = json.loads(js)
        except json.JSONDecodeError:
            out[Path(rel).name] = None      # corrupt label line
    return out

def load_csv_pages(book):
    f = OUTPUT_DIR / book / f"{book}.csv"
    if not f.exists():
        return set()
    try:
        return set(pd.read_csv(f, encoding="utf-8-sig")["img_id"].dropna().astype(str))
    except Exception:
        return set()'''

C_FLAG_MD = "## 3. QC flag logic (pure functions)"

C_FLAG = '''def box_arrays(boxes):
    return [np.array(b["points"], dtype=float) for b in boxes if b.get("points") and len(b["points"]) == 4]

def max_overlap_iou(polys):
    """Largest axis-aligned IoU between any two boxes on a page."""
    if len(polys) < 2:
        return 0.0
    b = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys])
    x1 = np.maximum(b[:, None, 0], b[None, :, 0]); y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2]); y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou = inter / (area[:, None] + area[None, :] - inter + 1e-9)
    np.fill_diagonal(iou, 0)
    return float(iou.max())

def page_flags(folder, boxes, img_w, img_h):
    """Per-page flags (count outliers are added later, per book)."""
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
    """counts: {(folder, name): n_boxes} -> {(folder, name): flag} for outliers within each folder."""
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
    return res'''

C_RENDER_MD = "## 4. Rendering: thumbnails and sheets"

C_RENDER = '''def render_thumb(args):
    """args = (book, folder, name, boxes). Returns dict with the cell image and page-level metadata."""
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
    """Add frame and caption (needs the final flags, so done after count-outlier detection)."""
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

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ok:
        buf.tofile(str(path))
    return ok'''

C_RUN_MD = "## 5. Run over all books"

C_RUN = '''ORDER = {f: i for i, f in enumerate(FOLDERS)}
flag_rows, book_stats, sheet_index = [], [], {}

def process_book(book):
    labels = {f: load_labels(book, f) for f in FOLDERS}
    jobs = []
    for f in FOLDERS:
        for p in sorted((OUTPUT_DIR / book / f).glob("*.jpg")):
            jobs.append((book, f, p.name, labels[f].get(p.name, [])))
    jobs.sort(key=lambda j: (page_no(j[2]), ORDER[j[1]]))
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        thumbs = list(ex.map(render_thumb, jobs))

    # book-level checks
    outl = count_outliers({(t["folder"], t["name"]): t["n"] for t in thumbs})
    csv_pages = load_csv_pages(book)
    present = {t["name"] for t in thumbs}
    for t in thumbs:
        k = (t["folder"], t["name"])
        if k in outl:
            t["flags"].append(outl[k])
    missing_csv = sorted(csv_pages - present)

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
    for n in missing_csv:
        frows.append({"book": book, "page": page_no(n), "folder": "-", "image": n, "n_boxes": 0,
                      "flag": "in_csv_but_no_image", "reviewer_verdict": "", "notes": ""})
    stats = {"book": book, "pages": len(thumbs), "boxes": sum(t["n"] for t in thumbs),
             "sheets": len(sheets), "flagged": len(frows)}
    return stats, frows, sheets

for i, b in enumerate(books, 1):
    s, fr, sh = process_book(b)
    book_stats.append(s); flag_rows += fr; sheet_index[b] = sh
    print(f"[{i:2d}/{len(books)}] {b}: {s['pages']} pages, {s['boxes']} boxes, {s['sheets']} sheets, {s['flagged']} flagged")

flags_df = pd.DataFrame(flag_rows, columns=["book", "page", "folder", "image", "n_boxes", "flag", "reviewer_verdict", "notes"])
flags_df.to_csv(SHEET_DIR / "qc_flags.csv", index=False, encoding="utf-8-sig")
stats_df = pd.DataFrame(book_stats)
print("done:", int(stats_df["pages"].sum()), "pages,", int(stats_df["sheets"].sum()), "sheets,", len(flags_df), "flags")'''

C_REVIEW_MD = "## 6. Review: summary, flag table, and viewers"

C_REVIEW = '''display(stats_df.merge(catalog[["book", "layout"]], on="book"))
print("\\nFlags by type:")
if len(flags_df):
    display(flags_df["flag"].str.replace(r"\\(.*?\\)", "", regex=True).str.split("; ").explode().value_counts().to_frame("pages"))
    display(flags_df.head(40))
else:
    print("no flags")

def show_book(book, only_flagged=False, limit=None):
    """Display a book's contact sheets inline."""
    shown = 0
    for s in sheet_index[book]:
        if only_flagged and not s["n_flagged"]:
            continue
        print(f"{book} / {s['file']}  (pages {s['first']}-{s['last']}, {s['n_flagged']} flagged)")
        display(IPImage(filename=str(SHEET_DIR / book / s["file"])))
        shown += 1
        if limit and shown >= limit:
            break

def show_page(book, folder, page, width=1100):
    """Full-resolution check of one page with boxes and transcriptions."""
    name = next((p.name for p in (OUTPUT_DIR / book / folder).glob("*.jpg") if page_no(p.name) == page), None)
    if name is None:
        print("page not found"); return
    img = imread_unicode(OUTPUT_DIR / book / folder / name)
    boxes = load_labels(book, folder).get(name) or []
    for p in box_arrays(boxes):
        cv2.polylines(img, [p.astype(np.int32).reshape(-1, 1, 2)], True, COLOR[folder], max(2, img.shape[1] // 500))
    s = width / img.shape[1]
    img = cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    display(IPImage(data=cv2.imencode(".jpg", img)[1].tobytes()))
    for b in boxes[:15]:
        print(b.get("transcription"))

# Auto-show the first flagged sheets (limited, to keep the notebook small)
budget = MAX_INLINE_SHEETS
for b in books:
    if budget <= 0:
        break
    n = sum(1 for s in sheet_index[b] if s["n_flagged"])
    if n:
        show_book(b, only_flagged=True, limit=budget)
        budget -= min(n, budget)'''

C_HTML_MD = "## 7. HTML index for browser review"

C_HTML = '''parts = ["<!doctype html><meta charset='utf-8'><title>Contact sheets</title>",
         "<style>body{font-family:sans-serif;margin:16px}img{max-width:100%;border:1px solid #ccc;margin:4px 0}"
         ".bad{color:#d2691e;font-weight:bold}</style><h1>Contact sheets for human QC</h1>",
         "<p><b style='color:red'>red</b> = ch boxes, <b style='color:blue'>blue</b> = vi boxes, gray frame = misc, "
         "<b class='bad'>orange frame = flagged</b>. Flag list: <a href='qc_flags.csv'>qc_flags.csv</a></p><ul>"]
for s in book_stats:
    parts.append(f"<li><a href='#{html.escape(s['book'], quote=True)}'>{html.escape(s['book'])}</a>: "
                 f"{s['pages']} pages, {s['flagged']} flagged</li>")
parts.append("</ul>")
for b in books:
    parts.append(f"<h2 id='{html.escape(b, quote=True)}'>{html.escape(b)}</h2>")
    for s in sheet_index[b]:
        tag = f" <span class='bad'>({s['n_flagged']} flagged)</span>" if s["n_flagged"] else ""
        src = html.escape(f"{b}/{s['file']}".replace(" ", "%20"), quote=True)
        parts.append(f"<div>pages {s['first']}-{s['last']}{tag}<br><img loading='lazy' src='{src}'></div>")
(SHEET_DIR / "contact_sheets_index.html").write_text("\\n".join(parts), encoding="utf-8")
print("wrote", SHEET_DIR / "contact_sheets_index.html")'''


def build():
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [md(MD_TITLE), code(C_CONFIG),
                md(C_DISCOVER_MD), code(C_DISCOVER),
                md(C_LOAD_MD), code(C_LOAD),
                md(C_FLAG_MD), code(C_FLAG),
                md(C_RENDER_MD), code(C_RENDER),
                md(C_RUN_MD), code(C_RUN),
                md(C_REVIEW_MD), code(C_REVIEW),
                md(C_HTML_MD), code(C_HTML)]
    out = ROOT / "contact_sheet.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    build()
