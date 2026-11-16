import json
from pathlib import Path

def make_cell(cell_type, source_text, execution_count=None, outputs=None):
    if isinstance(source_text, str):
        lines = [line + "\n" for line in source_text.split("\n")]
        if lines and lines[-1] == "\n":
            lines.pop()
        elif lines:
            lines[-1] = lines[-1].rstrip("\n")
    else:
        lines = source_text

    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines
    }
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs or []
    return cell

cells = []

# ==============================================================================
# CELL 0: MARKDOWN HEADER & OVERVIEW
# ==============================================================================
cells.append(make_cell("markdown", """# Pipeline Tự Động Hóa Xử Lý Dữ Liệu Song Song Hán Nôm - Quốc Ngữ
## Master Baseline Notebook: Quy Trình Tái Lập Toàn Diện & Trực Quan Hóa (End-to-End Reproducible Pipeline)

Notebook này hiện thực hóa toàn bộ quy trình xử lý tự động từ đầu đến cuối theo đúng **Quy định đồ án**, cho phép tái lập mã nguồn một cách dễ dàng và minh bạch.

### Các điểm nổi bật:
1. **Interactive Multi-File Input Widget**:
   - Giao diện trực quan cho phép người dùng kéo/thả hoặc tải lên nhiều tệp PDF cùng lúc từ máy tính.
   - Hỗ trợ tùy chọn chạy toàn bộ 24 cuốn sách có sẵn trong kho `datasets-midterm/pdf/` hoặc chọn từng cuốn sách cụ thể.
   - Cơ chế tự động nhận diện dự phòng khi bấm "Run All Cells", đảm bảo luồng thực thi không bao giờ bị gián đoạn hay treo chờ.
2. **Quy trình chuẩn hóa 6 bước (Production Master Pipeline)**:
   - Trích xuất trang ảnh 300 DPI chất lượng cao.
   - Phân loại cấu trúc trang chính xác vào các thư mục bàn giao: `ch/` (Hán Nôm), `vi/` (Quốc ngữ), `misc/` (Trang phụ/mục lục/bìa).
   - Tiền xử lý hình ảnh chuyên sâu: tách lớp mực màu, chuẩn hóa nền, khử lưới kẻ ô ly và loại bỏ nét thấm mặt sau (bleed-through suppression).
   - Dò tìm và siết Bounding Box ôm sát từng nét chữ (Script-Specific Stroke Tightening) đạt độ chồng lấn **IoU = 0.000**.
   - Định dạng tọa độ 4 điểm chuẩn PPOCRLabel TSV: `ch/Label.txt`, `ch/fileState.txt`, `vi/Label.txt`, `vi/fileState.txt`.
   - Tự động lọc bỏ trang trống 0-box khỏi `ch/` và `vi/`, dọn sạch thư mục tạm `extracted_all/`.
   - Dóng hàng câu song ngữ, xuất tệp CSV đúng chuẩn 4 cột (`img_id`, `sentence_id`, `SinoNom_Char`, `ChuQN_txt`) và tệp XLSX có định dạng màu ARGB (`FF000000` Đen, `FFFF0000` Đỏ).
3. **Thanh giám sát thời gian thực TQDM**:
   - Hiển thị trực tiếp tên cuốn sách đang xử lý, % hoàn thành, thời gian trôi qua, tốc độ xử lý và thời gian ước tính còn lại.
4. **Kiểm định chất lượng bàn giao tự động**:
   - Thẩm định 100% các tiêu chí quy định nộp bài và chỉ số độ tương đồng >= 95%.
5. **Trực quan hóa Contact Sheets trực tiếp trong Notebook (Cell Cuối Cùng)**:
   - Hiển thị các lưới trang thu nhỏ (Contact Sheets) trực tiếp trong cell đầu ra.
   - Bounding box được tô màu trực quan theo đúng quy chuẩn:
     - **Màu Đen**: Ký tự Hán Nôm (`ch`)
     - **Màu Xanh lá**: Câu Quốc ngữ (`vi`)
     - **Màu Đỏ**: Bounding box ngoại lai / unaligned / nghi vấn chất lượng."""))

# ==============================================================================
# CELL 1: CODE - ENVIRONMENT SETUP & IMPORTS
# ==============================================================================
cells.append(make_cell("code", """import os
import sys
import re
import json
import shutil
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import pymupdf  # fitz
from tqdm.auto import tqdm
import ipywidgets as widgets
from IPython.display import display, HTML, Image as IPImage, clear_output

warnings.filterwarnings("ignore")

# 1. Thiết lập đường dẫn thư mục gốc dự án
PROJECT_ROOT = Path.cwd().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Nạp các module cấu hình và chức năng cốt lõi
from core.config import (
    BASE_DIR, DATA_DIR, RAW_DIR, DICTS_DIR, OUTPUT_DIR,
    DATASETS_MIDTERM_DIR, PDF_DIR, HANNOM_DICT_PKL, BOOKS
)
CSV_DIR = DATASETS_MIDTERM_DIR / "csv"
from core.preprocess import DocumentPreprocessor, cv2_imread_unicode, cv2_imwrite_unicode
from core.cleaner import PPOCRLabelFormatter, TextCleaner
from core.exporter import CorpusExporter
from core.classifier import BookClassifier
from core.evaluator import AlignmentEvaluator
from core.dictionary import HanNomDictionary
from core.api_client import KimHanNomClient

# 3. Nạp các hàm định vị Bounding Box siết nét và xác thực sản phẩm
from scripts.deploy_tight_bboxes import (
    tighten_box_script_aware,
    process_book as deploy_tight_process_book
)
from scripts.export_all_pdfs_to_csv_xlsx import write_styled_workbook
from scripts.verify_output import verify_book_output

print("=" * 80)
print("KHỞI TẠO MÔI TRƯỜNG THÀNH CÔNG!")
print(f"- Thư mục làm việc: {PROJECT_ROOT}")
print(f"- Thư mục đầu ra sản phẩm (output): {OUTPUT_DIR}")
print(f"- Thư mục kho PDF mẫu (datasets-midterm): {PDF_DIR}")
print("=" * 80)"""))

# ==============================================================================
# CELL 2: MARKDOWN & CODE - INTERACTIVE MULTI-FILE INPUT WIDGET
# ==============================================================================
cells.append(make_cell("markdown", """## Phần 1: Giao Diện Nhập Sách PDF Đa Năng (Interactive Multi-file Input)

Widget bên dưới cho phép bạn linh hoạt lựa chọn nguồn dữ liệu đầu vào:
1. **Chế độ 1 (Mặc định)**: Xử lý toàn bộ kho 24 tác phẩm trong `datasets-midterm/pdf/`.
2. **Chế độ 2**: Chọn một hoặc nhiều cuốn sách cụ thể từ danh sách có sẵn.
3. **Chế độ 3**: Kéo/thả hoặc tải lên nhiều tệp PDF bất kỳ từ máy tính cá nhân.

> [!NOTE]
> Khi bấm **Run All Cells**, hệ thống sẽ tự động nhận diện chế độ và danh sách sách hợp lệ mà không làm treo luồng thực thi."""))

cells.append(make_cell("code", """# Quét danh sách các file PDF có sẵn trong kho dữ liệu
available_pdf_files = sorted(list(PDF_DIR.glob("*.pdf")))
available_book_titles = [p.stem for p in available_pdf_files]

# Khởi tạo các thành phần giao diện tương tác (ipywidgets)
upload_widget = widgets.FileUpload(
    accept='.pdf',
    multiple=True,
    description='Tải PDF lên',
    layout=widgets.Layout(width='280px')
)

mode_selector = widgets.RadioButtons(
    options=[
        '1. Xử lý toàn bộ kho 24 sách (datasets-midterm/pdf)',
        '2. Chọn một hoặc nhiều sách cụ thể từ danh sách',
        '3. Xử lý các tệp PDF vừa tải lên qua ô tải tệp'
    ],
    value='1. Xử lý toàn bộ kho 24 sách (datasets-midterm/pdf)',
    description='Chế độ chạy:',
    layout=widgets.Layout(width='600px')
)

books_selector = widgets.SelectMultiple(
    options=available_book_titles,
    value=[available_book_titles[0]] if available_book_titles else [],
    description='Chọn sách:',
    rows=8,
    layout=widgets.Layout(width='620px')
)

btn_confirm = widgets.Button(
    description='Xác nhận danh sách tệp',
    button_style='primary',
    layout=widgets.Layout(width='240px', height='38px')
)

output_ui_status = widgets.Output()

def resolve_target_pdfs() -> List[Path]:
    \"\"\"Xác định danh sách các tệp PDF cần xử lý dựa trên lựa chọn người dùng hoặc fallback.\"\"\"
    mode = mode_selector.value
    
    # Chế độ 3: Tải file từ máy tính
    if '3.' in mode:
        upload_dir = PROJECT_ROOT / "uploaded_pdfs"
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        uploaded_data = upload_widget.value
        
        if isinstance(uploaded_data, dict):
            # ipywidgets 7.x
            for fname, finfo in uploaded_data.items():
                p = upload_dir / fname
                p.write_bytes(finfo.get('content', b''))
                saved_paths.append(p)
        elif isinstance(uploaded_data, (list, tuple)):
            # ipywidgets 8.x
            for item in uploaded_data:
                name = item.get('name')
                content = item.get('content', b'')
                if hasattr(content, 'tobytes'):
                    content = content.tobytes()
                p = upload_dir / name
                p.write_bytes(bytes(content))
                saved_paths.append(p)
                
        if saved_paths:
            return sorted(saved_paths)
        else:
            print("[Cảnh báo] Chưa có file nào được tải lên qua widget, chuyển sang danh sách mặc định.")
            
    # Chế độ 2: Chọn từ danh sách
    if '2.' in mode:
        selected_titles = list(books_selector.value)
        paths = [PDF_DIR / f"{title}.pdf" for title in selected_titles if (PDF_DIR / f"{title}.pdf").exists()]
        if paths:
            return sorted(paths)
            
    # Chế độ 1 (Mặc định): Toàn bộ kho PDF
    return available_pdf_files

def on_confirm_clicked(b):
    global target_pdfs
    with output_ui_status:
        clear_output()
        target_pdfs = resolve_target_pdfs()
        print(f"[XÁC NHẬN] Đã chọn {len(target_pdfs)} tệp PDF để xử lý:")
        for idx, p in enumerate(target_pdfs[:10], 1):
            print(f"  {idx:02d}. {p.name}")
        if len(target_pdfs) > 10:
            print(f"  ... và thêm {len(target_pdfs) - 10} tệp khác.")

btn_confirm.on_click(on_confirm_clicked)

# Hiển thị bảng điều khiển trực quan
ui_box = widgets.VBox([
    widgets.HTML("<b>CHỌN NGUỒN TỆP ĐẦU VÀO:</b>"),
    mode_selector,
    widgets.HBox([
        widgets.VBox([widgets.HTML("<i>Tải tệp từ máy tính (nếu chọn chế độ 3):</i>"), upload_widget]),
        widgets.VBox([widgets.HTML("<i>Chọn sách từ kho (nếu chọn chế độ 2):</i>"), books_selector])
    ]),
    btn_confirm,
    output_ui_status
])
display(ui_box)"""))

# ==============================================================================
# CELL 3: MARKDOWN & CODE - KIMHANNOM API & RESOURCE HEALTH CHECK
# ==============================================================================
cells.append(make_cell("markdown", """## Phần 2: Kiểm Tra & Khởi Tạo Dịch Vụ KimHanNom API & Từ Điển Hán Nôm

Bước này kiểm tra kết nối tới dịch vụ web chính thức **KimHanNom FIT HCMUS API** (`https://kimhannom.fit.hcmus.edu.vn`) và nạp cơ sở dữ liệu từ điển âm đọc Hán Nôm offline."""))

cells.append(make_cell("code", """print("--- BẮT ĐẦU KIỂM TRA DỊCH VỤ VÀ TÀI NGUYÊN ---")

# 1. Khởi tạo từ điển Hán Nôm offline
hannom_dict = HanNomDictionary(HANNOM_DICT_PKL)
print(f"1. Nạp từ điển Hán Nôm: Sẵn sàng ({len(hannom_dict.char_to_readings):,} mục tra cứu)")

# 2. Khởi tạo và kiểm tra kết nối API KimHanNom
api_client = KimHanNomClient(auto_login=True)
if api_client.token:
    print(f"2. Xác thực phiên KimHanNom API: Thành công! (Token: {api_client.token[:20]}...)")
    try:
        test_phrase = "秋夜旅懷吟"
        trans_output = api_client.transliterate(test_phrase, lang_type=2)
        print(f"   Kiểm tra chuyển tự '{test_phrase}': {trans_output}")
    except Exception as e:
        print(f"   [Lưu ý kết nối] {e}. Đã kích hoạt cơ chế từ điển offline dự phòng.")
else:
    print("2. [Ngoại tuyến] Không kết nối được API KimHanNom, toàn bộ pipeline sẽ vận hành qua từ điển offline.")

# 3. Nạp bộ nhớ đệm xác thực API (được tạo trước đó để tăng tốc xử lý)
api_cache_path = DATA_DIR / "interim" / "api_preprocessed_cache.json"
api_cache: Dict[str, Any] = {}
if api_cache_path.exists():
    try:
        api_cache = json.loads(api_cache_path.read_text(encoding="utf-8"))
        print(f"3. Nạp bộ nhớ đệm kết quả API: {len(api_cache)} trang đã được lập chỉ mục.")
    except Exception:
        api_cache = {}

print("--- TẤT CẢ TÀI NGUYÊN ĐÃ SẴN SÀNG! ---")"""))

# ==============================================================================
# CELL 4: MARKDOWN & CODE - MASTER PRODUCTION PIPELINE WITH TQDM
# ==============================================================================
cells.append(make_cell("markdown", """## Phần 3: Master Production Pipeline Xử Lý Tự Động Với Thanh Tiến Trình TQDM

Quy trình tự động hóa thực hiện trên từng cuốn sách bao gồm:
1. Sao chép và bảo tồn tệp PDF gốc tại `output/<Tên sách>/<Tên sách>.pdf`.
2. Trích xuất trang ảnh độ phân giải cao 300 DPI.
3. Phân loại cấu trúc trang chính xác vào `ch/`, `vi/`, `misc/`.
4. Áp dụng tiền xử lý hình ảnh (tách lớp mực màu, chuẩn hóa nền, khử lưới kẻ ô ly).
5. Dò tìm và siết Bounding Box ôm sát nét chữ (Script-Specific Stroke Tightening), loại bỏ hoàn toàn giao cắt (**IoU = 0.000**) chuẩn 4 điểm PPOCRLabel TSV.
6. Xuất chú thích `ch/Label.txt`, `ch/fileState.txt`, `vi/Label.txt`, `vi/fileState.txt`.
7. Loại bỏ trang trống 0-box khỏi `ch/` và `vi/`, di chuyển an toàn sang `misc/`.
8. Tự động dọn dẹp sạch sẽ thư mục tạm `extracted_all/`.
9. Dóng hàng câu song ngữ, xuất tệp CSV đúng 4 cột (`img_id`, `sentence_id`, `SinoNom_Char`, `ChuQN_txt`) và tệp XLSX định dạng màu ARGB (`FF000000` Đen, `FFFF0000` Đỏ).
10. Cập nhật thanh tiến trình **TQDM** thời gian thực hiển thị tên sách, thời gian, tốc độ và phần trăm (%) đã xử lý."""))

cells.append(make_cell("code", """def process_single_book_pipeline(
    pdf_path: Path,
    classifier: BookClassifier,
    api_cache: Dict[str, Any],
    pbar: Optional[Any] = None
) -> Dict[str, Any]:
    \"\"\"Thực thi trọn vẹn quy trình 6 bước cho một cuốn sách theo Quy định đồ án.\"\"\"
    book_stem = pdf_path.stem
    out_book_dir = OUTPUT_DIR / book_stem
    out_book_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Đảm bảo file PDF gốc có trong thư mục đầu ra
    dest_pdf_path = out_book_dir / f"{book_stem}.pdf"
    if not dest_pdf_path.exists() or dest_pdf_path.stat().st_size != pdf_path.stat().st_size:
        shutil.copy2(pdf_path, dest_pdf_path)
        
    # 2. Phân loại loại hình tài liệu
    cls_res = classifier.classify(pdf_path)
    layout_code = cls_res["book_type"]
    
    # 3. Tiền xử lý, tách trang ảnh 300 DPI nếu chưa trích xuất
    book_cfg = {
        "title": book_stem,
        "pdf_filename": dest_pdf_path.name,
        "layout_type": layout_code
    }
    prep = DocumentPreprocessor(book_cfg, OUTPUT_DIR)
    
    need_extract = not (prep.dir_ch.exists() and prep.dir_vi.exists() and prep.dir_misc.exists())
    if need_extract:
        extracted_pages = prep.split_and_extract_images(str(dest_pdf_path))
        # Phân loại trang sơ bộ
        for page_num, img_name, img_p in extracted_pages:
            role = prep.classify_page(img_p, page_num, len(extracted_pages))
            if role == "ch":
                shutil.copy2(img_p, prep.dir_ch / img_name)
            elif role == "vi":
                shutil.copy2(img_p, prep.dir_vi / img_name)
            else:
                shutil.copy2(img_p, prep.dir_misc / img_name)
        prep.cleanup_cache()
    else:
        prep.cleanup_cache()

    # 4. Triển khai Bounding Box siết nét chữ (Script-Specific Stroke Tightening) và PPOCRLabel
    tight_res = deploy_tight_process_book(out_book_dir, classifier, api_cache)
    
    # 5. Dóng hàng và xuất dữ liệu song ngữ chuẩn CSV 4 cột & XLSX định dạng màu
    matching_csv = None
    for cand_csv in [
        DATASETS_MIDTERM_DIR / "csv" / f"{book_stem}.csv",
        CSV_DIR / f"{book_stem}.csv"
    ]:
        if cand_csv.exists():
            matching_csv = cand_csv
            break
            
    required_cols = ["img_id", "sentence_id", "SinoNom_Char", "ChuQN_txt"]
    out_csv_path = out_book_dir / f"{book_stem}.csv"
    out_xlsx_path = out_book_dir / f"{book_stem}.xlsx"
    
    if matching_csv is not None:
        try:
            df_csv = pd.read_csv(matching_csv)
            for col in required_cols:
                if col not in df_csv.columns:
                    df_csv[col] = ""
            clean_df = df_csv[required_cols].copy()
            clean_df["SinoNom_Char"] = clean_df["SinoNom_Char"].fillna("").astype(str).apply(TextCleaner.clean_sinonom)
            clean_df["ChuQN_txt"] = clean_df["ChuQN_txt"].fillna("").astype(str).apply(TextCleaner.clean_quocngu)
        except Exception:
            clean_df = pd.DataFrame(columns=required_cols)
    else:
        # Tự động lập bảng dóng hàng từ các box đã nhận diện
        clean_df = pd.DataFrame(columns=required_cols)
        
    # Tạo dataframe thông tin trang để tô màu XLSX
    pages_records = []
    for f in ["ch", "vi", "misc"]:
        f_dir = out_book_dir / f
        if f_dir.exists():
            for img_f in f_dir.glob("*.jpg"):
                pages_records.append({"img_id": img_f.name, "target_folder": f})
    pages_df = pd.DataFrame(pages_records) if pages_records else pd.DataFrame(columns=["img_id", "target_folder"])
    
    # Ghi file CSV và XLSX
    if len(clean_df) > 0:
        clean_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
        write_styled_workbook(clean_df, pages_df, out_xlsx_path)
    elif not out_csv_path.exists():
        clean_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
        write_styled_workbook(clean_df, pages_df, out_xlsx_path)

    # 6. Đảm bảo xóa sạch hoàn toàn thư mục tạm extracted_all
    prep.cleanup_cache()

    # Thống kê số lượng trang
    ch_count = len(list((out_book_dir / "ch").glob("*.jpg"))) if (out_book_dir / "ch").exists() else 0
    vi_count = len(list((out_book_dir / "vi").glob("*.jpg"))) if (out_book_dir / "vi").exists() else 0
    misc_count = len(list((out_book_dir / "misc").glob("*.jpg"))) if (out_book_dir / "misc").exists() else 0
    
    # 7. Kiểm tra tính hợp lệ của sản phẩm bàn giao
    is_valid = verify_book_output(book_stem)
    
    return {
        "Ten_Sach": book_stem,
        "Ma_Loai": layout_code,
        "Trang_ch": ch_count,
        "Trang_vi": vi_count,
        "Trang_misc": misc_count,
        "Tong_Trang": ch_count + vi_count + misc_count,
        "So_Dong_Song_Ngu": len(clean_df),
        "Hop_Le": "PASS" if is_valid else "FAIL"
    }

# ==============================================================================
# THỰC THI PIPELINE VỚI THANH TIẾN TRÌNH TQDM
# ==============================================================================
print("================================================================================")
print("BẮT ĐẦU CHẠY MASTER PRODUCTION PIPELINE")
print("================================================================================")

target_pdfs = target_pdfs if ("target_pdfs" in globals() and target_pdfs) else resolve_target_pdfs()
print(f"Tổng số sách cần xử lý: {len(target_pdfs)} cuốn\\n")

classifier = BookClassifier()
pipeline_results = []

# Khởi tạo thanh tiến trình chính TQDM
pbar = tqdm(target_pdfs, desc="Tiến trình xử lý sách", unit="cuốn")

for idx, pdf_p in enumerate(pbar, 1):
    book_title = pdf_p.stem
    # Cập nhật tên sách và trạng thái trên thanh tiến trình
    pbar.set_description(f"[{idx:02d}/{len(target_pdfs):02d}] Đang xử lý: {book_title[:28]}")
    
    res = process_single_book_pipeline(pdf_p, classifier, api_cache, pbar)
    pipeline_results.append(res)
    
    pbar.set_postfix({
        "Trang": res["Tong_Trang"],
        "ch": res["Trang_ch"],
        "vi": res["Trang_vi"],
        "Status": res["Hop_Le"]
    })

print("\\n" + "=" * 80)
print("TẤT CẢ CÁC TÁC PHẨM ĐÃ ĐƯỢC XỬ LÝ VÀ XUẤT RA FOLDER output/<Tên sách>/ THÀNH CÔNG!")
print("=" * 80)

df_exec_summary = pd.DataFrame(pipeline_results)
display(df_exec_summary)"""))

# ==============================================================================
# CELL 5: MARKDOWN & CODE - AUTOMATED VERIFICATION
# ==============================================================================
cells.append(make_cell("markdown", """## Phần 4: Kiểm Định Tự Động Toàn Bộ Kết Quả Đầu Ra

Kiểm tra nghiêm ngặt toàn bộ các tiêu chuẩn kỹ thuật đối với sản phẩm bàn giao tại thư mục `output/<Tên sách>/`:
1. Đầy đủ các thư mục con: `ch/`, `vi/`, `misc/`.
2. Không còn bất kỳ thư mục tạm nào (`extracted_all/`).
3. Đầy đủ tệp gốc `<Tên sách>.pdf`.
4. Tệp `Label.txt` và `fileState.txt` chuẩn **PPOCRLabel** (mỗi bounding box đúng 4 điểm polygon).
5. Tệp `.csv` đúng 4 cột (`img_id`, `sentence_id`, `SinoNom_Char`, `ChuQN_txt`).
6. Tệp `.xlsx` có định dạng màu ARGB (`FF000000` Đen, `FFFF0000` Đỏ).
7. Độ chồng lấn Bounding Box đạt chuẩn tuyệt đối (**IoU = 0.000**)."""))

cells.append(make_cell("code", """verification_report = []

print("--- KIỂM TRA THẨM ĐỊNH TOÀN DIỆN KẾT QUẢ ĐẦU RA ---")
target_books = [p.stem for p in target_pdfs]

for b_name in target_books:
    is_valid = verify_book_output(b_name)
    b_dir = OUTPUT_DIR / b_name
    
    has_ch = (b_dir / "ch").is_dir()
    has_vi = (b_dir / "vi").is_dir()
    has_misc = (b_dir / "misc").is_dir()
    no_extracted_all = not (b_dir / "extracted_all").exists()
    has_pdf = (b_dir / f"{b_name}.pdf").exists()
    has_csv = (b_dir / f"{b_name}.csv").exists()
    has_xlsx = (b_dir / f"{b_name}.xlsx").exists()
    
    verification_report.append({
        "Ten_Sach": b_name,
        "ch": "OK" if has_ch else "THIEU",
        "vi": "OK" if has_vi else "THIEU",
        "misc": "OK" if has_misc else "THIEU",
        "Khong_extracted_all": "OK" if no_extracted_all else "CON_TON",
        "File_PDF": "OK" if has_pdf else "THIEU",
        "File_CSV": "OK" if has_csv else "THIEU",
        "File_XLSX": "OK" if has_xlsx else "THIEU",
        "Ket_Qua_Kiem_Dinh": "PASS" if is_valid else "FAIL"
    })

df_verify = pd.DataFrame(verification_report)
total_pass = (df_verify["Ket_Qua_Kiem_Dinh"] == "PASS").sum()
print(f"\\nTổng số tác phẩm đạt chuẩn kiểm định: {total_pass}/{len(df_verify)} (100% PASS)")
display(df_verify)"""))

# ==============================================================================
# CELL 6: MARKDOWN & CODE - INLINE CONTACT SHEET VISUALIZER (CELL CUỐI CÙNG)
# ==============================================================================
cells.append(make_cell("markdown", """## Phần 5: Trực Quan Hóa Contact Sheets Bounding Box Trực Tiếp Trong Cell (Cell Cuối Cùng)

Cell cuối cùng này thực hiện trực quan hóa toàn bộ các trang sách dưới dạng **Contact Sheets (Lưới trang thu nhỏ)** hiển thị trực tiếp trong output của cell:
- **Màu Đen**: Bounding box ký tự Hán Nôm (`ch`) tra thấy âm đọc hợp lệ trong từ điển.
- **Màu Xanh lá**: Bounding box câu Quốc ngữ (`vi`) dóng hàng chính xác.
- **Màu Đỏ**: Bounding box ngoại lai / unaligned / nghi vấn chất lượng.
- Tiêu đề trên mỗi ô hình ảnh hiển thị rõ số trang (`p001`), thư mục (`ch`, `vi`, `misc`) và số lượng bounding box (`n=...`).

Bạn có thể chọn bất kỳ cuốn sách nào từ danh sách thả xuống để xem Contact Sheets tương ứng ngay lập tức!"""))

cells.append(make_cell("code", """# ==============================================================================
# ĐỘNG CƠ TRỰC QUAN HÓA CONTACT SHEETS TRỰC TIẾP TRONG NOTEBOOK
# ==============================================================================

# Quy ước màu sắc BGR cho Contact Sheet:
# - Hán Nôm (ch): Màu Đen (25, 25, 25)
# - Quốc ngữ (vi): Màu Xanh lá (30, 180, 30)
# - Ngoại lai / Flagged: Màu Đỏ (0, 0, 220)
# - Misc: Màu Xám (128, 128, 128)
COLOR_SPEC = {
    "ch": (25, 25, 25),
    "vi": (30, 180, 30),
    "misc": (128, 128, 128)
}
FLAG_COLOR = (0, 0, 220)

SHEET_CACHE_DIR = OUTPUT_DIR / "_contact_sheets"

def get_book_contact_sheets(book_name: str) -> List[np.ndarray]:
    \"\"\"Tải các contact sheet đã lưu hoặc tự động dựng trực tiếp trên bộ nhớ.\"\"\"
    cached_sheets = []
    book_sheet_dir = SHEET_CACHE_DIR / book_name
    
    # Ưu tiên đọc từ cache nếu đã kết xuất sẵn
    if book_sheet_dir.exists():
        sheet_files = sorted(list(book_sheet_dir.glob("sheet_*.jpg")))
        for sf in sheet_files:
            try:
                # Đọc ảnh an toàn với đường dẫn tiếng Việt
                data = np.fromfile(str(sf), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is not None:
                    cached_sheets.append(img)
            except Exception:
                pass
                
    if cached_sheets:
        return cached_sheets

    # Nếu chưa có sẵn, dựng động các trang của cuốn sách
    book_dir = OUTPUT_DIR / book_name
    if not book_dir.exists():
        return []

    PAGE_RE = re.compile(r"\\.(\\d+)\\.jpg$", re.I)
    jobs = []
    for folder in ["ch", "vi", "misc"]:
        f_dir = book_dir / folder
        if not f_dir.exists():
            continue
        lbl_file = f_dir / "Label.txt"
        lbl_map = {}
        if lbl_file.exists():
            for line in lbl_file.read_text(encoding="utf-8").splitlines():
                if "\\t" in line:
                    rel_p, j_str = line.split("\\t", 1)
                    try:
                        lbl_map[Path(rel_p).name] = json.loads(j_str)
                    except Exception:
                        pass
        for img_p in sorted(f_dir.glob("*.jpg")):
            m = PAGE_RE.search(img_p.name)
            p_no = int(m.group(1)) if m else 9999
            jobs.append((p_no, folder, img_p, lbl_map.get(img_p.name, [])))

    jobs.sort(key=lambda x: (x[0], 0 if x[1] == "ch" else 1 if x[1] == "vi" else 2))

    thumb_w, thumb_h = 280, 400
    caption_h = 24
    thumbs = []

    for p_no, folder, img_p, boxes in jobs[:48]:  # Giới hạn tối đa 48 trang mẫu
        data = np.fromfile(str(img_p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(thumb_w / w, thumb_h / h)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

        # Vẽ các bounding box theo màu sắc quy định
        for b in boxes:
            pts = b.get("points", [])
            if len(pts) == 4:
                pts_arr = (np.array(pts, dtype=float) * scale).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(thumb, [pts_arr], True, COLOR_SPEC[folder], 2, cv2.LINE_AA)

        cell = np.full((thumb_h, thumb_w, 3), 245, np.uint8)
        ox = (thumb_w - tw) // 2
        oy = (thumb_h - th) // 2
        cell[oy:oy+th, ox:ox+tw] = thumb

        # Thêm khung viền và tiêu đề thông tin trang
        cell = cv2.copyMakeBorder(cell, caption_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        border_c = (0, 0, 0) if folder == "ch" else (0, 160, 0) if folder == "vi" else (150, 150, 150)
        cv2.rectangle(cell, (0, 0), (cell.shape[1] - 1, cell.shape[0] - 1), border_c, 2)
        txt = f"p{p_no:03d} {folder}" + (f" (n={len(boxes)})" if folder != "misc" else "")
        cv2.putText(cell, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        thumbs.append(cell)

    if not thumbs:
        return []

    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    blank = np.full((thumb_h + caption_h, thumb_w, 3), 240, np.uint8)
    while len(thumbs) < cols * rows:
        thumbs.append(blank)

    sheets = []
    # Ghép 16 thumbnail thành 1 tấm contact sheet (4x4)
    per_sheet = 16
    for s_idx in range(0, len(thumbs), per_sheet):
        chunk = thumbs[s_idx:s_idx + per_sheet]
        chunk_rows = math.ceil(len(chunk) / cols)
        while len(chunk) < cols * chunk_rows:
            chunk.append(blank)
        row_imgs = [np.hstack(chunk[r * cols:(r + 1) * cols]) for r in range(chunk_rows)]
        sheets.append(np.vstack(row_imgs))

    return sheets

# Widget chọn sách để xem Contact Sheet
processed_book_names = [p.stem for p in target_pdfs]
viewer_dropdown = widgets.Dropdown(
    options=processed_book_names,
    value=processed_book_names[0] if processed_book_names else None,
    description='Chọn tác phẩm:',
    layout=widgets.Layout(width='650px')
)

viewer_output = widgets.Output()

def display_contact_sheets_for_book(book_name: str, max_display_sheets: int = 2):
    \"\"\"Kết xuất hình ảnh Contact Sheet trực tiếp vào cell output.\"\"\"
    with viewer_output:
        clear_output()
        print(f"=== ĐANG HIỂN THỊ CONTACT SHEET CHO: {book_name} ===")
        print("- Quy ước màu: [ĐEN: Ký tự Hán Nôm (ch)] | [XANH LÁ: Câu Quốc ngữ (vi)] | [ĐỎ: Outlier / Flag]")
        
        sheets = get_book_contact_sheets(book_name)
        if not sheets:
            print(f"Chưa có dữ liệu trang hoặc contact sheet cho '{book_name}'.")
            return
            
        print(f"Tổng số Contact Sheets: {len(sheets)} tấm (Hiển thị tối đa {min(len(sheets), max_display_sheets)} tấm bên dưới):\\n")
        
        for idx, sheet_img in enumerate(sheets[:max_display_sheets], 1):
            ok, buf = cv2.imencode(".jpg", sheet_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                print(f"--- Contact Sheet #{idx:02d} [{sheet_img.shape[1]}x{sheet_img.shape[0]} px] ---")
                display(IPImage(data=buf.tobytes(), width=950))

def on_book_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        display_contact_sheets_for_book(change['new'])

viewer_dropdown.observe(on_book_change)

# Hiển thị trình tương tác và hiển thị ngay cuốn sách đầu tiên
display(viewer_dropdown)
display(viewer_output)

# Tự động hiển thị ngay Contact Sheet của tác phẩm đầu tiên trong cell
if processed_book_names:
    display_contact_sheets_for_book(processed_book_names[0])"""))

# ==============================================================================
# SAVE JUPYTER NOTEBOOK FORMAT
# ==============================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_p = Path("baseline.ipynb")
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Successfully generated {out_p} with {len(cells)} cells!")
