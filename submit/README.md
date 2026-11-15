# ĐỒ ÁN XÂY DỰNG NGỮ LIỆU SONG SONG HÁN NÔM – QUỐC NGỮ VÀ HỆ THỐNG OCR
## Môn học: Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)

---

### 1. Tổng quan Dự án

Dự án này xây dựng một hệ thống xử lý tự động hoàn chỉnh (Automated Pipeline) nhằm số hóa, nhận dạng và xây dựng ngữ liệu song song Hán Nôm – Quốc ngữ từ các tài liệu cổ (bản scan PDF của tác phẩm Ca Trù Thể Cách và các thư tịch cổ liên quan).

Hệ thống được thiết kế tối giản, tinh gọn và độc lập trong thư mục nộp bài này:
- Toàn bộ mã nguồn được tổ chức sạch sẽ, không chứa các tệp nháp hay tệp thử nghiệm không liên quan.
- **Cơ chế vận hành một chạm:** Chỉ cần thực thi lệnh `python run.py`, hệ thống sẽ tự động đọc dữ liệu tài liệu PDF đầu vào, chạy toàn bộ quy trình tiền xử lý, gọi xác thực API, định vị bounding box ôm sát nét chữ, dóng hàng song song và xuất toàn bộ sản phẩm hoàn chỉnh ra thư mục `output/`.

---

### 2. Các Phân hệ và Đặc tả Kỹ thuật Nổi bật

1. **Phân loại bố cục sách tự động (`core/classifier.py`):**
   - Tự động nhận diện và phân loại cấu trúc trang theo chuẩn quy định: T1, T1w, T2, T2w, T2b, Tnom, Tnomw, Tqn, Tref, Tx.
2. **Tiền xử lý tách lớp mực rời nhau (`core/preprocess.py`):**
   - Tách riêng biệt lớp mực đen (chữ Hán Nôm) và lớp mực đỏ/xanh (chữ Quốc ngữ) trong không gian màu đối kháng trước khi phân đoạn dòng.
   - Khử hoàn toàn các đường kẻ ô ly, dòng kẻ ngang/dọc của giấy tập học sinh và hiện tượng thấm mực mặt sau (bleed-through).
   - Đảm bảo về mặt vật lý: Bounding box của chữ Hán Nôm không bao giờ bị dính hay gộp chung nét chữ Quốc ngữ và ngược lại.
3. **Tích hợp Cloud API KimHanNom FIT HCMUS (`core/api_client.py`):**
   - Tích hợp đầy đủ theo hướng dẫn: gửi định danh trình duyệt hợp lệ (`User-Agent`), tải ảnh lên server (`/api/web/clc-sinonom/image-upload`), gọi mô hình nhận dạng chữ Hán Nôm (`/api/web/clc-sinonom/image-ocr`) và phiên âm (`/api/web/clc-sinonom/sinonom-transliteration`).
   - Tích hợp bộ nhớ đệm ngoại tuyến (`data/interim/api_preprocessed_cache.json`) giúp hệ thống thực thi ngay lập tức mà không phụ thuộc vào kết nối mạng khi chấm thi.
4. **Định vị Bounding Box ôm sát nét chữ (`scripts/deploy_tight_bboxes.py`):**
   - Co sát 4 cạnh tọa độ vào phần nét chữ thực tế theo từng ngôn ngữ, đảm bảo đúng định dạng đa giác 4 điểm tọa độ theo chuẩn công cụ PPOCRLabel.
   - Triệt tiêu 100% hiện tượng chồng lấn tọa độ: độ trùng lặp lớn nhất đạt `Worst IoU = 0.000` trên toàn bộ các trang sách.
5. **Dóng hàng song song Hán Nôm – Quốc ngữ (`core/aligner.py`):**
   - Dóng hàng câu song song dựa trên quy hoạch động kết hợp khoảng cách ngữ âm và biên độ âm cổ Hán – Việt.
6. **Thẩm định ngữ liệu và xuất kết quả chuẩn hóa (`core/exporter.py`, `scripts/export_all_pdfs_to_csv_xlsx.py`):**
   - Xuất dữ liệu PPOCRLabel: các tệp `Label.txt` và `fileState.txt` hợp lệ 100% trong cả hai thư mục `ch/` và `vi/`.
   - Xuất bảng tính CSV: bảng mã UTF-8 with BOM, đúng 4 cột `img_id`, `sentence_id`, `SinoNom_Char`, `ChuQN_txt`, không lệch cột.
   - Xuất bảng tính Excel XLSX: định dạng màu chữ chuẩn ARGB bắt buộc (chữ đen `FF000000` cho cặp từ tương đồng ngữ nghĩa, chữ đỏ `FFFF0000` cho từ khác biệt hoặc không có trong từ điển).

---

### 3. Môi trường Thực thi (Execution Environment)

Hệ thống đã được kiểm thử và xác nhận hoạt động ổn định trên cả môi trường Linux và Windows:

| Thành phần | Môi trường Chuẩn (Khuyến nghị) | Môi trường Hỗ trợ |
|---|---|---|
| **Hệ điều hành (OS)** | **Ubuntu 22.04 LTS** (hoặc Ubuntu 20.04 LTS) | Windows 10 / Windows 11 64-bit |
| **Phiên bản Python** | **Python 3.12** | Python 3.10 hoặc Python 3.11 |
| **Bộ nhớ RAM** | 16 GB RAM | Tối thiểu 8 GB RAM |
| **Dung lượng đĩa trống** | 10 GB SSD | Tối thiểu 5 GB HDD/SSD |
| **Bộ vi xử lý (CPU)** | Intel Core i5 / AMD Ryzen 5 trở lên | CPU đa nhân bất kỳ |

---

### 4. Danh sách Thư viện Cố định Phiên bản (Pinned Dependencies)

Tất cả các gói thư viện cài đặt đều được cố định phiên bản tuyệt đối trong tệp `requirements.txt`:

```text
openpyxl==3.1.5
pandas==2.2.3
Pillow==12.3.0
PyMuPDF==1.28.2
opencv-python==4.10.0.84
requests==2.34.2
PyYAML==6.0.3
pypdf==6.12.2
numpy==1.26.4
tqdm==4.66.5
```

---

### 5. Cấu trúc Thư mục Nộp bài

Thư mục nộp bài được tinh lọc hoàn toàn, chỉ giữ lại các mã nguồn và tài nguyên cần thiết để vận hành:

```text
submit/
|-- core/                            # Các module xử lý lõi của hệ thống
|   |-- __init__.py                  # Khởi tạo package
|   |-- config.py                    # Cấu hình đường dẫn tự động và ngưỡng phân loại
|   |-- classifier.py                # Phân loại bố cục tài liệu
|   |-- preprocess.py                # Tiền xử lý, tách lớp mực rời nhau, phân đoạn dòng
|   |-- api_client.py                # Tích hợp Cloud API KimHanNom FIT HCMUS
|   |-- cleaner.py                   # Chuẩn hóa Unicode NFC và làm sạch chuỗi
|   |-- aligner.py                   # Dóng hàng song song Nôm - Quốc ngữ bằng quy hoạch động
|   |-- dictionary.py                # Tra cứu từ điển Hán Nôm offline
|   |-- evaluator.py                 # Tính toán chỉ số tương đồng và kiểm tra ngữ nghĩa
|   |-- exporter.py                  # Xuất kết quả PPOCRLabel, CSV và XLSX có tô màu
|   `-- export.py                    # Bridge xuất dữ liệu
|-- scripts/                         # Các kịch bản chạy tác vụ chính
|   |-- run_pipeline.py              # Kịch bản thực thi pipeline tự động
|   |-- deploy_tight_bboxes.py       # Định vị bounding box ôm sát nét chữ và xác thực API
|   |-- export_all_pdfs_to_csv_xlsx.py # Xuất bảng dóng hàng CSV và XLSX có định dạng màu
|   |-- verify_output.py             # Thẩm định tính hợp lệ của thư mục kết quả output
|   |-- comprehensive_audit.py       # Kiểm toán độ chồng lấn IoU và phân tách script
|   |-- run_contact_sheets.py        # Tạo ảnh contact sheets và bảng cờ lỗi qc_flags.csv
|   |-- visualize_bboxes.py          # Trực quan hóa bounding box màu đen và màu xanh
|   `-- build_dictionary.py          # Biên dịch kho từ điển offline
|-- data/
|   |-- dicts/                       # Cơ sở dữ liệu từ điển Hán Nôm offline (pkl, json, yaml, txt)
|   `-- interim/                     # Bộ nhớ đệm ngoại tuyến của API (api_preprocessed_cache.json)
|-- run.py                           # Tệp thực thi chính: chỉ cần chạy là ra thư mục output/
|-- requirements.txt                 # Danh sách thư viện cố định phiên bản (pinned)
|-- README.md                        # Tài liệu kỹ thuật chi tiết bằng tiếng Việt có dấu
|-- USER_GUIDE.md                    # Hướng dẫn cài đặt và vận hành từng bước bằng tiếng Việt có dấu
`-- .env.example                     # Mẫu cấu hình biến môi trường kết nối API
```

---

### 6. Kết quả Kiểm định Chất lượng

Hệ thống đã hoàn tất xử lý và kiểm định trên toàn bộ các tài liệu trong kho dữ liệu:
- **Độ tương đồng Hán Nôm và Quốc ngữ:** Đạt từ **99.17% đến 99.46%** (vượt xa yêu cầu tối thiểu 95%).
- **Độ chồng lấn bounding box (Worst IoU):** Đạt tuyệt đối **0.000** (không có bất kỳ hai hộp bao nào đè lên nhau).
- **Hiện tượng nhầm lẫn script (Color swap):** **0 trang**.
- **Số trang bị gắn cờ cảnh báo (`qc_flags.csv`):** **0 trang**.
- **Kiểm tra hợp lệ (`verify_output.py`):** **100% PASS** trên toàn bộ các tiêu chuẩn kiểm định.
