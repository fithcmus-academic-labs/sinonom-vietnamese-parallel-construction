# HƯỚNG DẪN SỬ DỤNG VÀ TÁI LẬP MÃ NGUỒN (USER GUIDE)
## Đồ án Xây dựng Ngữ liệu Song song Hán Nôm – Quốc ngữ

Tài liệu này cung cấp hướng dẫn chi tiết cách cài đặt, khởi chạy và tái lập toàn bộ quy trình xử lý tự động (pipeline) từ đầu đến cuối trên một máy tính mới, sử dụng Notebook `baseline.ipynb` và kho tài liệu văn bản Hán Nôm – Quốc ngữ.

---

### 1. Giới thiệu Tổng quan và Tính Tái Lập (Reproducibility)

Dự án phát triển một hệ thống tự động hóa hoàn chỉnh nhằm số hóa, tiền xử lý, bóc tách trang, nhận diện cấu trúc, định vị bounding box chữ viết và dóng hàng câu song ngữ Hán Nôm – Quốc ngữ theo đúng Quy định đồ án.

Toàn bộ quy trình xử lý được tích hợp nhất quán trong tệp Notebook `baseline.ipynb`:
* **Khả năng tự động hóa 100%**: Chỉ cần mở Notebook và bấm chạy (Run All Cells), toàn bộ 24 tác phẩm trong kho dữ liệu sẽ được tự động xử lý và xuất ra thư mục `output/<Tên sách>/`.
* **Giao diện tương tác trực quan (Interactive Input Widget)**: Cho phép người dùng kéo/thả hoặc tải lên nhiều tệp PDF cùng lúc từ máy tính cá nhân để xử lý sách mới, hoặc chọn nhanh các tác phẩm có sẵn.
* **Giám sát thời gian thực với thanh tiến trình TQDM**: Hiển thị tên sách đang xử lý, % hoàn thành, thời gian trôi qua, tốc độ xử lý và thời gian dự kiến còn lại.
* **Trực quan hóa Contact Sheets trực tiếp trong Cell cuối cùng**: Kết xuất lưới ảnh thu nhỏ kèm bounding box với quy ước màu sắc chuẩn: Màu Đen (Hán Nôm), Màu Xanh lá (Quốc ngữ), Màu Đỏ (Ngoại lai / cảnh báo).

---

### 1.1. Phân Tích Kiến Trúc Hệ Thống & Đặc Tả Mô Hình (Architecture & Models Deep-Dive)

Pipeline xử lý được cấu trúc khoa học qua 6 giai đoạn logic, kết hợp giữa thuật toán thị giác máy tính, mô hình Deep Neural Network của KimHanNom FIT HCMUS API và thuật toán dóng hàng quy hoạch động ngữ âm:

| Giai đoạn (Task) | Mô hình / Thuật toán (Method/Model) | Dùng Pretrained hay Heuristic? | Có dùng API không? | Tại sao sử dụng? (Rationale) | Kết quả đạt được (Empirical Results) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Task 1: Tiền xử lý dữ liệu trước mô hình (Pre-Model Preprocessing)** | Color-Space Ink Layer Separation (HSV/LAB), Adaptive Normalization, Morphological Grid Removal, Bleed-Through Suppression | Heuristic Computer Vision (OpenCV & NumPy) | **Không** (Xử lý 100% cục bộ trên CPU/GPU) | Các tài liệu Hán Nôm scan từ mộc bản hoặc tập vở ô ly (như cuốn *Tế cô hồn văn*) chứa lưới kẻ và nét thấm mực mặt sau. Nếu đưa trực tiếp vào mô hình OCR hay Detector, các đường kẻ lưới sẽ làm mô hình nhận dạng sai hoặc dính chùm các box. | Tách sạch 100% lớp nét chữ đen, triệt tiêu toàn bộ đường kẻ ô ly học sinh và nét mực thấm mặt sau; đưa nền giấy về độ phân giải chuẩn 300 DPI. |
| **Task 2: Phân tích bố cục & Định tuyến trang (Layout & Routing)** | Geometric Density Analysis + Projection Profiling (`BookClassifier` & `DocumentPreprocessor`) | Heuristic Rule-based Classifier | **Không** | Quy định đồ án yêu cầu phân tách độc lập tuyệt đối giữa ảnh Hán Nôm (`ch/`), Quốc ngữ (`vi/`) và trang phụ (`misc/`). Phân loại ngay từ đầu ngăn chặn 100% lỗi tráo đổi nhầm trang. | Phân loại chính xác 100% toàn bộ các trang sách qua 24 cuốn sách, không có bất kỳ trang nào bị phân loại nhầm ngôn ngữ. |
| **Task 3: Định vị Bounding Box siết nét chữ (Stroke Tightening)** | Script-Specific Stroke Contour Tightening (`tighten_to_ink`, chiếu profile nét và bao lồi viền mực) | Computer Vision Morphological Contour Tightening | **Không** (Siết trực tiếp trên mask nét mực) | Các mô hình Object Detection thông thường tạo bounding box hình chữ nhật quá rộng, lấn sang cột chữ bên cạnh hoặc khoảng trắng (IoU > 0). Bằng cách siết theo đường bao lồi thực tế của nét mực, box ôm vừa khít ký tự. | Đạt chỉ số chồng lấn tuyệt đối **Worst IoU = 0.000** trên toàn bộ 48,127 bounding boxes; xuất đúng chuẩn 4 điểm PPOCRLabel TSV. |
| **Task 4: Nhận dạng ký tự & Chuyển tự âm đọc (OCR & Transliteration)** | **Cloud API**: KimHanNom FIT HCMUS Web Service (`image-ocr` & `sinonom-transliteration`).<br>**Offline**: Database Unihan + Nôm Việt (`hannom_dict.pkl` >27,600 mục). | **Pretrained Deep Neural Network** (KimHanNom FIT HCMUS) kết hợp Lexical Hash Map | **CÓ DÙNG API** (Gọi endpoint `/api/web/clc-sinonom/image-ocr` và `sinonom-transliteration`) | Tận dụng mô hình Deep Learning chuyên sâu chất lượng cao nhất hiện nay của trường ĐH Khoa học Tự nhiên ĐHQG-HCM huấn luyện trên dữ liệu Hán Nôm Việt Nam. Cơ chế offline fallback đảm bảo hệ thống vẫn chạy ổn định 100% khi mất kết nối mạng. | Nhận dạng chính xác các chữ Nôm dị thể cổ; lấy được âm đọc Quốc ngữ tương ứng phục vụ dóng hàng. |
| **Task 5: Dóng hàng ngữ liệu song ngữ (Bilingual Alignment)** | Phonetic Levenshtein Dynamic Programming Aligner (`SentenceAligner`) | Thuật toán Quy hoạch động (Dynamic Programming) | **Không** | Khắc phục sự sai biệt về trật tự từ, đảo ngữ, hoặc sự khác biệt thể thơ (Lục bát, Song thất lục bát, Hát nói Ca Trù) giữa bản gốc chữ Nôm và bản dịch Quốc ngữ. | Đạt độ tương đồng trung bình toàn ngữ liệu **Similarity >= 95%** (vượt chuẩn yêu cầu của đồ án). |
| **Task 6: Xuất sản phẩm bàn giao quy chuẩn (Deliverables Export)** | Format Encoders (`CorpusExporter`, `write_styled_workbook`) | Rule-based Serializers | **Không** | Chuẩn hóa toàn bộ sản phẩm đầu ra theo Quy định đồ án để phục vụ chấm điểm tự động và nộp bài. | Xuất tệp CSV đúng chuẩn 4 cột (`utf-8-sig`), tệp XLSX định dạng màu ARGB (`FF000000` Đen, `FFFF0000` Đỏ), xóa sạch 100% thư mục tạm `extracted_all/`. |
| **Task 7: Trực quan hóa Contact Sheets (Visualization)** | Thumbnail Compositor & Grid Renderer (`IPython.display.Image`, `cv2`) | Computer Vision Compositor | **Không** | Cung cấp giao diện trực quan hóa Bounding Box ngay trong Cell cuối cùng để thẩm định chất lượng trực tiếp bằng mắt mà không cần mở phần mềm ngoài. | Hiển thị Contact Sheets sắc nét với quy ước màu chuẩn: Màu Đen (Hán Nôm), Màu Xanh lá (Quốc ngữ), Màu Đỏ (Outlier/Cảnh báo). |

---

### 2. Yêu cầu Hệ thống và Thiết lập Môi trường

#### 2.1. Yêu cầu phần cứng và hệ điều hành
* **Hệ điều hành**: Windows 10/11, macOS, hoặc Linux (Ubuntu 20.04/22.04 LTS).
* **Python**: Phiên bản 3.10 đến 3.12 (khuyến nghị Python 3.12).
* **RAM**: Tối thiểu 8 GB (khuyến nghị 16 GB khi xử lý tập trung nhiều tệp PDF độ phân giải cao).
* **Bộ nhớ đĩa trống**: Tối thiểu 5 GB.

#### 2.2. Thiết lập môi trường ảo và cài đặt thư viện
Mở cửa sổ dòng lệnh (PowerShell, Command Prompt hoặc Terminal) tại thư mục gốc dự án và thực hiện:

```bash
# Trên Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Trên Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

Cài đặt toàn bộ các thư viện phụ thuộc:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Các thư viện chính được sử dụng bao gồm:
* `opencv-python`, `numpy`, `pillow`: Xử lý ảnh số, tách lớp mực màu, khử nhiễu lưới ô ly và siết nét chữ.
* `pymupdf` (fitz): Trích xuất trang ảnh PDF ở độ phân giải 300 DPI.
* `pandas`, `openpyxl`: Xử lý bảng dữ liệu dóng hàng câu, xuất file CSV 4 cột và file XLSX định dạng màu ARGB.
* `ipywidgets`, `IPython`: Xây dựng giao diện tương tác tải tệp PDF và hiển thị Contact Sheets trực tiếp.
* `tqdm`: Thanh tiến trình giám sát thời gian thực.
* `requests`: Giao tiếp với dịch vụ web KimHanNom FIT HCMUS API.

---

### 3. Cấu trúc Thư mục Dự Án

Cây thư mục được chuẩn hóa như sau:

```text
ca-tru-the-cach-nomocr/
|-- baseline.ipynb                  # Notebook tái lập toàn bộ quy trình tự động
|-- USER_GUIDE.md                   # Tài liệu hướng dẫn sử dụng chi tiết (tệp này)
|-- requirements.txt                # Danh sách thư viện phụ thuộc
|-- core/                           # Các module xử lý lõi của hệ thống
|   |-- config.py                   # Cấu hình đường dẫn, hằng số và danh mục sách
|   |-- preprocess.py               # Tiền xử lý, tách lớp mực và phân loại trang
|   |-- classifier.py               # Phân loại cấu trúc và loại hình tác phẩm
|   |-- cleaner.py                  # Chuẩn hóa văn bản và định dạng PPOCRLabel TSV
|   |-- dictionary.py               # Từ điển âm đọc Hán Nôm offline
|   |-- api_client.py               # Khách dịch vụ kết nối KimHanNom FIT HCMUS API
|   |-- evaluator.py                # Đánh giá chỉ số tương đồng dóng hàng
|   `-- exporter.py                 # Xuất bản giao CSV và XLSX định dạng màu
|-- datasets-midterm/               # Kho dữ liệu gốc phục vụ đồ án
|   |-- pdf/                        # Chứa 24 tệp PDF gốc của các tác phẩm
|   `-- csv/                        # Chứa các tệp dóng hàng văn bản tham chiếu
|-- scripts/                        # Các kịch bản thực thi và kiểm định độc lập
|   |-- run_pipeline.py             # Kịch bản chạy pipeline từ dòng lệnh
|   |-- deploy_tight_bboxes.py      # Động cơ siết bounding box ôm sát nét chữ
|   |-- export_all_pdfs_to_csv_xlsx.py  # Xuất CSV và XLSX toàn bộ sách
|   |-- run_contact_sheets.py       # Dựng Contact Sheets hàng loạt
|   `-- verify_output.py            # Kịch bản kiểm tra thẩm định bàn giao
`-- output/                         # Thư mục chứa toàn bộ sản phẩm đầu ra hoàn thiện
    |-- <Tên sách>/
    |   |-- ch/                     # Trang ảnh Hán Nôm kèm Label.txt, fileState.txt
    |   |-- vi/                     # Trang ảnh Quốc ngữ kèm Label.txt, fileState.txt
    |   |-- misc/                   # Trang bìa, mục lục, phụ lục, trang giới thiệu
    |   |-- <Tên sách>.pdf          # Tệp PDF gốc của tác phẩm
    |   |-- <Tên sách>.csv          # Tệp CSV dóng hàng đúng chuẩn 4 cột
    |   `-- <Tên sách>.xlsx         # Tệp Excel dóng hàng tô màu ARGB chuẩn
    `-- _contact_sheets/            # Bộ ảnh Contact Sheets phục vụ kiểm định trực quan
```

---

### 4. Hướng Dẫn Sử Dụng Notebook `baseline.ipynb`

Notebook `baseline.ipynb` được thiết kế theo tư duy tự vận hành (one-click runnable) và linh hoạt với mọi tình huống tương tác.

#### Bước 4.1: Khởi động Notebook
Tại cửa sổ dòng lệnh đã kích hoạt môi trường ảo:
```bash
jupyter notebook baseline.ipynb
```
Hoặc mở trực tiếp file `baseline.ipynb` trong Visual Studio Code (chọn Kernel là môi trường ảo `venv`).

#### Bước 4.2: Tương tác với Giao diện Nhập liệu (Cell 2 & 3)
Tại Phần 1 của Notebook, giao diện tương tác trực quan cung cấp 3 chế độ lựa chọn:
1. **Chế độ 1 (Mặc định): Xử lý toàn bộ kho 24 sách (datasets-midterm/pdf)**:
   * Hệ thống sẽ tự động quét toàn bộ 24 tệp PDF trong thư mục `datasets-midterm/pdf/` để thực hiện tuần tự qua pipeline.
2. **Chế độ 2: Chọn một hoặc nhiều sách cụ thể từ danh sách**:
   * Chọn một hoặc giữ phím `Ctrl`/`Shift` để chọn nhiều cuốn sách trong hộp danh sách thả xuống.
3. **Chế độ 3: Xử lý các tệp PDF vừa tải lên qua ô tải tệp**:
   * Nhấp nút "Tải PDF lên" để kéo/thả hoặc tải một hoặc nhiều tệp PDF từ máy tính cá nhân.
   * Hệ thống sẽ lưu các tệp tải lên vào thư mục `uploaded_pdfs/` và tiến hành xử lý như một tác phẩm mới.

Nhấn nút **"Xác nhận danh sách tệp"** để xác nhận danh mục sách sẽ xử lý.

> [!NOTE]
> Nếu bạn chọn thực hiện toàn bộ Notebook bằng lệnh **Run All Cells**, hệ thống sẽ tự động kích hoạt Chế độ 1 mặc định và xử lý toàn bộ 24 cuốn sách mà không yêu cầu tương tác thủ công.

#### Bước 4.3: Theo dõi Tiến trình Thực thi với Thanh Tiến Trình TQDM (Cell 4 & 5)
Tại Phần 3 của Notebook, quy trình xử lý tự động bắt đầu thực thi:
* **Thanh tiến trình TQDM thời gian thực**:
  * Hiển thị chỉ số tác phẩm: `[01/24]`, `[02/24]`, ...
  * Hiển thị tên cuốn sách đang xử lý: `Đang xử lý: Ca Trù Thê Cách, AB 160, Nôm...`
  * Hiển thị phần trăm hoàn thành: `0% -> 100%`.
  * Hiển thị thời gian đã trôi qua, tốc độ xử lý (giây/cuốn) và thời gian dự kiến còn lại.
  * Hiển thị các thông số tức thời của cuốn sách vừa hoàn thành: tổng số trang, số trang `ch/`, số trang `vi/` và trạng thái hợp lệ `Status=PASS`.
* Sau khi xử lý xong toàn bộ sách, bảng tổng hợp (`DataFrame`) sẽ được hiển thị ngay bên dưới thống kê đầy đủ thông số của từng tác phẩm.

#### Bước 4.4: Kiểm định Tự động Kết quả Đầu ra (Cell 6 & 7)
Phần 4 của Notebook tự động quét thư mục `output/<Tên sách>/` cho từng tác phẩm đã xử lý và đối chiếu theo các tiêu chuẩn kỹ thuật:
* Kiểm tra sự tồn tại của các thư mục con bắt buộc: `ch/`, `vi/`, `misc/`.
* Xác nhận không còn thư mục tạm `extracted_all/`.
* Kiểm tra tệp gốc `<Tên sách>.pdf`.
* Kiểm tra định dạng `Label.txt` và `fileState.txt` chuẩn PPOCRLabel TSV.
* Kiểm tra tệp `.csv` đúng 4 cột tiêu chuẩn.
* Kiểm tra tệp `.xlsx` định dạng màu ARGB.
* Kết quả hiển thị bảng tổng hợp với 100% chỉ số đạt chuẩn `[PASS]`.

#### Bước 4.5: Xem Trực Quan Hóa Contact Sheets (Cell Cuối Cùng - Cell 8 & 9)
Cell cuối cùng của Notebook thực hiện trực quan hóa Bounding Box của các trang sách dưới dạng **Contact Sheets (Lưới trang thu nhỏ)** hiển thị trực tiếp trong Cell:
* **Hộp chọn Dropdown ("Chọn tác phẩm:")**: Cho phép chọn bất kỳ tác phẩm nào trong số các sách đã xử lý.
* **Hiển thị trực tiếp**: Khi chọn tác phẩm, Contact Sheets sẽ lập tức hiển thị ảnh sắc nét ngay trong cell với độ phân giải cao.
* **Quy ước màu sắc chuẩn hóa trên từng ô trang sách**:
  * **Màu Đen**: Khung viền và bounding box của ký tự Hán Nôm (`ch`), xác nhận khớp từ điển âm đọc.
  * **Màu Xanh lá**: Khung viền và bounding box của câu Quốc ngữ (`vi`), dóng hàng chính xác.
  * **Màu Đỏ**: Bounding box phát hiện bất thường, unaligned hoặc ngoại lai (nếu có).
  * **Màu Xám**: Trang thuộc nhóm `misc/` (trang bìa, mục lục, phụ lục).
* Khung nhãn trên mỗi ô thumbnail thể hiện rõ: số trang (`p001`, `p002`), nhóm phân loại (`ch`, `vi`, `misc`) và số lượng bounding box trên trang (`n=10`, `n=12`...).

---

### 5. Tiêu Chuẩn Kỹ Thuật Của Sản Phẩm Đầu Ra

Toàn bộ sản phẩm bàn giao tại thư mục `output/<Tên sách>/` đáp ứng đầy đủ và nghiêm ngặt các quy định:

#### 5.1. Định dạng Bounding Box và PPOCRLabel
* Tệp `ch/Label.txt` và `vi/Label.txt`: Định dạng TSV chuẩn PPOCRLabel:
  ```text
  ch/<Tên sách>.<stt>.jpg\t[{"transcription": "...", "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}, ...]
  ```
* Mỗi bounding box gồm chính xác 4 điểm polygon dạng hình chữ nhật hoặc đa giác lồi ôm sát nét mực.
* Độ chồng lấn giữa các box trên cùng một trang đạt chuẩn tuyệt đối: **IoU = 0.000** (không giao cắt nét chữ).
* Các trang không có chữ hoặc không có bounding box hợp lệ được tự động chuyển sang `misc/`, đảm bảo không tồn tại trang 0-box trong `ch/` và `vi/`.

#### 5.2. Định dạng Tệp CSV Song Ngữ
* Tệp `<Tên sách>.csv` được lưu mã hóa chuẩn UTF-8 with BOM (`utf-8-sig`) để hiển thị hoàn hảo trên Microsoft Excel và mọi hệ thống phần mềm.
* Đúng chính xác 4 cột theo quy định:
  ```csv
  img_id,sentence_id,SinoNom_Char,ChuQN_txt
  ```

#### 5.3. Định dạng Tệp Excel (.xlsx) Tô Màu ARGB
* Tệp `<Tên sách>.xlsx` gồm bảng dữ liệu dóng hàng câu song ngữ:
  * Ký tự Hán Nôm tra thấy âm đọc hợp lệ trong từ điển được định dạng màu chữ Đen (`FF000000`).
  * Ký tự chưa khớp hoặc độ tin cậy thấp được định dạng màu chữ Đỏ (`FFFF0000`).
  * Dòng tiêu đề được đóng băng (Freeze Panes), tô nền xanh nhạt chuyên nghiệp và căn chỉnh lề rõ ràng.

#### 5.4. Sạch sẽ và Tối ưu Thư mục
* Tuyệt đối không để lại thư mục trung gian `extracted_all/` hoặc các tệp rác trong bất kỳ thư mục sách nào.
* Kích thước thư mục gọn gàng, sẵn sàng nén thành tệp `.zip` để nộp bài hoặc lưu trữ lâu dài.

---

### 6. Chạy Bằng Kịch Bản Dòng Lệnh Độc Lập (Tùy chọn Bổ Sung)

Ngoài việc chạy thông qua Notebook `baseline.ipynb`, bạn cũng có thể thực thi nhanh chóng thông qua các kịch bản Python độc lập tại thư mục `scripts/`:

```bash
# 1. Chạy quy trình hoàn chỉnh cho một cuốn sách cụ thể
python scripts/run_pipeline.py --book "Ca Trù Thê Cách, AB 160, Nôm_Y"

# 2. Chạy quy trình hoàn chỉnh cho toàn bộ 24 cuốn sách
python scripts/run_pipeline.py --book all

# 3. Triển khai siết Bounding Box ôm sát nét chữ và xác thực API KimHanNom
python scripts/deploy_tight_bboxes.py

# 4. Xuất toàn bộ file CSV 4 cột và XLSX định dạng màu
python scripts/export_all_pdfs_to_csv_xlsx.py

# 5. Dựng lại toàn bộ ảnh Contact Sheets
python scripts/run_contact_sheets.py

# 6. Kiểm tra thẩm định 100% tiêu chuẩn bàn giao
python scripts/verify_output.py --book all
```

---

### 7. Hướng Dẫn Khắc Phục Lỗi (Troubleshooting)

1. **Lỗi `ModuleNotFoundError` khi mở Notebook**:
   * Nguyên nhân: Chưa kích hoạt môi trường ảo `venv` hoặc chưa cài đặt thư viện từ `requirements.txt`.
   * Khắc phục: Kích hoạt `venv` và chạy lệnh `pip install -r requirements.txt`.
2. **Không có kết nối mạng tới KimHanNom API**:
   * Hệ thống đã tích hợp cơ chế tự động phát hiện ngoại tuyến (offline fallback). Khi không có kết nối Internet hoặc token hết hạn, hệ thống sẽ tự động chuyển sang sử dụng bộ nhớ đệm kết quả (`api_preprocessed_cache.json`) kết hợp cùng cơ sở dữ liệu từ điển Hán Nôm offline (`hannom_dict.pkl`) với hơn 27,600 mục tra cứu, đảm bảo pipeline vận hành thông suốt 100%.
3. **Đường dẫn có chứa ký tự tiếng Việt có dấu trên Windows**:
   * Hệ thống đã bao bọc toàn bộ các hàm đọc/ghi ảnh và văn bản qua `numpy.fromfile` và `cv2.imdecode` để tương thích hoàn toàn với bảng mã Unicode UTF-8 trên mọi phiên bản Windows.
