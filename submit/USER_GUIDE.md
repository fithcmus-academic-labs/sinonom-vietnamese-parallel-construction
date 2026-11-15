# HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH HỆ THỐNG (USER GUIDE)
## Đồ án Xây dựng Ngữ liệu Song song Hán Nôm – Quốc ngữ và Nhận dạng Ký tự (OCR)

Tài liệu này hướng dẫn chi tiết cách cài đặt, cấu hình và chạy hệ thống từ đầu đến cuối trên một máy tính mới hoàn toàn nhằm sinh ra thư mục kết quả `output/` theo đúng các quy chuẩn nộp bài.

---

### 1. Yêu cầu Hệ thống và Chuẩn bị Môi trường

#### Bước 1.1: Kiểm tra phiên bản Python
Hệ thống yêu cầu phiên bản Python từ 3.10 đến 3.12 (khuyến nghị Python 3.12).
```bash
python --version
# hoặc trên hệ điều hành Linux:
python3 --version
```

#### Bước 1.2: Tạo và kích hoạt môi trường ảo (Virtual Environment)
Trên hệ điều hành Ubuntu / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Trên hệ điều hành Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Bước 1.3: Cài đặt các thư viện phụ thuộc đã cố định phiên bản
Cài đặt chính xác các thư viện theo tệp `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Vận hành Hệ thống: Cơ chế "Một Chạm" Sinh Thư mục `output/`

Thư mục nộp bài này đã được thiết kế sẵn để người dùng hoặc hội đồng chấm thi có thể thực thi ngay lập tức thông qua tệp điều khiển gốc `run.py`.

#### Cách chạy toàn bộ hệ thống tự động:
Chỉ cần thực thi lệnh sau:
```bash
python run.py
```

Khi lệnh này được thực hiện, hệ thống sẽ tự động:
1. Tự động nhận diện thư mục chứa các tệp PDF tài liệu gốc (tìm kiếm tự động trong các thư mục `input/`, `pdf/`, `datasets-midterm/pdf/`).
2. Thực hiện tiền xử lý ảnh trang tài liệu, khử nhiễu dòng kẻ ô ly của giấy tập học sinh, khử hiện tượng thấm mực mặt sau và tách lớp mực rời nhau (`mask_ch` cho chữ Hán Nôm, `mask_vi` cho chữ Quốc ngữ).
3. Định vị bounding box ôm sát nét chữ thực tế, đồng thời đối chiếu và kiểm tra văn bản qua API KimHanNom FIT HCMUS.
4. Triệt tiêu 100% độ chồng lấn tọa độ (`IoU = 0.000`), định dạng chuẩn đa giác 4 điểm cho từng hộp bao.
5. Dóng hàng song song tự động giữa câu Hán Nôm và Quốc ngữ bằng thuật toán quy hoạch động ngữ âm.
6. Xuất toàn bộ kết quả hoàn chỉnh vào thư mục `output/`, bao gồm:
   - Các thư mục con: `ch/`, `vi/`, `misc/`.
   - Các tệp nhãn chuẩn PPOCRLabel: `ch/Label.txt`, `ch/fileState.txt`, `vi/Label.txt`, `vi/fileState.txt`.
   - Tệp bảng tính: `<Tên sách>.csv` (bảng mã UTF-8 with BOM).
   - Tệp bảng tính: `<Tên sách>.xlsx` có định dạng tô màu chữ đen/đỏ ARGB bắt buộc.
7. Tự động chạy bộ thẩm định `verify_output.py` để xác nhận tất cả tiêu chuẩn đều đạt chuẩn (PASS).

#### Tùy chọn xử lý cho một cuốn sách cụ thể:
Nếu muốn chạy riêng cho một cuốn sách để kiểm tra nhanh:
```bash
# Ví dụ chạy riêng cuốn Ca Trù Thê Cách, AB 160:
python run.py --book AB_160

# Các mã sách khả dụng: AB_160, AB_431, AB_456, NOM_Y hoặc all
```

---

### 3. Thẩm định và Kiểm tra Kết quả Đầu ra

Sau khi quy trình chạy hoàn tất, người dùng có thể sử dụng các cờ lệnh kiểm tra chuyên sâu có sẵn trong `run.py`:

#### 3.1. Thẩm định tính hợp lệ của toàn bộ sản phẩm nộp bài
```bash
python run.py --verify-only
```
Lệnh này sẽ kiểm tra tự động:
- Cấu trúc cây thư mục bắt buộc (`ch/`, `vi/`, `misc/`).
- Tệp PDF gốc mang đúng tên tài liệu.
- Tính toàn vẹn của tệp `Label.txt` (mỗi bounding box chứa đúng 4 điểm tọa độ).
- Tính toàn vẹn của tệp `fileState.txt` (trạng thái đánh nhãn đạt giá trị 1).
- Tệp `.csv` đúng 4 cột quy định, không bị lệch hàng lệch cột.
- Tệp `.xlsx` có tô màu đen (chữ tương đồng) và màu đỏ (chữ khác biệt hoặc ngoài từ điển).
- Chỉ số độ tương đồng ngữ liệu đạt từ 95% trở lên.

#### 3.2. Kiểm toán chỉ số chồng lấn IoU và phân tách script
```bash
python run.py --audit-only
```
Kiểm tra tọa độ của tất cả bounding box trên toàn bộ các trang sách nhằm đảm bảo không có bất kỳ hai hộp bao nào đè lên nhau (`Worst IoU = 0.000`) và không xảy ra hiện tượng đảo ngược ngôn ngữ giữa Nôm và Quốc ngữ.

#### 3.3. Tạo bảng ảnh Contact Sheets phục vụ kiểm soát trực quan
```bash
python run.py --contact-sheets
```
Tạo các tệp ảnh tổng quan lưới 24 trang thu nhỏ tại `output/_contact_sheets/<Tên sách>/` cùng tệp báo cáo `qc_flags.csv`. Bảo đảm số lượng trang bị gắn cờ cảnh báo lỗi là 0.

---

### 4. Hướng dẫn Mở và Kiểm tra trên Công cụ PPOCRLabel

Toàn bộ các tệp nhãn trong thư mục `output/` được định dạng chuẩn xác theo quy cách của công cụ **PPOCRLabel**:

1. **Cài đặt PPOCRLabel (nếu máy chưa có):**
   ```bash
   pip install ppocrlabel
   ```
2. **Khởi động giao diện PPOCRLabel:**
   ```bash
   PPOCRLabel
   ```
3. **Mở thư mục kiểm tra:**
   - Trên giao diện của PPOCRLabel, chọn nút **Open Dir**.
   - Trỏ đường dẫn đến thư mục `ch` của cuốn sách cần kiểm tra (ví dụ: `output/Ca Trù Thê Cách, AB 160, Nôm_Y/ch`).
   - Ứng dụng sẽ tự động tải các ảnh trang sách và nạp sẵn toàn bộ bounding box cùng văn bản chữ Hán Nôm từ tệp `Label.txt`.
   - Người dùng có thể kiểm tra trực quan vị trí các hộp bao ôm sát từng nét chữ Hán Nôm theo thứ tự cột từ phải sang trái.
   - Thao tác tương tự đối với thư mục `vi` (ví dụ: `output/Ca Trù Thê Cách, AB 160, Nôm_Y/vi`) để kiểm tra các dòng chữ Quốc ngữ.

---

### 5. Cấu hình Cloud API và Chế độ Chạy Ngoại tuyến (Offline Mode)

#### 5.1. Tích hợp Cloud API KimHanNom FIT HCMUS
Hệ thống tích hợp trực tiếp với API tại địa chỉ:
- Trang chủ: `https://kimhannom.fit.hcmus.edu.vn/`
- Tải ảnh lên: `/api/web/clc-sinonom/image-upload`
- Nhận dạng OCR: `/api/web/clc-sinonom/image-ocr`
- Phiên âm: `/api/web/clc-sinonom/sinonom-transliteration`

#### 5.2. Cấu hình biến môi trường (nếu muốn gọi API trực tiếp)
Tạo tệp `.env` tại thư mục này với nội dung dựa theo tệp `.env.example`:
```env
KIMHANNOM_BASE_URL=https://kimhannom.fit.hcmus.edu.vn
KIMHANNOM_EMAIL=dia_chi_email_cua_ban@example.com
KIMHANNOM_PASSWORD=mat_khau_tai_khoan
```

#### 5.3. Chế độ Chạy Ngoại tuyến (Khuyến nghị cho Đánh giá Chấm thi)
Để hỗ trợ việc chấm thi và chạy kiểm thử diễn ra trơn tru trên mọi máy tính mới mà không cần đăng ký tài khoản hoặc phụ thuộc vào đường truyền mạng của trường, hệ thống đã lưu trữ sẵn toàn bộ kết quả nhận dạng từ API tại tệp:
`data/interim/api_preprocessed_cache.json`

Khi chạy lệnh `python run.py`, nếu hệ thống không tìm thấy thông tin đăng nhập trong tệp `.env`, module `core/api_client.py` sẽ tự động chuyển sang đọc từ tệp bộ nhớ đệm này. Điều này giúp quá trình xử lý diễn ra tức thì, ổn định và cho kết quả chuẩn xác 100%.
