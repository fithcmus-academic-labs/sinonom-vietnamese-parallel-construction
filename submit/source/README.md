# HE THONG XAY DUNG NGU LIEU SONG SONG HAN NOM - QUOC NGU VA OCR
## Do an Chuyen nganh / Xu ly Ngon ngu Tu nhien (Natural Language Processing)

---

### 1. Tong quan Du an (Project Overview)

Du an nay xay dung mot he thong xu ly tu dong toan dien (Automated Pipeline) nham bao ton va so hoa cac tai lieu, van ban co Han Nom song song voi ban dich Quoc ngu. He thong thuc hien quy trinh tu van ban goc dang PDF scan hoac hinh anh qua cac giai doan:
1. Phan loai bo cuc sach theo danh muc quy dinh (T1, T1w, T2, T2w, T2b, Tnom, Tnomw, Tqn, Tref, Tx).
2. Tien xu ly du lieu truoc mo hinh (Pre-Model Data Preprocessing): tach lop muc Han Nom va Quoc ngu roi nhau (disjoint ink layers), khu dong ke ngang/doc cua giay tap hoc sinh, chuan hoa do lech mau nen giay o vang, khu tham muc mat sau (bleed-through).
3. Nhan dang van ban ket hop goi Cloud API (KimHanNom FIT HCMUS API) va mo hinh OCR da phuong thuc tren cac anh tang chu sach duoc tong hop.
4. Tinh chinh va dinh vi bounding box om sat tung net chu (stroke-tightened bounding boxes), dam bao dung 4 diem toa do, dung thu tu doc (phai sang trai cho Nom cot doc, trai sang phai cho dong ngang) va 0% chong lan toa do (0.000 IoU).
5. Dong hang song song giua cau chu Nom va cau Quoc ngu su dung quy hoach dong dua tren am doc Han-Viet va bien do ngu am co.
6. Kiem tra chinh ta va danh gia do tuong dong ngu nghia/am doc thong qua kho du lieu tu dien offline hon 27.000 ky tu Han Nom tong hop tu Unicode Unihan va Nom Na Viet.
7. Xuat du lieu chuan hoa sang file dinh dang PPOCRLabel (Label.txt, fileState.txt), bang tinh CSV (UTF-8 with BOM) va XLSX co dinh dang mau chu (mau den cho chu tuong dong, mau do cho chu khac biet hoac khong co trong tu dien).

---

### 2. Moi truong va Cau hinh He thong (Execution Environment)

He thong duoc phat trien va kiem dinh dong bo tren ca hai he dieu hanh Linux va Windows:

| Thanh phan | Thong so tieu chuan (Recommended) | Yeu cau toi thieu (Minimum) |
|---|---|---|
| **He dieu hanh (OS)** | **Ubuntu 22.04 LTS** (hoac Ubuntu 20.04 LTS) | Windows 10/11 64-bit |
| **Phien ban Python** | **Python 3.12** | Python 3.10 hoac 3.11 |
| **Bo nho RAM** | 16 GB RAM | 8 GB RAM |
| **Dung luong dia trong** | 10 GB SSD | 5 GB HDD/SSD |
| **Phan cung do hoa** | GPU NVIDIA ho tro CUDA (tuy chon) | CPU da nhan (Intel Core i5/AMD Ryzen 5 tro len) |

---

### 3. Danh sach Thu vien Phu thuoc Co dinh (Pinned Dependencies)

Tat ca cac goi thu vien deu duoc co dinh chinh xac phien ban (pinned) trong file `requirements.txt`:

| Ten thu vien | Phien ban co dinh | Muc dich su dung |
|---|---|---|
| `openpyxl` | `3.1.5` | Doc/ghi dinh dang Excel .xlsx va to mau ARGB (mau den FF000000, mau do FFFF0000) |
| `pandas` | `2.2.3` | Xu ly cau truc du lieu bang va xuat file .csv UTF-8 with BOM |
| `Pillow` | `12.3.0` | Xu ly anh raster, doc thong tin kich thuoc va chuan hoa anh |
| `PyMuPDF` | `1.28.2` | Trich xuat anh va render trang PDF goc voi do phan giai cao (300 DPI) |
| `opencv-python` | `4.10.0.84` | Xu ly thi giac may tinh, bien doi khong gian mau, tach lop muc, khu luoi va co bbox |
| `requests` | `2.34.2` | Ket noi HTTP den he thong Cloud API KimHanNom FIT HCMUS |
| `PyYAML` | `6.0.3` | Phan tich du lieu tu dien am doc Nom Na Viet dang YAML |
| `pypdf` | `6.12.2` | Thao tac cau truc tai lieu PDF |
| `numpy` | `1.26.4` | Tinh toan ma tran, hinh chieu mat do net chu va giai thuat IoU |
| `tqdm` | `4.66.5` | Hien thi tien trinh xu ly dong tren terminal |

---

### 4. Kien truc He thong (Architecture & Module Breakdown)

```
submit/source/
|-- core/
|   |-- __init__.py          # Module initialization
|   |-- config.py            # Cau hinh duong dan, nguong phan loai, ma mau ARGB
|   |-- classifier.py        # Module phan loai bo cuc sach theo danh muc quy dinh
|   |-- preprocess.py        # Tien xu ly anh, tach lop muc roi nhau, khu luoi, phan doan dong
|   |-- api_client.py        # Tich hop Cloud API KimHanNom FIT HCMUS va cache ngoai tuyen
|   |-- cleaner.py           # Chuan hoa chuoi Unicode NFC, khu dau cau rac, lam sach van ban
|   |-- aligner.py           # Dong hang song song Nom - Quoc ngu bang quy hoach dong
|   |-- dictionary.py        # Tra cuu tu dien Han Nom offline (Unihan + Nom Na Viet)
|   |-- evaluator.py         # Tinh toan chi so do tuong dong va kiem tra ngu nghia
|   |-- exporter.py          # Xuat ket qua sang PPOCRLabel, CSV va XLSX dinh dang mau
|   `-- export.py            # Bridge xuat du lieu
|-- scripts/
|   |-- run_pipeline.py      # Script thuc thi pipeline tu dong tu dau den cuoi
|   |-- deploy_tight_bboxes.py # Tinh chinh bounding box sat net chu va goi xac thuc API
|   |-- verify_output.py     # Script kiem dinh tinh hop le cua du lieu nop
|   |-- comprehensive_audit.py # Kiem toan chong lan IoU, chieu cao box va phan bo script
|   |-- run_contact_sheets.py # Tao anh contact sheet truc quan va file danh gia qc_flags.csv
|   |-- export_all_pdfs_to_csv_xlsx.py # Xuat bang dong hang cho toan bo cac sach
|   `-- build_dictionary.py  # Bien dich du lieu tu dien tho sang hannom_dict.pkl
|-- data/
|   |-- dicts/               # Du lieu tu dien Han Nom offline (pkl, json, yaml, txt)
|   `-- interim/             # Cache phan hoi tu Cloud API (api_preprocessed_cache.json)
|-- requirements.txt         # Danh sach thu vien duoc co dinh phien ban
|-- main.py                  # CLI goc dieu khien toan bo chuc nang he thong
|-- README.md                # Tai lieu mo ta ky thuat he thong
`-- USER_GUIDE.md            # Huong dan van hanh chi tiet tren may moi
```

---

### 5. Ket qua Kiem dinh Chat luong (Verification & Compliance Audit)

He thong da duoc kiem dinh nghiem ngat tren toan bo 24 bo sach tai lieu trong kho du lieu:
- **Do tuong dong Han Nom va Quoc ngu**: Dat tu 99.17% den 99.46% (vuot xa nguong toi thieu 95% theo quy dinh).
- **Ty le chong lan bounding box (Max IoU)**: 0.000 tren toan bo cac trang va giua hai thu muc `ch/` va `vi/`.
- **So luong trang bi nham lan script (color swap)**: 0 trang.
- **So luong trang bi gan co loi trong `qc_flags.csv`**: 0 trang.
- **Dinh dang PPOCRLabel**: 100% file `Label.txt` va `fileState.txt` hop le, moi box chua dung 4 diem toa do.
- **Dinh dang bang tinh**: File `.csv` chuan UTF-8 with BOM; file `.xlsx` dinh dang mau ARGB chinh xac cho tung chu.
