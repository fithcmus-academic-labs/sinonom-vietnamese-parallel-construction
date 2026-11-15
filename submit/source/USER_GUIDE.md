# HUONG DAN VAN HANH VA CHAY DU AN (USER GUIDE)
## Do an Xay dung Ngu lieu Song song Han Nom va Quoc ngu

Tai lieu nay huong dan chi tiet cach thiet lap moi truong va van hanh he thong xu ly tu dau den cuoi tren mot may tinh moi hoan toan voi ma nguon da nop.

---

### 1. Thiet lap Moi truong tren May moi (Environment Setup)

#### Buoc 1.1: Kiem tra phien ban Python
He thong yeu cau Python 3.10, 3.11 hoac 3.12 (khuyen nghi Python 3.12).
```bash
python --version
# hoac tren Linux:
python3 --version
```

#### Buoc 1.2: Tao va kich hoat moi truong ao (Virtual Environment)
Tren Ubuntu / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Tren Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Buoc 1.3: Cai dat cac thu vien duoc co dinh (Pinned Dependencies)
Cai dat day du cac goi thu vien theo dung phien ban co dinh trong `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Thiet lap Co so Du lieu Tu dien Offline (Dictionary Setup)

Du lieu tu dien duoc tong hop tu hai nguon uy tin:
- **Unicode Consortium Unihan Database** (`Unihan_Readings.txt`)
- **Nom Na Viet** (`nom_viet.dict.yaml`)

Kho du lieu da duoc bien dich san tai `data/dicts/hannom_dict.pkl` va `data/dicts/hannom_dict.json`. Neu can bien dich lai tu file tho:
```bash
python scripts/build_dictionary.py
```
Ket qua ghi nhan 27.612 ky tu Han Nom voi 14.127 am doc phuc vu tra cuu ngoai tuyen voi toc do cao.

---

### 3. Huong dan Cau hinh va Goi API / Mo hinh (API & Model Configuration)

#### He thong Cloud API KimHanNom FIT HCMUS
He thong su dung cong cu chuyen dung cho chu Han Nom tai dia chi:
- URL: `https://kimhannom.fit.hcmus.edu.vn/`
- Endpoint OCR: `/api/web/clc-sinonom/image-ocr`
- Endpoint Transliteration: `/api/web/clc-sinonom/sinonom-transliteration`

#### Cau hinh bien moi truong (tuy chon neu chay live API)
Tao file `.env` tai thu muc goc cua ma nguon (tham khao `.env.example`):
```env
KIMHANNOM_BASE_URL=https://kimhannom.fit.hcmus.edu.vn
KIMHANNOM_EMAIL=your_email@example.com
KIMHANNOM_PASSWORD=your_password
```

#### Che do chay Ngoai tuyen (Offline Cache Mode)
Nham giup hoi dong cham thi va nguoi danh gia co the kiem thu lap tuc tren may moi ma khong can dang ky tai khoan hay ket noi internet, he thong da tich hop co che luu tru dem (caching) toan bo phan hoi cua API tai:
`data/interim/api_preprocessed_cache.json`
Khi he thong chay tren may moi, `api_client.py` se tu dong doc du lieu tu cache nay neu khong co tai khoan mang, dam bao quy trinh hoat dong 100% on dinh va nhanh chong.

---

### 4. Quy trinh Chay He thong qua CLI (Running the Pipeline)

He thong cung cap giao dien dong lenh thong nhat `main.py` de thuc thi cac cong doan:

#### Cach 1: Chay toan bo Pipeline tu dong
```bash
# Chay cho toan bo cac sach trong danh muc
python main.py pipeline --book all

# Hoac chay rieng cho mot cuon sach cu the
python main.py pipeline --book AB_160
```

#### Cach 2: Tinh chinh Bounding Box sat net chu va goi xac thuc API
```bash
python main.py deploy-bboxes
```
Lenh nay se doc cac lop muc da tien xu ly, co sat 4 canh hop bao vao net chu cua dung script, triet tieu chong lan (IoU = 0.000) va cap nhat file `Label.txt`, `fileState.txt`.

#### Cach 3: Xuat bang tinh dongs hang sang CSV va XLSX
```bash
python main.py export-tables
```
Tao ra file `<Tên sách>.csv` (UTF-8 with BOM) va `<Tên sách>.xlsx` co to mau chu theo chuan quy dinh (mau den cho chu khop nghia, mau do cho chu sai nghia hoac ngoai tu dien).

---

### 5. Kiem dinh va Danh gia Chat luong Du lieu (Verification & Audit)

#### Buoc 5.1: Kiem tra tinh tuan thu quy dinh nop bai
```bash
python main.py verify --book all
```
Script se kiem tra tu dong toan bo cac tieu chi:
1. Su hien dien cua cac thu muc bat buoc (`ch/`, `vi/`, `misc/`,...).
2. File PDF goc mang dung ten `<Tên sách>.pdf`.
3. File `ch/Label.txt` va `vi/Label.txt` co dung 4 diem toa do cho moi box.
4. File `ch/fileState.txt` va `vi/fileState.txt` co trang thai danh nhan la 1.
5. File `.csv` dung 4 cot yeu cau: `img_id`, `sentence_id`, `SinoNom_Char`, `ChuQN_txt`.
6. File `.xlsx` co dinh dang to mau chu ARGB chuan xac.
7. Chi so do tuong dong dat >= 95%.

#### Buoc 5.2: Kiem toan chong lan IoU va nhan dang script
```bash
python main.py audit
```
Kiem tra toan bo toa do tren 24 cuon sach, bao dam:
- `WORST IoU ACROSS ALL BOOKS: 0.000` (khong co bat ky 2 box nao de len nhau).
- Khong co hien tuong dao nguoc script giua Nom va Quoc ngu.

#### Buoc 5.3: Sinh Contact Sheet truc quan de kiem tra bang mat
```bash
python main.py contact-sheets
```
Tao cac tam anh contact sheet tai `output/_contact_sheets/<Tên sách>/sheet_XX.jpg` va file tong hop danh gia `qc_flags.csv`. Dam bao so luong trang bi gan co la `0`.

---

### 6. Huong dan Mo va Kiem tra tren Cong cu PPOCRLabel

De kiem tra truc quan cac bounding box va nhan nhan dang:
1. Cai dat PPOCRLabel:
   ```bash
   pip install ppocrlabel
   ```
2. Khoi dong ung dung:
   ```bash
   PPOCRLabel
   ```
3. Chon thu muc mo anh:
   - Nhan `Open Dir` tren giao dien PPOCRLabel.
   - Tro den thu muc `ch` (vi du: `output/Ca Trù Thê Cách, AB 160, Nôm_Y/ch`).
   - Ung dung se tu dong nap anh va file `Label.txt` trong thu muc do.
   - Cac bounding box chu Han Nom se hien thi chinh xac om sat tung cot chu.
4. Tuong tu doi voi thu muc `vi` (vi du: `output/Ca Trù Thê Cách, AB 160, Nôm_Y/vi`) de kiem tra bounding box cac dong Quoc ngu.
