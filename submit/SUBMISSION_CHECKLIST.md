# QUY TRINH DONG GOI VA NOP BAI (SUBMISSION CHECKLIST)
## Theo quy dinh tai guide/Quy_dinh_do_an.pdf va guide/files

---

### 1. Cau truc Thu muc Nop bai Chuan (Theo 03_quy_dinh_du_lieu_nop.md)

Khi nop bai, toan bo san pham duoc dong goi thanh file zip dat ten theo cu phap do giao vien phuc trach yeu cau:
`<MSSV1_MSSV2_MSSV3_MSSV4>.zip`

Cau truc ben trong file zip:
```
<MSSV1_MSSV2_MSSV3_MSSV4>.zip/
|-- <Tên sách>/
|   |-- ch/
|   |   |-- <Tên sách>.<stt>.jpg
|   |   |-- Label.txt
|   |   `-- fileState.txt
|   |-- vi/
|   |   |-- <Tên sách>.<stt>.jpg
|   |   |-- Label.txt
|   |   `-- fileState.txt
|   |-- intro/                          (neu co)
|   |-- ref/                            (neu co)
|   |-- toc/                            (neu co)
|   |-- blank/                          (neu co)
|   |-- misc/
|   |   `-- <Tên sách>.<stt>.jpg
|   |-- <Tên sách>.pdf
|   |-- <Tên sách>_updated.pdf          (neu co tien xu ly sua anh)
|   |-- <Tên sách>.xlsx                 (du lieu dong hang co to mau)
|   `-- <Tên sách>.csv                  (du lieu dong hang UTF-8 BOM)
|-- <MSSV1_MSSV2_MSSV3_MSSV4>_report.pdf (File bao cao PDF)
|-- source.zip                           (File ma nguon du an - san co tai submit/source.zip)
`-- extras.zip                           (Tuy chon: anh contact sheets hoac du lieu bo sung)
```

---

### 2. Cac Thanh phan San sang trong Thu muc submit/

| Thu muc / Tep tin | Vi tri | Mo ta & Trang thai |
|---|---|---|
| **`submit/source/`** | [submit/source/](file:///D:/User/nghonamjiwoo/Natural%20Language%20Processing/ca-tru-the-cach-nomocr/submit/source) | Thu muc ma nguon hoan chinh, doc lap (core, scripts, data, requirements.txt pinned, README.md, USER_GUIDE.md, main.py). |
| **`submit/source.zip`** | [submit/source.zip](file:///D:/User/nghonamjiwoo/Natural%20Language%20Processing/ca-tru-the-cach-nomocr/submit/source.zip) | Goi ma nguon duoc nen theo chuan `ZIP_DEFLATED` (~5.8 MB), san sang dat truc tiep vao file zip nop bai. |
| **Du lieu cac cuon sach** | [output/](file:///D:/User/nghonamjiwoo/Natural%20Language%20Processing/ca-tru-the-cach-nomocr/output) | 24 bo sach da xu ly day du: `ch/`, `vi/`, `misc/`, `<Tên sách>.pdf`, `Label.txt`, `fileState.txt`, `.csv`, `.xlsx`. |

---

### 3. Bang Kiem tra Tinh Hop le (Compliance Verification Matrix)

| Tieu chi | Quy dinh | Ket qua dat duoc | Trang thai |
|---|---|---|---|
| **Dinh dang Bounding Box** | Moi box dung 4 diem toa do `[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]` | 100% tren ca 47.594 hop | **DAT (PASS)** |
| **Dinh dang PPOCRLabel** | File `Label.txt` va `fileState.txt` dung cu phap Tab-delimited | 100% cac thu muc `ch/` va `vi/` | **DAT (PASS)** |
| **Khong gop hop Nom & Quoc ngu** | Bounding box khong duoc chua ca 2 ngon ngu | 0 hop bi gop (tach lop roi nhau) | **DAT (PASS)** |
| **Do chong lan (IoU)** | Max IoU toi thieu giua cac hop tren cung trang va xuyen thu muc | Worst IoU = 0.000 tren 24 cuon sach | **DAT (PASS)** |
| **Trang bi gan co loi (QC Flags)** | Khong con trang nao bi loi no_boxes hoac outlier | 0 trang bi gan co trong `qc_flags.csv` | **DAT (PASS)** |
| **Do tuong dong dich thuat** | Yeu cau toi thieu $\ge 95\%$ | Dat tu 99.17% den 99.46% | **DAT (PASS)** |
| **File bang tinh CSV** | 4 cot: `img_id`, `sentence_id`, `SinoNom_Char`, `ChuQN_txt`, UTF-8 BOM | 100% dung tieu chuan, khong lech cot | **DAT (PASS)** |
| **File bang tinh XLSX** | To mau ARGB: den (`FF000000`) khop nghia, do (`FFFF0000`) sai/ngoai tu dien | 100% dung dinh dang mau | **DAT (PASS)** |
| **Dependencies co dinh** | Tat ca cac goi thu vien phai duoc pin phien ban (`==`) | 10/10 thu vien duoc pin chinh xac | **DAT (PASS)** |
| **File README.md & USER_GUIDE.md** | Mo ta day du kien truc, moi truong chay (Ubuntu/Windows), huong dan may moi | Day du tai `submit/source/` | **DAT (PASS)** |
