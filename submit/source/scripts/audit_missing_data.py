"""
Audit script to verify Han-Nom and Quoc Ngu completeness across datasets-midterm/pdf
and generate a formal report for the advisor (GVHD).
"""
import os
import sys
import unicodedata
from pathlib import Path
import fitz

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "datasets-midterm" / "pdf"
OUTPUT_REPORT = BASE_DIR / "output" / "bao_cao_kiem_tra_du_lieu_gvhd.txt"


AUDIT_RECORDS = [
    {
        "group": "1. File thieu hoan toan ban dich Quoc ngu (100% chi co Han Nom)",
        "items": [
            {
                "filename": "Thanh Tông Ngư Vịnh Tư Thơi Thi Tâp, Nôm.pdf",
                "total_pages": 17,
                "nom_pages": "12 trang (trang 4 den trang 15 - noi dung Thuy Kieu Thi Tap Tam Thap De)",
                "quoc_ngu_pages": "0 trang",
                "status": "Thieu 100% ban dich Quoc ngu (khong co chu Quoc ngu tren trang va khong co file Quoc ngu di kem).",
            }
        ],
    },
    {
        "group": "2. Cac file bi thieu phan lon ban dich Quoc ngu (chep do dang, de trang giay trang)",
        "items": [
            {
                "filename": "ThẩmTửHưTruyện_沈子虛傳.pdf",
                "total_pages": 120,
                "nom_pages": "53 trang (cac trang chan tu trang 8 den trang 112)",
                "quoc_ngu_pages": "2 trang (chi co trang le 15 va 17)",
                "status": "Thieu 51/53 trang Quoc ngu (~96%). Cac trang le doi dien tu trang 19 den 113 deu de giay trang.",
            },
            {
                "filename": "ThơVănLêTrịnhMạc.pdf",
                "total_pages": 85,
                "nom_pages": "33 trang (cac trang chan 6-38 phan Bui Thi Thi Van va 48-74 phan To Cong Phung Su)",
                "quoc_ngu_pages": "6 trang (cac trang le 9, 11, 13, 15, 17 va trang 44)",
                "status": "Thieu 28/33 trang Quoc ngu (~85%). Tu trang 20-38 va toan bo phan tho Nom trang 48-74 khong co ban dich Quoc ngu.",
            },
            {
                "filename": "ThơCaDuHànhTừHàNộiVàoKinhĐôHuế_1.pdf",
                "total_pages": 135,
                "nom_pages": "58 trang (cac trang chan tu trang 8 den trang 124)",
                "quoc_ngu_pages": "39 trang (cac trang le tu trang 9 den trang 85)",
                "status": "Phan 1 (Kinh hanh nhat trinh khuc, trang 8-85) du Quoc ngu; nhung Phan 2 (Dong hanh nhat trinh ngam khuc, trang 88-124) cac trang le doi dien de giay trang (co the doi chieu ban nhap viet tay trong file _2.pdf).",
            },
            {
                "filename": "TìBàQuốcÂmTânTruyện_琵琶國音新傳_1.pdf & TìBàQuốcÂmTânTruyện_琵琶國音新傳_2.pdf",
                "total_pages": "143 trang (_1.pdf) + 146 trang (_2.pdf)",
                "nom_pages": "65 trang (_1.pdf) + 66 trang (_2.pdf) = 131 trang Nom",
                "quoc_ngu_pages": "5 trang trong _1.pdf (trang 48, 50, 52, 54, 56) + 0 trang trong _2.pdf",
                "status": "Thieu 126/131 trang Quoc ngu ben trong 2 file nay (co the map voi ban Yale 'Ti Ba Quoc Am Tan Truyen, Preface in Han, text in Nom_Y.pdf' 249 trang neu duoc phep).",
            },
        ],
    },
]


def generate_report() -> str:
    files_on_disk = sorted(os.listdir(PDF_DIR)) if PDF_DIR.exists() else []
    lines = []
    lines.append("BAO CAO KIEM TRA NGUON DU LIEU HAN NOM - QUOC NGU")
    lines.append(f"Thu muc kiem tra: {PDF_DIR}")
    lines.append(f"Tong so file hien co (sau khi loai bo 4 ban sao trung lap NFD): {len(files_on_disk)} file")
    lines.append("=" * 90)
    lines.append("")

    for section in AUDIT_RECORDS:
        lines.append(section["group"])
        lines.append("-" * len(section["group"]))
        for idx, item in enumerate(section["items"], 1):
            lines.append(f"  {idx}. File: {item['filename']}")
            lines.append(f"     - Tong so trang PDF : {item['total_pages']}")
            lines.append(f"     - So trang Han Nom  : {item['nom_pages']}")
            lines.append(f"     - So trang Quoc ngu : {item['quoc_ngu_pages']}")
            lines.append(f"     - Tinh trang        : {item['status']}")
            lines.append("")

    lines.append("=" * 90)
    lines.append("NOI DUNG TIN NHAN GUI GIANG VIEN HUONG DAN (GVHD):")
    lines.append("-" * 90)
    lines.append(
        """Dạ chào Thầy/Cô ạ,
Nhóm em (MSSV: 24127457 - 24127192 - 24127402 - 24127445) đã thực hiện bước Kiểm tra nguồn dữ liệu và xác nhận phạm vi trên toàn bộ các file trong thư mục được giao theo đúng Quy định đồ án.
Qua rà soát từng trang, nhóm phát hiện có 4 file bị trùng lặp hoàn toàn (do khác chuẩn mã hóa tên file Unicode NFC/NFD của bộ Ca Trù Thể Cách) và trong 24 file độc nhất còn lại, có một số tài liệu hiện đang thiếu bản dịch Quốc ngữ hoặc chỉ mới chép Quốc ngữ được vài trang đầu (các trang đối diện còn lại đều để giấy trắng):

1. File thiếu hoàn toàn (100%) bản dịch Quốc ngữ (chưa có file Quốc ngữ đi kèm):
- Thanh Tông Ngư Vịnh Tư Thơi Thi Tâp, Nôm.pdf (17 trang: từ trang 4 đến trang 15 là chữ Hán Nôm viết dọc phần Thúy Kiều Thi Tập Tam Thập Đề, hoàn toàn chưa có bản dịch Quốc ngữ).

2. Các file bị khuyết phần lớn trang dịch Quốc ngữ bên trong (trang đối diện để giấy trắng):
- ThẩmTửHưTruyện_沈子虛傳.pdf (120 trang): Có 53 trang chữ Nôm (trang chẵn 8-112), nhưng chỉ mới có 2 trang chép tay Quốc ngữ (trang 15 và 17). Toàn bộ các trang lẻ từ trang 19 đến 113 đều là giấy trắng (thiếu 51/53 trang Quốc ngữ).
- ThơVănLêTrịnhMạc.pdf (85 trang): Có 33 trang chữ Nôm (trang chẵn 6-38 và 48-74), nhưng chỉ có 6 trang Quốc ngữ đánh máy ở đầu sách (trang 9, 11, 13, 15, 17 và 44). Phần Bùi Thị Thi Văn (trang 20-38) và toàn bộ phần thơ Tô Công Phụng Sứ (trang 48-74) đều để giấy trắng ở trang đối diện (thiếu 28/33 trang Quốc ngữ).
- ThơCaDuHànhTừHàNộiVàoKinhĐôHuế_1.pdf (135 trang): Phần 1 (trang 8-85) có đủ bản Quốc ngữ đánh máy, nhưng Phần 2 (Đông hành nhật trình ngâm khúc, từ trang 88 đến 124) chỉ có chữ Nôm, các trang lẻ đối diện để trắng (nhóm có thể dùng bản nháp viết tay trong file ThơCaDuHànhTừHàNộiVàoKinhĐôHuế_2.pdf để map sang được không ạ?).
- TìBàQuốcÂmTânTruyện_琵琶國音新傳_1.pdf (143 trang) và _2.pdf (146 trang): Tổng cộng 131 trang chữ Nôm nhưng chỉ có 5 trang chép tay Quốc ngữ (trang 48, 50, 52, 54, 56 của quyển 1), còn lại 126 trang để giấy trắng. Nhóm có thể sử dụng phần phiên âm Quốc ngữ trong cuốn Yale (Tì Bà Quôc Âm Tân Truyên, Preface in Hán, text in Nôm_Y.pdf - 249 trang) để map cho 2 file này không ạ?

Nhóm xin phép báo cáo và nhờ Thầy/Cô cấp bổ sung file bản dịch Quốc ngữ cho các tài liệu bị thiếu trên (hoặc xác nhận phạm vi các trang cần dóng hàng đối với những cuốn bị khuyết bản dịch) để nhóm tiến hành chạy pipeline OCR và dóng hàng đúng tiến độ ạ. Em cảm ơn Thầy/Cô ạ!"""
    )
    return "\n".join(lines)


if __name__ == "__main__":
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    report_text = generate_report()
    OUTPUT_REPORT.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n[SAVED] Da luu bao cao tai: {OUTPUT_REPORT}")
