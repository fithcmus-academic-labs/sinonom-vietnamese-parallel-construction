"""
Diem vao chay tu dong chinh cua du an: Do an xay dung ngu lieu song song Han Nom - Quoc ngu.
Chi can chay:
    python run.py
He thong se tu dong doc du lieu PDF, tien xu ly tach lop muc, goi xac thuc API,
dinh vi bounding box sat net chu, dong hang van ban va xuat toan bo ra thu muc output/.
"""

import os
import sys
import argparse
from pathlib import Path

# Thiet lap duong dan goc
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.config import OUTPUT_DIR, DATASETS_MIDTERM_DIR, PDF_DIR, BOOKS


def print_banner(text: str):
    print("=" * 80)
    print(text)
    print("=" * 80)


def execute_pipeline(book: str = "all", run_verification: bool = True):
    print_banner("KHOI DONG HE THONG XU LY TU DONG HAN NOM - QUOC NGU")
    print(f"Thu muc dau vao (PDF): {PDF_DIR}")
    print(f"Thu muc dau ra (Output): {OUTPUT_DIR}")
    print(f"Doi tuong xu ly: {book}")
    print("-" * 80)

    # Buoc 1: Chay Pipeline xu ly anh, phan loai va dong hang
    print("\n[BUOC 1/4] Tien xu ly tai lieu va dong hang tu dong...")
    from scripts.run_pipeline import run_pipeline
    run_pipeline(book)

    # Buoc 2: Trien khai Bounding Box sat net chu va xac thuc API KimHanNom
    print("\n[BUOC 2/4] Dinh vi Bounding Box om sat net chu va xac thuc qua API KimHanNom...")
    from scripts.deploy_tight_bboxes import main as deploy_main
    deploy_main()

    # Buoc 3: Xuat du lieu bang tinh CSV va XLSX to mau ARGB chuan quy dinh
    print("\n[BUOC 3/4] Xuat bang tinh dong hang sang dinh dang CSV va XLSX co to mau...")
    from scripts.export_all_pdfs_to_csv_xlsx import main as export_main
    export_main()

    # Buoc 4: Tham dinh tu dong chat luong va tinh hop le cua thu muc output
    if run_verification:
        print("\n[BUOC 4/4] Tham dinh tinh hop le cua cac tep trong thu muc output/...")
        from scripts.verify_output import verify_book_output
        all_passed = True
        if book == "all":
            for k in BOOKS.keys():
                if not verify_book_output(k):
                    all_passed = False
        else:
            all_passed = verify_book_output(book)

        print("-" * 80)
        if all_passed:
            print("KET QUA: TAT CA CAC TIEU CHUAN KIEM DINH DEU DAT CHUAN [THANH CONG]")
        else:
            print("CANH BAO: MOT SO TIEU CHUAN CAN DUOC KIEM TRA LAI")

    print_banner(f"HOAN TAT: Toan bo ket qua da duoc xu ly va luu tai: {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Chuong trinh chay tu dong toan bo he thong tao ra thu muc output"
    )
    parser.add_argument(
        "--book",
        type=str,
        default="all",
        help="Cuon sach can xu ly (AB_160, AB_431, AB_456, NOM_Y hoac all). Mac dinh: all",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Bo qua buoc kiem dinh tinh hop le sau khi chay xong",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Chi chay kiem toan chi so IoU va phan tach script tren thu muc output hien co",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Chi chay tham dinh tinh hop le tren thu muc output hien co",
    )
    parser.add_argument(
        "--contact-sheets",
        action="store_true",
        help="Chi tao cac anh contact sheets de kiem tra truc quan",
    )

    args = parser.parse_args()

    if args.audit_only:
        from scripts.comprehensive_audit import main as audit_main
        audit_main()
        return

    if args.verify_only:
        from scripts.verify_output import verify_book_output
        if args.book == "all":
            for k in BOOKS.keys():
                verify_book_output(k)
        else:
            verify_book_output(args.book)
        return

    if args.contact_sheets:
        from scripts.run_contact_sheets import main as sheets_main
        sheets_main()
        return

    execute_pipeline(book=args.book, run_verification=not args.skip_verify)


if __name__ == "__main__":
    main()
