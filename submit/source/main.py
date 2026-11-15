"""
Unified Command-Line Interface for the Sino-Nom and Quoc Ngu Parallel Corpus Pipeline.
Conforms strictly to the project submission requirements.
"""
import sys
import argparse
from pathlib import Path

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_pipeline_cmd(args):
    """Run full automated pipeline."""
    from scripts.run_pipeline import run_pipeline
    book = getattr(args, "book", "all")
    print(f"Running automated pipeline for: {book}")
    run_pipeline(book)


def deploy_bboxes_cmd(args):
    """Deploy stroke-tightened bounding boxes with API verification."""
    from scripts.deploy_tight_bboxes import main as deploy_main
    print("Running stroke-aware bounding box deployment and API verification...")
    deploy_main()


def verify_cmd(args):
    """Verify output deliverable compliance."""
    from scripts.verify_output import verify_book_output
    from core.config import BOOKS
    book = getattr(args, "book", "all")
    print(f"Verifying deliverables for: {book}")
    if book == "all":
        success = True
        for k in BOOKS.keys():
            if not verify_book_output(k):
                success = False
        sys.exit(0 if success else 1)
    else:
        success = verify_book_output(book)
        sys.exit(0 if success else 1)


def audit_cmd(args):
    """Run comprehensive IoU and script audit across all output books."""
    from scripts.comprehensive_audit import main as audit_main
    print("Running comprehensive audit (IoU, heights, scripts)...")
    audit_main()


def contact_sheets_cmd(args):
    """Generate quality control contact sheets and qc_flags.csv."""
    from scripts.run_contact_sheets import main as sheets_main
    print("Generating visual contact sheets and QC flags...")
    sheets_main()


def export_tables_cmd(args):
    """Export alignment tables to CSV and formatted XLSX."""
    from scripts.export_all_pdfs_to_csv_xlsx import main as export_main
    print("Exporting alignment tables to CSV and formatted XLSX...")
    export_main()


def main():
    parser = argparse.ArgumentParser(
        description="Sino-Nom and Quoc Ngu Parallel Corpus Construction & OCR Pipeline CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available pipeline commands")

    # Command: pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run end-to-end automated pipeline")
    p_pipe.add_argument("--book", type=str, default="all", help="Target book (AB_160, AB_431, AB_456, NOM_Y, all)")
    p_pipe.set_defaults(func=run_pipeline_cmd)

    # Command: deploy-bboxes
    p_dep = subparsers.add_parser("deploy-bboxes", help="Deploy stroke-tightened bboxes with API verification")
    p_dep.set_defaults(func=deploy_bboxes_cmd)

    # Command: verify
    p_ver = subparsers.add_parser("verify", help="Verify deliverables against submission specifications")
    p_ver.add_argument("--book", type=str, default="all", help="Target book to verify (default: all)")
    p_ver.set_defaults(func=verify_cmd)

    # Command: audit
    p_aud = subparsers.add_parser("audit", help="Run IoU overlap and script classification audit")
    p_aud.set_defaults(func=audit_cmd)

    # Command: contact-sheets
    p_cs = subparsers.add_parser("contact-sheets", help="Generate inspection contact sheets")
    p_cs.set_defaults(func=contact_sheets_cmd)

    # Command: export-tables
    p_exp = subparsers.add_parser("export-tables", help="Export aligned pairs to CSV and colored XLSX")
    p_exp.set_defaults(func=export_tables_cmd)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
