"""
CLI Tool for Automated Book Layout and Type Classification
Classifies an uploaded or local PDF manuscript into taxonomy codes from Quy_định_đồ_án.pdf:
  T1, T1w, T2, T2w, T2b, Tnom, Tnomw, Tqn, Tref, Tx
"""

import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.classifier import BookClassifier


def main():
    parser = argparse.ArgumentParser(description="Classify PDF Manuscript Layout and Route Pipeline")
    parser.add_argument("--pdf", type=str, required=True, help="Path to manuscript PDF file")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)

    classifier = BookClassifier()
    result = classifier.classify(pdf_path)

    print("=" * 80)
    print(f"BOOK CLASSIFICATION & PIPELINE ROUTING REPORT")
    print(f"Target Document: {pdf_path.name}")
    print("=" * 80)
    print(f"Classified Book Code : {result['book_type']}")
    print(f"Classification Name  : {result['type_name']}")
    print(f"Description          : {result['description']}")
    print(f"Confidence Score     : {result['confidence'] * 100:.1f}%")
    print("-" * 80)
    print("Detected Layout Signals:")
    for sig in result["layout_details"].get("detected_signals", []):
        print(f"  * {sig}")
    print("-" * 80)
    rec_pipe = result["recommended_pipeline"]
    print(f"RECOMMENDED PIPELINE: {rec_pipe['name']} (ID: {rec_pipe['pipeline_id']})")
    print(f"Processing Strategy : {rec_pipe['strategy']}")
    print("Pipeline Execution Stages:")
    for step in rec_pipe.get("steps", []):
        print(f"  {step}")
    print("=" * 80)


if __name__ == "__main__":
    main()
