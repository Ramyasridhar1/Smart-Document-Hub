import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

# Ensure project root is importable when script is run from /training.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse the existing OCR/text extraction stack from the app.
from app import extract_text


SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp', '.pdf', '.txt', '.doc', '.docx'}


def iter_files(root):
    for base, _, files in os.walk(root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTS:
                yield os.path.join(base, name)


def build_dataset(label_dirs, output_csv, min_chars=30, max_per_label=0, enable_ocr=True, failed_log_path=None):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if failed_log_path:
        os.makedirs(os.path.dirname(failed_log_path), exist_ok=True)

    written = 0
    skipped = 0
    per_label = defaultdict(int)
    failed_rows = []

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'text', 'source_path', 'text_length', 'ocr_enabled', 'created_at'])

        for label, root in label_dirs.items():
            if not os.path.exists(root):
                print(f"[WARN] Missing path for label '{label}': {root}")
                continue

            for path in iter_files(root):
                if max_per_label > 0 and per_label[label] >= max_per_label:
                    break

                text = (extract_text(path, enable_ocr=enable_ocr) or '').strip()
                if len(text) < min_chars:
                    skipped += 1
                    failed_rows.append([label, path, len(text), f'min_chars<{min_chars}'])
                    continue

                writer.writerow([label, text, path, len(text), int(bool(enable_ocr)), datetime.now(timezone.utc).isoformat()])
                written += 1
                per_label[label] += 1

                if written % 100 == 0:
                    print(f"[INFO] Written {written} rows...")

    print("\n=== Dataset Build Complete ===")
    print(f"Output: {output_csv}")
    print(f"Rows written: {written}")
    print(f"Rows skipped (short/empty): {skipped}")
    for label, count in sorted(per_label.items()):
        print(f"  - {label}: {count}")

    if failed_log_path:
        with open(failed_log_path, 'w', newline='', encoding='utf-8') as ff:
            fw = csv.writer(ff)
            fw.writerow(['label', 'source_path', 'text_length', 'reason'])
            fw.writerows(failed_rows)
        print(f"Failed extraction log: {failed_log_path} ({len(failed_rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description='Build document classification CSV from image/PDF folders using shared OCR/text extraction.')
    parser.add_argument('--invoice-dir', default=os.path.join('data', 'invoice'))
    parser.add_argument('--receipt-dir', default=os.path.join('data', 'receipt'))
    parser.add_argument('--resume-dir', default=os.path.join('data', 'Resume-data'))
    parser.add_argument('--output', default=os.path.join('training', 'data_generated', 'document_ocr_dataset.csv'))
    parser.add_argument('--failed-log', default=os.path.join('training', 'reports', 'document_ocr_failed_rows.csv'))
    parser.add_argument('--min-chars', type=int, default=30)
    parser.add_argument('--max-per-label', type=int, default=0, help='0 means unlimited')
    parser.add_argument('--disable-ocr', action='store_true')
    args = parser.parse_args()

    label_dirs = {
        'invoice': args.invoice_dir,
        'receipt': args.receipt_dir,
        'resume': args.resume_dir,
    }

    build_dataset(
        label_dirs=label_dirs,
        output_csv=args.output,
        min_chars=args.min_chars,
        max_per_label=args.max_per_label,
        enable_ocr=not args.disable_ocr,
        failed_log_path=args.failed_log,
    )


if __name__ == '__main__':
    main()
