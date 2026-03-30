import argparse
import os
import subprocess
import sys


def run_cmd(cmd):
    print("\n[RUN]", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Build OCR dataset from labeled image/PDF folders and train document classifier.")
    parser.add_argument('--invoice-dir', default=os.path.join('data', 'invoice'))
    parser.add_argument('--receipt-dir', default=os.path.join('data', 'receipt'))
    parser.add_argument('--resume-dir', default=os.path.join('data', 'Resume-data'))
    parser.add_argument('--base-csv', default=os.path.join('data', 'train_data.csv'))
    parser.add_argument('--generated-csv', default=os.path.join('training', 'data_generated', 'document_ocr_dataset.csv'))
    parser.add_argument('--failed-log', default=os.path.join('training', 'reports', 'document_ocr_failed_rows.csv'))
    parser.add_argument('--output-model', default=os.path.join('training', 'models', 'tfidf_logreg_trained.joblib'))
    parser.add_argument('--min-chars', type=int, default=30)
    parser.add_argument('--max-per-label', type=int, default=0)
    parser.add_argument('--disable-ocr', action='store_true')
    parser.add_argument('--map-receipt-to', default=None)
    parser.add_argument('--test-size', type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(os.path.join('training', 'data_generated'), exist_ok=True)
    os.makedirs(os.path.join('training', 'models'), exist_ok=True)
    os.makedirs(os.path.join('training', 'reports'), exist_ok=True)

    build_cmd = [
        sys.executable,
        os.path.join('training', 'build_doc_dataset_from_images.py'),
        '--invoice-dir', args.invoice_dir,
        '--receipt-dir', args.receipt_dir,
        '--resume-dir', args.resume_dir,
        '--output', args.generated_csv,
        '--failed-log', args.failed_log,
        '--min-chars', str(args.min_chars),
        '--max-per-label', str(args.max_per_label),
    ]
    if args.disable_ocr:
        build_cmd.append('--disable-ocr')

    train_cmd = [
        sys.executable,
        os.path.join('training', 'train_doc_classifier.py'),
        '--base', args.base_csv,
        '--generated', args.generated_csv,
        '--output', args.output_model,
        '--test-size', str(args.test_size),
    ]
    if args.map_receipt_to:
        train_cmd.extend(['--map-receipt-to', args.map_receipt_to])

    run_cmd(build_cmd)
    run_cmd(train_cmd)

    print("\n[OK] Pipeline complete")
    print(f"Generated dataset: {args.generated_csv}")
    print(f"Model artifact: {args.output_model}")
    print(f"Failed extraction log: {args.failed_log}")
    print(f"Training report: {os.path.join('training', 'reports', 'doc_classifier_report.txt')}")


if __name__ == '__main__':
    main()
