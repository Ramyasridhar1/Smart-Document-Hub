import argparse
import csv
import os
import sys

import db_compat as sqlite3
from dotenv import load_dotenv

# Ensure project root is importable when script is run from /training.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse shared extraction from app to keep behavior consistent.
from app import extract_text


def build_resume_ranker_dataset(db_url, output_csv, min_chars=30):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    conn = sqlite3.connect(db_url)
    try:
        c = conn.cursor()
        c.execute(
            '''
            SELECT rf.upload_id, rf.shortlist_fit, u.saved_path, u.summary
            FROM review_feedback rf
            JOIN uploads u ON u.id = rf.upload_id
            WHERE LOWER(COALESCE(rf.to_category, '')) = 'resume'
              AND LOWER(COALESCE(rf.shortlist_fit, '')) IN ('shortlist', 'reject')
            ORDER BY rf.created_at DESC
            '''
        )
        rows = c.fetchall()
    finally:
        conn.close()

    written = 0
    skipped = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label', 'upload_id', 'source_path', 'shortlist_fit'])

        for upload_id, shortlist_fit, saved_path, summary in rows:
            label = 1 if str(shortlist_fit).strip().lower() == 'shortlist' else 0

            text = ''
            if saved_path and os.path.exists(saved_path):
                text = (extract_text(saved_path, enable_ocr=True) or '').strip()

            if not text:
                text = (summary or '').strip()

            if len(text) < min_chars:
                skipped += 1
                continue

            writer.writerow([text, label, upload_id, saved_path or '', shortlist_fit])
            written += 1

    print("\n=== Resume Ranker Dataset Build Complete ===")
    print(f"Output: {output_csv}")
    print(f"Rows written: {written}")
    print(f"Rows skipped (short/empty): {skipped}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description='Build resume ranker dataset from review feedback labels.')
    parser.add_argument('--db-url', default=os.getenv('DATABASE_URL', 'postgresql://smartdoc:smartdoc@localhost:5432/smartdoc'))
    parser.add_argument('--output', default=os.path.join('data', 'resume_ranker_training.csv'))
    parser.add_argument('--min-chars', type=int, default=30)
    args = parser.parse_args()

    build_resume_ranker_dataset(args.db_url, args.output, args.min_chars)


if __name__ == '__main__':
    main()
