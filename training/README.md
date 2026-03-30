# Training Workspace (OCR + NLP/Text Extraction)

This folder is a separate model-training workspace. It converts image/PDF datasets into text CSV using the app OCR/text extraction stack, then trains models from generated CSV.

Generated artifacts are stored under:
- `training/data_generated/` (dataset CSV)
- `training/models/` (trained model artifacts)
- `training/reports/` (failed-row logs and training metrics)

## 1) One-command pipeline (recommended)

```bash
python training/run_training_pipeline.py \
  --invoice-dir data/invoice \
  --receipt-dir data/receipt \
  --resume-dir data/Resume-data \
  --max-per-label 500
```

This will:
1. build OCR dataset CSV
2. train document classifier
3. save reports and artifacts in the separate `training/` workspace

## 2) Build document dataset from images/PDFs

Uses the shared app extraction stack (`extract_text`) with OCR support.

```bash
python training/build_doc_dataset_from_images.py \
  --invoice-dir data/invoice \
  --receipt-dir data/receipt \
  --resume-dir data/Resume-data \
  --output training/data_generated/document_ocr_dataset.csv \
  --min-chars 30
```

Optional controls:
- `--max-per-label 1000` to cap very large folders.
- `--disable-ocr` if you only want native text extraction.

## 3) Train main document classifier

```bash
python training/train_doc_classifier.py \
  --base data/train_data.csv \
  --generated training/data_generated/document_ocr_dataset.csv \
  --output training/models/tfidf_logreg_trained.joblib
```

If you don't want a separate `receipt` class yet in runtime:

```bash
python training/train_doc_classifier.py --map-receipt-to other
```

## 4) Build resume ranker dataset from review feedback

Requires review actions with shortlist labels from admin review queue.

```bash
python training/build_resume_ranker_dataset.py \
  --db-url "$DATABASE_URL" \
  --output data/resume_ranker_training.csv
```

## 5) Train resume ranker model

```bash
python training/train_resume_ranker.py \
  --input data/resume_ranker_training.csv \
  --output model/resume_ranker.joblib
```

## Notes
- OCR extraction for images is enabled in the shared extractor when `enable_ocr=True`.
- `ENABLE_RESUME_RANKER=0` keeps existing behavior until you enable it.
- Rotate secrets in `.env` before deployment.
