import argparse
import os
from collections import Counter

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_dataset(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=['label', 'text'])
    df = pd.read_csv(path)
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError(f"Dataset {path} must include columns: label,text")
    df = df.dropna(subset=['label', 'text'])
    df['label'] = df['label'].astype(str)
    df['text'] = df['text'].astype(str)
    return df[['label', 'text']]


def train(base_csv, generated_csv, output_model, test_size=0.15, random_state=42, map_receipt_to=None):
    base_df = load_dataset(base_csv)
    gen_df = load_dataset(generated_csv)

    df = pd.concat([base_df, gen_df], ignore_index=True)
    if df.empty:
        raise ValueError('No training rows found across provided datasets.')

    if map_receipt_to:
        df.loc[df['label'].str.lower() == 'receipt', 'label'] = map_receipt_to

    counts = Counter(df['label'].tolist())
    class_distribution = dict(sorted(counts.items()))
    print('Class distribution:', class_distribution)

    labels = df['label']
    stratify = labels if min(counts.values()) >= 2 and int(len(df) * test_size) >= len(counts) else None

    X_train, X_val, y_train, y_val = train_test_split(
        df['text'],
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=True,
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=1)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')),
    ])

    print(f'Training classifier on {len(X_train)} samples...')
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    report = classification_report(y_val, preds, zero_division=0)
    print('\nValidation report:')
    print(report)

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    joblib.dump(pipeline, output_model)
    print(f"\nSaved document classifier model: {output_model}")

    report_path = os.path.join('training', 'reports', 'doc_classifier_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as rf:
        rf.write('Document classifier training report\n')
        rf.write(f'base_csv={base_csv}\n')
        rf.write(f'generated_csv={generated_csv}\n')
        rf.write(f'output_model={output_model}\n')
        rf.write(f'test_size={test_size}\n')
        rf.write(f'random_state={random_state}\n')
        rf.write(f'class_distribution={class_distribution}\n\n')
        rf.write(report)
    print(f"Saved training report: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train main document classifier from base + OCR-generated datasets.')
    parser.add_argument('--base', default=os.path.join('data', 'train_data.csv'))
    parser.add_argument('--generated', default=os.path.join('training', 'data_generated', 'document_ocr_dataset.csv'))
    parser.add_argument('--output', default=os.path.join('training', 'models', 'tfidf_logreg_trained.joblib'))
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--map-receipt-to', default=None, help='Optional: map receipt label to another class (e.g., other)')
    args = parser.parse_args()

    train(
        base_csv=args.base,
        generated_csv=args.generated,
        output_model=args.output,
        test_size=args.test_size,
        map_receipt_to=args.map_receipt_to,
    )
