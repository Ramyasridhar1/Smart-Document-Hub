import argparse
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train(input_csv, output_model, test_size=0.2, random_state=42):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input dataset not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Expected columns: text,label")

    df = df.dropna(subset=['text', 'label']).copy()
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)

    if df.empty:
        raise ValueError('No usable rows found in dataset after cleaning.')

    class_counts = df['label'].value_counts().to_dict()
    print('Class distribution:', class_counts)

    stratify = df['label'] if len(class_counts) > 1 and min(class_counts.values()) >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'],
        df['label'],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=True,
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=1)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='lbfgs')),
    ])

    print(f'Training on {len(X_train)} samples...')
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    print('\nValidation report:')
    print(classification_report(y_val, preds, zero_division=0, target_names=['reject', 'shortlist']))

    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    joblib.dump(pipeline, output_model)
    print(f"\nSaved resume ranker model: {output_model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train resume shortlist ranker from feedback-derived dataset.')
    parser.add_argument('--input', default=os.path.join('data', 'resume_ranker_training.csv'))
    parser.add_argument('--output', default=os.path.join('model', 'resume_ranker.joblib'))
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    train(args.input, args.output, test_size=args.test_size)
