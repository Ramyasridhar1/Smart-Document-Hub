# train_classifier.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from collections import Counter

# Config
DATA_CSV = "data/train_data.csv"   # create this CSV (see sample below)
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "tfidf_logreg.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    # Expect columns: label,text
    df = df.dropna(subset=['label', 'text'])
    return df['text'].astype(str).tolist(), df['label'].astype(str).tolist()

def safe_train_test_split(X, y, test_size=0.15, random_state=42):
    """
    Try stratified split when each class has >=2 members; otherwise do a plain split.
    Returns X_train, X_val, y_train, y_val, and a flag whether stratify was used.
    """
    counts = Counter(y)
    too_small = [lbl for lbl, cnt in counts.items() if cnt < 2]
    if too_small:
        print("Warning: the following labels have fewer than 2 examples (cannot stratify):", too_small)
        print("Proceeding without stratification. Consider adding more labeled examples for better validation.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    # check also that test_size will allow >=1 sample per class in validation
    n = len(y)
    expected_test = int(n * test_size)
    if expected_test < len(counts):
        # if there are many classes relative to dataset, fallback to no stratify
        print("Warning: test set may be too small relative to number of classes; proceeding without stratify.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    # safe to stratify
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train():
    if not os.path.exists(DATA_CSV):
        print(f"Training data not found: {DATA_CSV}")
        print("Create data/train_data.csv with columns: label,text")
        return

    print("Loading data from", DATA_CSV)
    X, y = load_data(DATA_CSV)
    if not X:
        print("No training data found (empty). Please add labeled examples to data/train_data.csv")
        return

    print(f"Total samples: {len(y)} ; classes: {len(set(y))}")
    # split safely
    X_train, X_val, y_train, y_val = safe_train_test_split(X, y, test_size=0.15, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=20000, min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear'))
    ])

    print("Training model on", len(X_train), "samples...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # eval if validation set exists
    if X_val:
        preds = pipeline.predict(X_val)
        print("Validation results:")
        print(classification_report(y_val, preds))
    else:
        print("No validation set created (very small dataset).")

    # save
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")

if __name__ == "__main__":
    train()
