# -------------------------------
# IMPORT LIBRARIES
# -------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# -------------------------------
# LOAD DATASET
# (Replace with your actual CSV / dataset variable)
# -------------------------------
data = pd.read_csv("train_dataa.csv")   # <-- change this name if needed

texts = data["text"]        # your cleaned text column
labels = data["label"]      # your category/class column


# -------------------------------
# TF-IDF VECTORIZATION
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)


# -------------------------------
# TRAIN / TEST SPLIT (IMPORTANT!)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)


# -------------------------------
# TRAIN LOGISTIC REGRESSION MODEL
# -------------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)


# -------------------------------
# PREDICT ON TEST SET
# -------------------------------
y_pred = model.predict(X_test)


# -------------------------------
# PRINT CLASSIFICATION REPORT
# -------------------------------
print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred))


# -------------------------------
# PLOT CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()
