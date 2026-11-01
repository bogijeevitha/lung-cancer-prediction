from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

DATA = Path(__file__).resolve().parent / "lung_cancer_examples.csv"
FEATURES = ["Age", "Smokes", "AreaQ", "Alkhol"]
TARGET = "Result"

df = pd.read_csv(DATA)
df = df[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
X, y = df[FEATURES], df[TARGET].astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC()),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
}
scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

print(f"Dataset shape used: {df.shape}")
print("Note: these are cross-validation estimates on a very small educational dataset.")
for name, estimator in models.items():
    result = cross_validate(estimator, X, y, cv=cv, scoring=scoring)
    print(f"\n{name}")
    for metric in scoring:
        values = result[f"test_{metric}"]
        print(f"{metric}: {values.mean():.4f} (+/- {values.std():.4f})")
