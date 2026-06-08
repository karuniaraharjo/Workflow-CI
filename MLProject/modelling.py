# modelling.py
# Training model Titanic dengan MLflow autolog

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv('titanic_preprocessing/titanic_preprocessed.csv')

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Features         : {X.columns.tolist()}")

# ── 2. Aktifkan MLflow Autolog ────────────────────────────────
mlflow.sklearn.autolog()

# ── 3. Latih Model ───────────────────────────────────────────
with mlflow.start_run(run_name="titanic_random_forest_basic"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"\n✅ Accuracy  : {score:.4f}")

print("\n🎉 Training selesai!")
print("Jalankan: mlflow ui")
print("Buka    : http://localhost:5000")