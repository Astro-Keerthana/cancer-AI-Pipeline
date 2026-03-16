import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

PROCESSED_DIR = "data/processed"
MODEL_DIR     = "models"
import os; os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────
X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
X_test  = np.load(f"{PROCESSED_DIR}/X_test.npy")
y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
y_test  = np.load(f"{PROCESSED_DIR}/y_test.npy")
genes   = pd.read_csv(f"{PROCESSED_DIR}/X_features.csv", nrows=0).columns.tolist()

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ─── Model Definition ─────────────────────────────────────────
model = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    use_label_encoder= False,
    eval_metric      = "logloss",
    random_state     = 42,
    n_jobs           = -1
)

# ─── Cross-Validation ─────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=cv, scoring="roc_auc"
)
print(f"\n5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── Final Training ───────────────────────────────────────────
model.fit(
    X_train, y_train,
    eval_set        = [(X_test, y_test)],
    verbose         = 50
)

# ─── Evaluation ───────────────────────────────────────────────
y_pred     = model.predict(X_test)
y_prob     = model.predict_proba(X_test)[:, 1]
auc_score  = roc_auc_score(y_test, y_prob)

print("\n" + "="*50)
print("TEST SET RESULTS")
print("="*50)
print(classification_report(y_test, y_pred,
      target_names=["Normal", "Tumor"]))
print(f"ROC-AUC Score: {auc_score:.4f}")

# ─── Confusion Matrix ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal","Tumor"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - TCGA-BRCA Classifier")
plt.savefig(f"{MODEL_DIR}/confusion_matrix.png", dpi=150)
print(f"Confusion matrix saved")

# ─── Top Biomarkers (Feature Importance) ─────────────────────
importances = model.feature_importances_
feat_df = pd.DataFrame({
    "gene":       genes,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\n🧬 Top 10 Biomarker Genes:")
print(feat_df.head(10).to_string(index=False))

feat_df.head(20).plot(
    kind="barh", x="gene", y="importance",
    figsize=(8, 6), legend=False, color="steelblue"
)
plt.title("Top 20 Biomarker Genes (XGBoost Feature Importance)")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/biomarkers.png", dpi=150)

# ─── Save Model ───────────────────────────────────────────────
joblib.dump(model,   f"{MODEL_DIR}/xgb_cancer_model.pkl")
feat_df.to_csv(f"{MODEL_DIR}/biomarkers.csv", index=False)
print(f"\nModel saved → {MODEL_DIR}/xgb_cancer_model.pkl")
