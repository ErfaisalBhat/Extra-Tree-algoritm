import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("  EXTRA TREES TRAINING — MNEDLY.csv  (Ransomware Detection)")
print("="*70)
print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)


# =========================================
# 1. LOAD DATA
# =========================================
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv("/content/MNEDLY.csv")   # ← change path if needed
print(f"  ✓ Loaded : {df.shape[0]:,} samples  |  {df.shape[1]} columns")


# =========================================
# 2. PREPARE FEATURES AND LABELS
# =========================================
print("\n[STEP 2] Preparing features and labels...")

drop_cols = ['ID', 'filename', 'RG', 'family']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df['RG']

X = X.fillna(0)
X = X.select_dtypes(exclude='object')

print(f"  ✓ Raw features  : {X.shape[1]}")
print(f"  ✓ Ransomware(1) : {y.sum():,}")
print(f"  ✓ Benign    (0) : {(y==0).sum():,}")


# =========================================
# 3. REMOVE NEAR-ZERO VARIANCE FEATURES
#    (Anti-overfitting: removes noisy/
#     constant columns that cause overfit)
# =========================================
print("\n[STEP 3] Removing near-zero variance features...")

selector  = VarianceThreshold(threshold=0.01)
X_reduced = selector.fit_transform(X)
kept_mask = selector.get_support()
X_reduced = pd.DataFrame(X_reduced, columns=X.columns[kept_mask])

removed = X.shape[1] - X_reduced.shape[1]
print(f"  ✓ Removed  : {removed} low-variance features")
print(f"  ✓ Remaining: {X_reduced.shape[1]} features")


# =========================================
# 4. TRAIN / TEST SPLIT  80 / 20
# =========================================
print("\n[STEP 4] Splitting data 80% train / 20% test...")

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"  ✓ Training : {len(X_train):,} samples  ({y_train.sum():,} ransomware)")
print(f"  ✓ Testing  : {len(X_test):,}  samples  ({y_test.sum():,} ransomware)")



print("\n[STEP 5] Building and training Extra Trees model...")

model = ExtraTreesClassifier(
    n_estimators=300,
    max_features='sqrt',
    max_depth=30,               # ← prevents deep overfitting trees
    min_samples_split=10,       # ← anti-overfit
    min_samples_leaf=5,         # ← anti-overfit
    class_weight='balanced',
    bootstrap=True,             # ← adds randomness, reduces overfit
    oob_score=True,             # ← out-of-bag score (free validation)
    n_jobs=-1,
    random_state=42
)

start = datetime.now()
model.fit(X_train, y_train)
t = (datetime.now() - start).total_seconds()

print(f"  ✓ Training complete in {t:.2f} seconds")
print(f"  ✓ OOB Score (out-of-bag): {model.oob_score_:.4f}  "
      f"← internal overfit check (should be close to test score)")


# =========================================
# 6. EVALUATE ON TEST SET
# =========================================
print("\n[STEP 6] Evaluating on test set...")

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  ✓ Performance:")
print(f"    Accuracy  : {acc*100:.2f}%")
print(f"    Precision : {prec*100:.2f}%")
print(f"    Recall    : {rec*100:.2f}%")
print(f"    F1-Score  : {f1:.4f}")
print(f"    ROC-AUC   : {auc:.4f}")
print(f"\n  ✓ Confusion Matrix:")
print(f"    TN: {tn:4d}  FP: {fp:4d}")
print(f"    FN: {fn:4d}  TP: {tp:4d}")
print(f"\n  ✓ Ransomware missed (FN) : {fn} out of {fn+tp}")
print(f"    False Negative Rate    : {fn/(fn+tp)*100:.2f}%")
print(f"    False Positive Rate    : {fp/(fp+tn)*100:.2f}%")

print("\n  ✓ Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Ransomware']))


# =========================================
# 7. OVERFITTING CHECK
#    Compare train vs test accuracy
#    Gap > 5% = likely overfitting
# =========================================
print("\n[STEP 7] Overfitting check (Train vs Test)...")

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = acc
gap       = (train_acc - test_acc) * 100

print(f"  ✓ Train Accuracy : {train_acc*100:.2f}%")
print(f"  ✓ Test  Accuracy : {test_acc*100:.2f}%")
print(f"  ✓ Gap            : {gap:.2f}%  "
      f"→ {'✅ No significant overfit' if gap < 5 else '⚠️  Possible overfit — consider tuning'}")


# =========================================
# 8. 5-FOLD CROSS VALIDATION
# =========================================
print("\n[STEP 8] Running 5-Fold Stratified Cross Validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Score on multiple metrics
for metric in ['roc_auc', 'f1', 'precision', 'recall']:
    scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring=metric, n_jobs=-1)
    label = metric.upper().replace('_', '-')
    print(f"\n  [{label}]")
    for i, s in enumerate(scores, 1):
        print(f"    Fold {i}: {s:.4f}")
    print(f"    Mean : {scores.mean():.4f}  |  Std: {scores.std():.4f}  "
          f"{'✅' if scores.std() < 0.02 else '⚠️  high variance'}")


# =========================================
# 9. TOP FEATURE IMPORTANCES
# =========================================
print("\n[STEP 9] Top 15 important features...")

importances = model.feature_importances_
indices     = np.argsort(importances)[::-1]
feat_names  = list(X_reduced.columns)

print(f"\n  {'Rank':<5} {'Feature':<40} {'Importance'}")
print(f"  {'-'*60}")
for i in range(15):
    idx = indices[i]
    print(f"  {i+1:<5} {feat_names[idx]:<40} {importances[idx]:.4f}")


# =========================================
# 10. SAVE MODEL AND ARTIFACTS
# =========================================
print("\n[STEP 10] Saving model and artifacts...")

joblib.dump(model,                  'extratrees_ransomware_model.pkl')
joblib.dump(list(X_reduced.columns),'et_feature_names.pkl')
joblib.dump(selector,               'et_variance_selector.pkl')

print(f"  ✓ extratrees_ransomware_model.pkl")
print(f"  ✓ et_feature_names.pkl")
print(f"  ✓ et_variance_selector.pkl")


# =========================================
# 11. PLOTS
# =========================================
print("\n[STEP 11] Generating plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Extra Trees — Ransomware Detection Results',
             fontsize=14, fontweight='bold')

# ROC Curve
fpr_c, tpr_c, _ = roc_curve(y_test, y_proba)
axes[0].plot(fpr_c, tpr_c, linewidth=2.5,
             label=f'Extra Trees  AUC = {auc:.4f}', color='#4CAF50')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curve', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Confusion Matrix
axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
axes[1].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontsize=11)
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Benign', 'Ransomware'])
axes[1].set_yticklabels(['Benign', 'Ransomware'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]),
                     ha='center', va='center', fontsize=18, fontweight='bold',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.tight_layout()
plt.savefig('et_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ et_results.png saved")


# =========================================
# FINAL SUMMARY
# =========================================
print("\n" + "="*70)
print("  TRAINING COMPLETE — EXTRA TREES")
print("="*70)
print(f"\n  Accuracy          : {acc*100:.2f}%")
print(f"  Precision         : {prec*100:.2f}%")
print(f"  Recall            : {rec*100:.2f}%")
print(f"  F1-Score          : {f1:.4f}")
print(f"  ROC-AUC           : {auc:.4f}")
print(f"  OOB Score         : {model.oob_score_:.4f}")
print(f"  Train/Test Gap    : {gap:.2f}%  "
      f"({'✅ Good' if gap < 5 else '⚠️  Overfit'})")
print(f"\n  Files saved:")
print(f"  • extratrees_ransomware_model.pkl")
print(f"  • et_feature_names.pkl")
print(f"  • et_variance_selector.pkl")
print(f"  • et_results.png")
print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70 + "\n")
