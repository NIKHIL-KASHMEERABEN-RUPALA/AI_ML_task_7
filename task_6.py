import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.inspection import DecisionBoundaryDisplay
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("dark")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

print("SUPPORT VECTOR MACHINES - Complete Mastery Showcase")
print("="*100)

data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Benign: {sum(y==1)} | Malignant: {sum(y==0)} → {sum(y==1)/len(y):.2%} benign")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTuning SVM with RBF Kernel (Grid Search + 5-fold CV)...")

param_grid = {
    'C': [0.1, 1, 10, 50, 100, 500],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    cv=StratifiedKFold(5),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

best_svm = grid.best_estimator_
print(f"\nBest Parameters: {grid.best_params_}")
print(f"Best CV AUC: {grid.best_score_:.5f}")

linear_svm = SVC(kernel='linear', C=1, probability=True, random_state=42)
rbf_svm = best_svm

linear_svm.fit(X_train, y_train)
rbf_svm.fit(X_train, y_train)

y_pred_linear = linear_svm.predict(X_test)
y_pred_rbf = rbf_svm.predict(X_test)
y_proba_rbf = rbf_svm.predict_proba(X_test)[:, 1]

coef = np.abs(linear_svm.coef_[0])
top2_idx = np.argsort(coef)[-2:]
feat1, feat2 = top2_idx

X_2d = X_scaled[:, top2_idx]
X_train_2d, X_test_2d, _, _ = train_test_split(X_2d, y, test_size=0.2, random_state=42, stratify=y)

svm_2d_linear = SVC(kernel='linear', C=1).fit(X_train_2d, y_train)
svm_2d_rbf = SVC(kernel='rbf', C=best_svm.C, gamma=best_svm.gamma).fit(X_train_2d, y_train)

fig = plt.figure(figsize=(26, 20))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
DecisionBoundaryDisplay.from_estimator(
    svm_2d_linear, X_2d, cmap='RdBu', alpha=0.8, ax=ax1, response_method="predict"
)
plt.scatter(X_train_2d[y_train==0, 0], X_train_2d[y_train==0, 1], c='red', label='Malignant', edgecolors='k')
plt.scatter(X_train_2d[y_train==1, 0], X_train_2d[y_train==1, 1], c='blue', label='Benign', edgecolors='k')
plt.xlabel(f"{feature_names[feat1]} (scaled)")
plt.ylabel(f"{feature_names[feat2]} (scaled)")
plt.title("SVM Decision Boundary - Linear Kernel\n(Maximum Margin Classifier)", fontweight='bold', fontsize=14)
plt.legend()

ax2 = fig.add_subplot(gs[0, 1])
DecisionBoundaryDisplay.from_estimator(
    svm_2d_rbf, X_2d, cmap='RdBu', alpha=0.8, ax=ax2, response_method="predict"
)
plt.scatter(X_train_2d[y_train==0, 0], X_train_2d[y_train==0, 1], c='red', edgecolors='k')
plt.scatter(X_train_2d[y_train==1, 0], X_train_2d[y_train==1, 1], c='blue', edgecolors='k')
plt.xlabel(f"{feature_names[feat1]} (scaled)")
plt.ylabel(f"{feature_names[feat2]} (scaled)")
plt.title("SVM Decision Boundary - RBF Kernel\n(Non-linear via Kernel Trick)", fontweight='bold', fontsize=14)

ax3 = fig.add_subplot(gs[0, 2])
DecisionBoundaryDisplay.from_estimator(svm_2d_rbf, X_2d, cmap='RdBu', alpha=0.6, ax=ax3)
sv = svm_2d_rbf.support_
plt.scatter(X_train_2d[sv, 0], X_train_2d[sv, 1], s=200, facecolors='none', edgecolors='black', linewidth=2, label='Support Vectors')
plt.scatter(X_train_2d[y_train==0, 0], X_train_2d[y_train==0, 1], c='red', alpha=0.7)
plt.scatter(X_train_2d[y_train==1, 0], X_train_2d[y_train==1, 1], c='blue', alpha=0.7)
plt.title(f"Support Vectors = {len(sv)} points\n(Only these define the margin!)", fontweight='bold')
plt.legend()

ax4 = fig.add_subplot(gs[1, 0])
C_values = [0.1, 1, 10, 100]
colors = plt.cm.viridis(np.linspace(0, 1, len(C_values)))
for i, C in enumerate(C_values):
    svm_c = SVC(kernel='linear', C=C).fit(X_train_2d, y_train)
    DecisionBoundaryDisplay.from_estimator(svm_c, X_2d, alpha=0.3, ax=ax4, cmap='RdBu')
    ax4.set_title("Effect of C (Regularization)\nHigher C → Less regularization → Tighter margin", fontweight='bold')

ax5 = fig.add_subplot(gs[1, 1])
gammas = [0.001, 0.01, 0.1, 1]
for i, g in enumerate(gammas):
    svm_g = SVC(kernel='rbf', C=10, gamma=g).fit(X_train_2d, y_train)
    DecisionBoundaryDisplay.from_estimator(svm_g, X_2d, alpha=0.3, ax=ax5, cmap='RdBu')
    ax5.set_title("Effect of Gamma (RBF Kernel)\nHigher γ → More complex boundary → Overfitting risk", fontweight='bold')

ax6 = fig.add_subplot(gs[1, 2])
for name, model in [("Linear SVM", linear_svm), ("RBF SVM (Best)", rbf_svm)]:
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_score = roc_auc_score(y_test, proba)
    ax6.plot(fpr, tpr, lw=3, label=f'{name} (AUC = {auc_score:.4f})')
ax6.plot([0,1],[0,1], 'k--')
ax6.set_xlabel('False Positive Rate')
ax6.set_ylabel('True Positive Rate')
ax6.set_title('ROC Curve: Linear vs RBF SVM', fontweight='bold')
ax6.legend()

ax7 = fig.add_subplot(gs[2, 0])
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
Z = svm_2d_linear.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax7.contour(xx, yy, Z, levels=[-1, 0, 1], alpha
