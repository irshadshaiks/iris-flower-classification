# ============================================================ # Task 1: Iris Flower Classification # Internship Assignment import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc )
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle

# -- Output directory --
OUTPUT_DIR = os.path.expanduser("~/Desktop/iris_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, filename):
      path = os.path.join(OUTPUT_DIR, filename)
      fig.savefig(path, dpi=150, bbox_inches="tight")
      plt.close(fig)
      print(f" [Saved] {filename}")

# 1. LOAD & EXPLORE
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names
df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, class_names)

print("=" * 62)
print(" IRIS FLOWER CLASSIFICATION - ENHANCED")
print("=" * 62)
print(f"\n[Dataset] Dataset shape : {X.shape} (samples x features)")
print(f"[Iris] Classes : {list(class_names)}")
print(f"[Samples] Samples/class : {dict(zip(class_names, np.bincount(y)))}")
print("\n-- Descriptive Statistics --")
print(df.groupby("species").mean().round(2).to_string())

# 2. VISUALISATION - Scatter plots & Histograms
PALETTE = ["#4C72B0", "#DD8452", "#55A868"] # blue / orange / green

# --- 2a. Sepal & Petal scatter side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Iris Dataset - Exploratory Scatter Plots", fontsize=14, fontweight="bold")
pairs = [
      (0, 1, "Sepal Length (cm)", "Sepal Width (cm)", "Sepal Measurements"),
      (2, 3, "Petal Length (cm)", "Petal Width (cm)", "Petal Measurements"),
]
for ax, (xi, yi, xl, yl, title) in zip(axes, pairs):
      for i, (name, colour) in enumerate(zip(class_names, PALETTE)):
                mask = y == i
                ax.scatter(X[mask, xi], X[mask, yi], label=name, color=colour, alpha=0.8, edgecolors="white", linewidths=0.4)
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
    ax.legend(framealpha=0.7)
plt.tight_layout()
save(fig, "01_scatter_plots.png")

# --- 2b. Feature Histograms (2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Feature Distributions by Class", fontsize=13, fontweight="bold")
for idx, (ax, feat) in enumerate(zip(axes.flatten(), feature_names)):
      for i, (name, colour) in enumerate(zip(class_names, PALETTE)):
                ax.hist(X[y == i, idx], bins=15, alpha=0.65, label=name, color=colour, edgecolor="white")
            ax.set_title(feat, fontweight="bold")
    ax.set_xlabel("cm"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)
plt.tight_layout()
save(fig, "02_feature_histograms.png")

# --- 2c. Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(7, 5))
corr = df.drop(columns="species").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "03_correlation_heatmap.png")

# --- 2d. Box plots per feature ---
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Feature Box Plots by Species", fontsize=13, fontweight="bold")
for idx, (ax, feat) in enumerate(zip(axes, feature_names)):
      data_by_class = [X[y == i, idx] for i in range(3)]
    bp = ax.boxplot(data_by_class, patch_artist=True, labels=class_names, medianprops=dict(color="black", linewidth=2))
    for patch, colour in zip(bp["boxes"], PALETTE):
              patch.set_facecolor(colour); patch.set_alpha(0.75)
          ax.set_title(feat.replace(" (cm)", ""), fontsize=9, fontweight="bold")
    ax.set_ylabel("cm"); ax.tick_params(axis="x", labelrotation=20)
plt.tight_layout()
save(fig, "04_box_plots.png")

# 3. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n-- Train / Test Split --")
print(f" Train samples : {len(X_train)} | Test samples : {len(X_test)}")

# 4. PREPROCESSING
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# 5. TRAIN MULTIPLE CLASSIFIERS + CROSS-VALIDATION
models = {
      "K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=5),
      "Logistic Regression" : LogisticRegression(max_iter=300, random_state=42),
      "Decision Tree" : DecisionTreeClassifier(max_depth=4, random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print(f"\n-- Model Training & Evaluation --")
print(f" (5-Fold Stratified Cross-Validation on full dataset)")
print()

for name, model in models.items():
      cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
      model.fit(X_train_sc, y_train)
      y_pred = model.predict(X_test_sc)
      acc = accuracy_score(y_test, y_pred)
      prec = precision_score(y_test, y_pred, average="weighted")
      rec = recall_score(y_test, y_pred, average="weighted")
      f1 = f1_score(y_test, y_pred, average="weighted")
      results[name] = {
          "model" : model,
          "y_pred" : y_pred,
          "accuracy" : acc,
          "precision": prec,
          "recall" : rec,
          "f1" : f1,
          "cv_mean" : cv_scores.mean(),
          "cv_std" : cv_scores.std(),
      }
      print(f" {name}")
      print(f" Test -> Acc: {acc:.4f} Prec: {prec:.4f} Rec: {rec:.4f} F1: {f1:.4f}")
      print(f" CV -> {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
      print()

# 6. BEST MODEL - DETAILED REPORT
best_name = max(results, key=lambda k: results[k]["accuracy"])
best = results[best_name]
print("=" * 62)
print(f" [Best] Best Model : {best_name}")
print(f" Test Accuracy : {best['accuracy']:.4f} | F1 : {best['f1']:.4f}")
print("=" * 62)
print("\nClassification Report:")
print(classification_report(y_test, best["y_pred"], target_names=class_names))

# 7. VISUALISATION - Confusion Matrices (all models)
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Confusion Matrices - All Classifiers", fontsize=14, fontweight="bold")
for ax, (name, res) in zip(axes, results.items()):
      cm = confusion_matrix(y_test, res["y_pred"])
      sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax, linewidths=0.5, cbar=False)
      ax.set_title(f"{name}\nAcc: {res['accuracy']:.4f}", fontweight="bold", fontsize=10)
      ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
  plt.tight_layout()
save(fig, "05_confusion_matrices.png")

# 8. MODEL COMPARISON BAR CHART
metrics = ["accuracy", "precision", "recall", "f1"]
labels = [m.capitalize() for m in metrics]
x = np.arange(len(metrics))
width = 0.25
fig, ax = plt.subplots(figsize=(11, 5))
for idx, (name, res) in enumerate(results.items()):
      vals = [res[m] for m in metrics]
      bars = ax.bar(x + idx * width, vals, width, label=name, color=PALETTE[idx], alpha=0.85, edgecolor="white")
      for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.set_ylabel("Score")
ax.set_ylim(0.75, 1.05)
ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.legend(framealpha=0.8)
ax.axhline(1.0, linestyle="--", color="gray", linewidth=0.8)
plt.tight_layout()
save(fig, "06_model_comparison.png")

# 9. CROSS-VALIDATION SCORE COMPARISON
fig, ax = plt.subplots(figsize=(9, 5))
names_list = list(results.keys())
cv_means = [results[n]["cv_mean"] for n in names_list]
cv_stds = [results[n]["cv_std"] for n in names_list]
bars = ax.bar(names_list, cv_means, color=PALETTE, alpha=0.85, edgecolor="white", yerr=cv_stds, capsize=6)
for bar, mean in zip(bars, cv_means):
      ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{mean:.4f}", ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("CV Accuracy (5-Fold)")
ax.set_ylim(0.7, 1.05)
ax.set_title("5-Fold Cross-Validation Accuracy +/- Std Dev", fontsize=13, fontweight="bold")
ax.axhline(1.0, linestyle="--", color="gray", linewidth=0.8)
plt.tight_layout()
save(fig, "07_cross_validation.png")

# 10. ROC CURVES (One-vs-Rest, Best Model)
best_model_cls = {
      "K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=5),
      "Logistic Regression" : LogisticRegression(max_iter=300, random_state=42),
      "Decision Tree" : DecisionTreeClassifier(max_depth=4, random_state=42),
}[best_name]
ovr = OneVsRestClassifier(best_model_cls)
y_bin = label_binarize(y, classes=[0, 1, 2])
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split( X, y_bin, test_size=0.2, random_state=42, stratify=y )
X_train_bsc = scaler.fit_transform(X_train_b)
X_test_bsc = scaler.transform(X_test_b)
ovr.fit(X_train_bsc, y_train_b)
y_score = ovr.predict_proba(X_test_bsc)
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = cycle(["#4C72B0", "#DD8452", "#55A868"])
for i, (colour, cname) in enumerate(zip(colors_roc, class_names)):
      fpr, tpr, _ = roc_curve(y_test_b[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colour, lw=2, label=f"{cname} (AUC = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curves - {best_name} (One-vs-Rest)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", framealpha=0.85)
plt.tight_layout()
save(fig, "08_roc_curves.png")

# 11. FEATURE IMPORTANCE (Decision Tree)
dt_model = results["Decision Tree"]["model"]
importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(range(4), importances[indices], color=[PALETTE[i % 3] for i in range(4)], alpha=0.85, edgecolor="white")
for bar, val in zip(bars, importances[indices]):
      ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
ax.set_xticks(range(4))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=15, ha="right")
ax.set_ylabel("Importance Score")
ax.set_title("Decision Tree - Feature Importance", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "09_feature_importance.png")

# 12. DECISION TREE VISUALISATION
fig, ax = plt.subplots(figsize=(18, 8))
plot_tree(dt_model, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title("Decision Tree Structure (max_depth=4)", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "10_decision_tree.png")

# SUMMARY
print("\n-- Output Files --")
print(f" All plots saved to: {OUTPUT_DIR}")
print("""
01_scatter_plots.png - Sepal & Petal scatter
02_feature_histograms.png - Feature distributions by class
03_correlation_heatmap.png - Feature correlation matrix
04_box_plots.png - Box plots per species
05_confusion_matrices.png - Confusion matrices (all models)
06_model_comparison.png - Acc / Prec / Recall / F1 bar chart
07_cross_validation.png - 5-Fold CV accuracy comparison
08_roc_curves.png - ROC curves (One-vs-Rest)
09_feature_importance.png - Decision Tree feature importance
10_decision_tree.png - Decision Tree structure
""")
print("Task Complete!\n")
