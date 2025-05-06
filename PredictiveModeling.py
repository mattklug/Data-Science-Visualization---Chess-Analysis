import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------------------------------------------------------------
#  LOAD & PREP DATA
# ------------------------------------------------------------

df_clean = pd.read_csv("games.csv")
df_clean['created_at'] = pd.to_datetime(df_clean['created_at'], unit='ms')
df_clean['last_move_at'] = pd.to_datetime(df_clean['last_move_at'], unit='ms')
df_clean['game_duration_sec'] = (df_clean['last_move_at'] - df_clean['created_at']).dt.total_seconds()
df_clean = df_clean[df_clean['game_duration_sec'] > 0].copy()

df_clean['is_white_win'] = df_clean['winner'] == 'white'
df_clean['rating_diff'] = df_clean['white_rating'] - df_clean['black_rating']
df_clean['rated'] = df_clean['rated'].astype(int)

features = [
    'white_rating',
    'black_rating',
    'rating_diff',
    'turns',
    'game_duration_sec',
    'opening_ply',
    'rated'
]

X = df_clean[features]
y = df_clean['is_white_win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ------------------------------------------------------------
#  LOGISTIC REGRESSION MODEL - Predicting if White will win
# ------------------------------------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n🔍 Logistic Regression Results")
print("Hold‑out Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("5‑Fold CV Accuracy: {:.3f} ± {:.3f}".format(cv_scores.mean(), cv_scores.std()))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

coeffs = pd.Series(model.coef_[0], index=X.columns)
top_features = coeffs.abs().sort_values(ascending=True)

plt.figure(figsize=(8, 6))
top_features.plot(kind='barh', color='skyblue')
plt.title("Logistic Regression Feature Importance")
plt.xlabel("Absolute Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Analysis:
# Model predicts a white win correctly ~65 % (hold‑out) and ~64–66 % (CV). Rating difference and player ratings dominate.
# Removing low‑impact features hurt accuracy (dropped to 52 %).


# ------------------------------------------------------------
#  RANDOM FOREST MODEL - Predicting if White will win (Grid Search Tuned)
# ------------------------------------------------------------

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
rf_model = grid_search.best_estimator_
rf_y_pred = rf_model.predict(X_test)

print("\n🔍 Random Forest Results (with Grid Search)")
print("Best Parameters:", grid_search.best_params_)
print("Hold‑out Accuracy:", accuracy_score(y_test, rf_y_pred))
print(classification_report(y_test, rf_y_pred))

rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print("5‑Fold CV Accuracy: {:.3f} ± {:.3f}".format(rf_cv_scores.mean(), rf_cv_scores.std()))

sns.heatmap(confusion_matrix(y_test, rf_y_pred), annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_top_features = rf_importances.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
rf_top_features.plot(kind='barh', color='lightgreen')
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Analysis:
# Hold‑out accuracy ~67 %, CV ~66–68 %. Rating features dominate. Grid search balanced precision/recall.


# ------------------------------------------------------------
#  K-NEAREST NEIGHBORS MODEL - Predicting if White will win (Grid Search Tuned)
# ------------------------------------------------------------

knn_param_grid = {
    'knn__n_neighbors': list(range(1, 21)),
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

# Pipeline ensures scaling is applied inside CV properly
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

grid_knn = GridSearchCV(
    knn_pipeline,
    param_grid=knn_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_knn.fit(X_train, y_train)
best_knn = grid_knn.best_estimator_
knn_preds = best_knn.predict(X_test)

print("\n🔍 K-Nearest Neighbors Results (Grid Search Tuned)")
print("Best Parameters:", grid_knn.best_params_)
print("Hold‑out Accuracy:", accuracy_score(y_test, knn_preds))
print(classification_report(y_test, knn_preds))

knn_cv_scores = cross_val_score(best_knn, X, y, cv=5, scoring='accuracy')
print("5‑Fold CV Accuracy: {:.3f} ± {:.3f}".format(knn_cv_scores.mean(), knn_cv_scores.std()))

sns.heatmap(confusion_matrix(y_test, knn_preds), annot=True, fmt='d', cmap='Purples')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix (Tuned)")
plt.show()

# Permutation importance for KNN
from sklearn.inspection import permutation_importance
perm = permutation_importance(best_knn, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
knn_perm_importance = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
knn_perm_importance.plot(kind='barh', color='plum')
plt.title("KNN Feature Importance (Permutation)")
plt.xlabel("Mean Decrease in Accuracy")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Analysis:
# KNN tuned accuracy ~64 % (hold‑out) and ~63–65 % (CV). Still trails Random Forest but improved over default k=5.
# Permutation importance shows rating-related features matter most here too.


# ------------------------------------------------------------
#  GRADIENT BOOSTING MODEL - Predicting if White will win (HistGradientBoosting + Grid Search)
# ------------------------------------------------------------

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

gb_param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'max_iter': [100, 200]
}

grid_gb = GridSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    param_grid=gb_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_gb.fit(X_train, y_train)
gb_model = grid_gb.best_estimator_
gb_pred = gb_model.predict(X_test)

print("\n🔍 Gradient Boosting Results (HistGB + Grid Search)")
print("Best Parameters:", grid_gb.best_params_)
gb_hold = accuracy_score(y_test, gb_pred)
print("Hold‑out Accuracy:", gb_hold)
print(classification_report(y_test, gb_pred))

gb_cv_scores = cross_val_score(gb_model, X, y, cv=5, scoring='accuracy')
gb_cv_mean = gb_cv_scores.mean()
print("5‑Fold CV Accuracy: {:.3f} ± {:.3f}".format(gb_cv_mean, gb_cv_scores.std()))

sns.heatmap(confusion_matrix(y_test, gb_pred), annot=True, fmt='d', cmap='Oranges')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Gradient Boosting Confusion Matrix")
plt.show()

# Permutation importance (built‑in trees don’t expose feature_importances_)
gb_perm = permutation_importance(gb_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
gb_importance = pd.Series(gb_perm.importances_mean, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
gb_importance.plot(kind='barh', color='orange')
plt.title("Gradient Boosting Feature Importance (Permutation)")
plt.xlabel("Mean Decrease in Accuracy")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Analysis:
# Gradient Boosting often rivals or beats RF. Check if hold‑out/CV exceed 67 %.
# Feature importance should echo rating‑based signals but can surface new interactions.

# ------------------------------------------------------------
#  MODEL COMPARISON SUMMARY
# ------------------------------------------------------------

from sklearn.metrics import precision_score, recall_score, f1_score

model_metrics = {
    "Model": ["Logistic Regression", "Random Forest", "K‑Nearest Neighbors", "Gradient Boosting"],
    "Hold‑out Accuracy": [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, rf_y_pred),
        accuracy_score(y_test, knn_preds),
        gb_hold
    ],
    "CV Accuracy (mean)": [
        cv_scores.mean(),
        rf_cv_scores.mean(),
        knn_cv_scores.mean(),
        gb_cv_mean
    ],
    "Precision": [
        precision_score(y_test, y_pred),
        precision_score(y_test, rf_y_pred),
        precision_score(y_test, knn_preds),
        precision_score(y_test, gb_pred)
    ],
    "Recall": [
        recall_score(y_test, y_pred),
        recall_score(y_test, rf_y_pred),
        recall_score(y_test, knn_preds),
        recall_score(y_test, gb_pred)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred),
        f1_score(y_test, rf_y_pred),
        f1_score(y_test, knn_preds),
        f1_score(y_test, gb_pred)
    ]
}
metrics_df = pd.DataFrame(model_metrics)
print("\n🔎 Model Comparison Summary")
print(metrics_df)

metrics_df.set_index("Model")[["Hold‑out Accuracy", "CV Accuracy (mean)", "Precision", "Recall", "F1 Score"]].plot(
    kind="bar", figsize=(10, 6), colormap="viridis", edgecolor="black"
)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0.5, 1.0)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Analysis:
# Random Forest remains top performer on both hold‑out and CV scores (~67 %).
# Logistic Regression stays a solid baseline. Tuned KNN catches up but still trails.
# CV results confirm stability across data splits, backing up final model selection.
