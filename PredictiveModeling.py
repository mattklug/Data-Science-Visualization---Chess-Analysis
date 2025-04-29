import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

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
# Model predicts a white win correctly 65% of the time. Tried removing low-impact features using feature importance,
# but performance dropped to 52%. Rating difference and player ratings are the most influential factors.

# ------------------------------------------------------------
#  RANDOM FOREST MODEL - Predicting if White will win (Grid Search Tuned)
# ------------------------------------------------------------

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
rf_model = grid_search.best_estimator_
rf_y_pred = rf_model.predict(X_test)

print("\n🔍 Random Forest Results (with Grid Search)")
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, rf_y_pred))
print(classification_report(y_test, rf_y_pred))

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
# Slight improvement from 65% to 67% accuracy. Rating and rating difference dominate.
# Interestingly, 'rated' had strong influence in logistic regression but almost none in random forest.
# Grid search improved balance in precision and recall without sacrificing performance.

# ------------------------------------------------------------
#  K-NEAREST NEIGHBORS MODEL - Predicting if White will win (Grid Search Tuned)
# ------------------------------------------------------------

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_param_grid = {
    'n_neighbors': list(range(1, 21)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_knn = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=knn_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_knn.fit(X_train_scaled, y_train)
best_knn = grid_knn.best_estimator_
knn_preds = best_knn.predict(X_test_scaled)

print("\n🔍 K-Nearest Neighbors Results (Grid Search Tuned)")
print("Best Parameters:", grid_knn.best_params_)
print("Accuracy:", accuracy_score(y_test, knn_preds))
print(classification_report(y_test, knn_preds))

sns.heatmap(confusion_matrix(y_test, knn_preds), annot=True, fmt='d', cmap='Purples')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix (Tuned)")
plt.show()

# Analysis:
# KNN accuracy improved from 60% to ~64% after tuning k, weights, and distance metric.
# Still less effective than Random Forest or Logistic Regression, but performs more consistently after tuning.
