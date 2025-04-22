import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load the data
df_clean = pd.read_csv("games.csv")
# Filter out instant games (duration = 0)
df_clean['created_at'] = pd.to_datetime(df_clean['created_at'], unit='ms')
df_clean['last_move_at'] = pd.to_datetime(df_clean['last_move_at'], unit='ms')
df_clean['game_duration_sec'] = (df_clean['last_move_at'] - df_clean['created_at']).dt.total_seconds()
df_clean = df_clean[df_clean['game_duration_sec'] > 0].copy()

# ------------------------------------------------------------
#  LOGISTIC REGRESSION MODEL - Predicting if White will win
# ------------------------------------------------------------

# Create target variable and new features
df_clean['is_white_win'] = df_clean['winner'] == 'white'
df_clean['rating_diff'] = df_clean['white_rating'] - df_clean['black_rating']
df_clean['rated'] = df_clean['rated'].astype(int)  # convert boolean to int
# Select features (all numerical)
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
# # Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Train logistic regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
# # Predictions and evaluation
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# # Confusion matrix
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()
# # Get and sort feature importance
# coeffs = pd.Series(model.coef_[0], index=X.columns)
# top_features = coeffs.abs().sort_values(ascending=True)  # smallest to largest for horizontal bar chart
# # Plot
# plt.figure(figsize=(8, 6))
# top_features.plot(kind='barh', color='skyblue')
# plt.title("Feature Importance (Logistic Regression Coefficients)")
# plt.xlabel("Absolute Coefficient Value")
# plt.ylabel("Feature")
# plt.tight_layout()
# plt.show()

# Analysis from above Logistic Regression model, model predicts a white win correctly 65% of the time, tried feature importance to remove unnecessary features, removed
# ones with little to no impact, model efficiency was reduced to being correct 52% of the time, relying way too much on the rating difference, added back in previously
# removed features. Running model displays evaluation metrics (precision/recall/f1-score/support), confusion matrix, & feature importance bar chart.

# ------------------------------------------------------
#  RANDOM FOREST MODEL - Predicting if White will win
# ------------------------------------------------------

# Set up parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Use GridSearchCV to find best hyperparameters
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1  # Use all available cores
)

# Fit grid search on training data
grid_search.fit(X_train, y_train)

# Grab the best model found during the search
rf_model = grid_search.best_estimator_

# Predict and evaluate
rf_y_pred = rf_model.predict(X_test)

print("\n🔍 Random Forest Results (with Grid Search)")
print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy_score(y_test, rf_y_pred))
print(classification_report(y_test, rf_y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, rf_y_pred), annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Feature Importance Plot
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
rf_top_features = rf_importances.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
rf_top_features.plot(kind='barh', color='lightgreen')
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# Some cool ass data things to note, we slightly bettered our prediction of if white wins or not, from 65% to 67%. With rating and rating difference as our dominant features
# within the model. Something interesting, the 'rated' flag, which was very influential in our linear model, did not seem to ave much effect at all in our random forest model,
# meaning actual gameplay metrics seem to hold more value than if a player is rated or not.