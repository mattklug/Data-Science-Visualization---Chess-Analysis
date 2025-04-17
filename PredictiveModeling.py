import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# # Load the data
# df_clean = pd.read_csv("games.csv")
# # Filter out instant games (duration = 0)
# df_clean['created_at'] = pd.to_datetime(df_clean['created_at'], unit='ms')
# df_clean['last_move_at'] = pd.to_datetime(df_clean['last_move_at'], unit='ms')
# df_clean['game_duration_sec'] = (df_clean['last_move_at'] - df_clean['created_at']).dt.total_seconds()
# df_clean = df_clean[df_clean['game_duration_sec'] > 0].copy()
# # Create target variable and new features
# df_clean['is_white_win'] = df_clean['winner'] == 'white'
# df_clean['rating_diff'] = df_clean['white_rating'] - df_clean['black_rating']
# df_clean['rated'] = df_clean['rated'].astype(int)  # convert boolean to int
# # Select features (all numerical)
# features = [
#     'white_rating',
#     'black_rating',
#     'rating_diff',
#     'turns',
#     'game_duration_sec',
#     'opening_ply',
#     'rated'
# ]
# X = df_clean[features]
# y = df_clean['is_white_win']
# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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