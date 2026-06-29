import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("flood.csv")

# Features and Target
X = data[['Rainfall', 'Cloud', 'Season']]
y = data['Flood']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create XGBoost model
model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", round(accuracy * 100, 2), "%")

# Save model
joblib.dump(model, "model.pkl")

print("Model saved successfully as model.pkl")