import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path


# Load dataset
df = pd.read_csv("data/synthetic_patients.csv")

# Select input features and target
features = [
    "glucose",
    "heart_rate",
    "steps",
    "sleep_hours",
    "age",
    "bmi",
    "hba1c",
    "years_since_diagnosis",
]

X = df[features]
y = df["spike_2h"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save model and feature names
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {"model": model, "features": features},
    "models/risk_model.pkl",
)

print("\nModel saved to models/risk_model.pkl")