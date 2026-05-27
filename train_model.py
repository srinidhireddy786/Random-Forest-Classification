# =========================
# Import Libraries
# =========================
import joblib

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# Load Dataset
# =========================
data = load_wine()

X = data.data
y = data.target

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Scaling
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Random Forest Classifier
# =========================
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# =========================
# Train Model
# =========================
model.fit(X_train_scaled, y_train)

# =========================
# Predictions
# =========================
y_pred = model.predict(X_test_scaled)

# =========================
# Accuracy
# =========================
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nAccuracy : {accuracy * 100:.2f}%")

# =========================
# Save Model & Scaler
# =========================
joblib.dump(
    model,
    "models/random_forest_classifier.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nModel Saved Successfully")