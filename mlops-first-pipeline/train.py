from pathlib import Path
import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 5. Create the models directory
BASE_DIR = Path(__file__).resolve().parent
model_dir = BASE_DIR / "models"
model_dir.mkdir(parents=True, exist_ok=True)

# 6. Save the model and metadata
model_path = model_dir / "iris_model.joblib"

saved_data = {
    "model": model,
    "target_names": iris.target_names.tolist(),
    "feature_names": iris.feature_names
}

joblib.dump(saved_data, model_path)

print(f"Model saved successfully at: {model_path}")