import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

# Load Dataset
df = pd.read_csv("breast-cancer.csv")

# Remove unnecessary column
df.drop(columns=["id"], inplace=True)

# Encode target
encoder = LabelEncoder()
df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

# Split data
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123
)

# Model
model = LogisticRegression(max_iter=1000)

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)
print("Accuracy:", best_model.score(X_test, y_test))

# Save files
joblib.dump(best_model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("Done.")
