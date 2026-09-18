import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# =========================
# 1. Load Dataset
# =========================

data = pd.read_csv("crop_yield.csv")

# =========================
# 2. Define Features & Target
# =========================

X = data[["Year", "Crop"]]
y = data["Value"]

# =========================
# 3. Preprocessing
# =========================

categorical_features = ["Crop"]
numeric_features = ["Year"]

preprocessor = ColumnTransformer(
    transformers=[
        ("crop", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("year", StandardScaler(), numeric_features)
    ]
)

# =========================
# 4. Build Model
# =========================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# =========================
# 5. Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# 6. Train Model
# =========================

model.fit(X_train, y_train)

# =========================
# 7. Evaluate Model
# =========================

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)

# =========================
# 8. Save Model
# =========================

joblib.dump(
    model,
    "improved_crop_yield_model.pkl"
)

print("Model saved successfully.")
