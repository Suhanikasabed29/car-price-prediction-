from pathlib import Path
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "car data.csv"
MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"

TARGET = "Selling_Price"
FEATURES = [
    "Car_Name", "Year", "Present_Price", "Driven_kms",
    "Fuel_Type", "Selling_type", "Transmission", "Owner"
]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    X = df[FEATURES].copy()
    y = df[TARGET].astype(float)

    categorical = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
    numeric = ["Year", "Present_Price", "Driven_kms", "Owner"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            min_samples_leaf=1,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test MAE: {mae:.3f} lakh")
    print(f"Test R²: {r2:.3f}")

if __name__ == "__main__":
    main()
