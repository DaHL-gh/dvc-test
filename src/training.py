import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

import mlflow


def train_model():
    df = pd.read_csv("./data/housing_processed.csv")

    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    n_estimators = 15
    random_state = 42

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=15, random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "random_state": random_state,
                "mse": mse,
                "r2": r2,
            }
        )

        joblib.dump(model, "model/model.pkl")
        mlflow.sklearn.log_model(model, "housing_random_forest")


if __name__ == "__main__":
    train_model()
