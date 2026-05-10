from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split


def train_model():

    Path("./models").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("./data/model_ready/dataset_final.csv")

    print(f"Dataset carregado → {df.shape}")

    X = df[
        [
            "latitude",
            "longitude",
            "mes",
            "temperatura_c",
            "umidade_relativa_percentual",
            "precipitacao_mmdia",
            "vento_velocidade_ms"
        ]
    ]

    y = df["risco_queimada"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"\nTreino → {X_train.shape}")
    print(f"Teste → {X_test.shape}")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    print("\nTreinando modelo...")

    model.fit(X_train, y_train)

    print("Modelo treinado com sucesso!")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    print("\n===== MÉTRICAS =====")

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\n===== IMPORTÂNCIA DAS FEATURES =====")
    print(feature_importance)

    joblib.dump(
        model,
        "./models/wildfire_model.pkl"
    )

    print("\nModelo salvo em:")
    print("./models/wildfire_model.pkl")