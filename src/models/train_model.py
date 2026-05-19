from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
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
            "vento_velocidade_ms",

            "estacao_seca"
        ]
    ]

    y = df["risco"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"\nTreino → {X_train.shape}")
    print(f"Teste → {X_test.shape}")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    print("\nTreinando modelo...")

    model.fit(X_train, y_train)

    print("Modelo treinado com sucesso!")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n===== MÉTRICAS =====")

    print(f"Accuracy: {accuracy:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\n===== CONFUSION MATRIX =====")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

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

    print(df["focos"].describe())