from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def build_dataset():

    Path("./data/model_ready").mkdir(parents=True, exist_ok=True)

    lista_dfs = []

    for caminho in glob("./data/processed/*.csv"):

        df = pd.read_csv(caminho)

        lista_dfs.append(df)

        print(f"Adicionado: {Path(caminho).name} → {df.shape}")

    df = pd.concat(lista_dfs, ignore_index=True)

    df = df.sort_values([
        "uf",
        "cidade",
        "ano",
        "mes"
    ])

    scaler = MinMaxScaler()

    colunas_climaticas = [
        "temperatura_c",
        "umidade_relativa_percentual",
        "precipitacao_mmdia",
        "vento_velocidade_ms"
    ]

    df_normalized = scaler.fit_transform(df[colunas_climaticas])

    df_normalized = pd.DataFrame(
        df_normalized,
        columns=[
            "temp_norm",
            "umidade_norm",
            "precipitacao_norm",
            "vento_norm"
        ]
    )

    df["risco_queimada"] = (
        (df_normalized["temp_norm"] * 0.4)
        +
        (df_normalized["vento_norm"] * 0.2)
        +
        ((1 - df_normalized["umidade_norm"]) * 0.3)
        +
        ((1 - df_normalized["precipitacao_norm"]) * 0.1)
    )

    df["risco_queimada"] = df["risco_queimada"].clip(0, 1)

    df["risco_queimada"] = df["risco_queimada"].round(2)

    df.to_csv(
        "./data/model_ready/dataset_final.csv",
        index=False
    )

    print("\nDataset final criado com sucesso!")
    print(df.shape)

    print("\nPrévia:")
    print(df.head())