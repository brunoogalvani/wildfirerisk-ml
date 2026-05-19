from glob import glob
from pathlib import Path
from src.utils.climate_rules import estacao_seca

import pandas as pd

def build_dataset():

    Path("./data/model_ready").mkdir(
        parents=True,
        exist_ok=True
    )

    lista_clima = []

    for caminho in glob("./data/processed/*.csv"):

        df = pd.read_csv(caminho)

        lista_clima.append(df)

        print(
            f"Clima adicionado: "
            f"{Path(caminho).name} → {df.shape}"
        )

    df_clima = pd.concat(
        lista_clima,
        ignore_index=True
    )

    lista_fogo = []

    for caminho in glob("./data/fire_processed/*.csv"):

        df = pd.read_csv(caminho)

        lista_fogo.append(df)

        print(
            f"Fogo adicionado: "
            f"{Path(caminho).name} → {df.shape}"
        )

    df_fogo = pd.concat(
        lista_fogo,
        ignore_index=True
    )

    df_final = df_clima.merge(
        df_fogo,
        on=[
            "uf",
            "cidade",
            "ano",
            "mes"
        ],
        how="left"
    )

    df_final["focos"] = (
        df_final["focos"]
        .fillna(0)
        .astype(int)
    )

    df_final["estacao_seca"] = df_final.apply(
        lambda row: estacao_seca(
            row["uf"],
            row["mes"]
        ),
        axis=1
    )

    df_final["risco"] = (
        df_final["focos"]
        .apply(classificar_risco)
    )

    df_final = df_final[
        [
            "uf",
            "cidade",
            "ano",
            "mes",

            "latitude",
            "longitude",

            "temperatura_c",
            "umidade_relativa_percentual",
            "precipitacao_mmdia",
            "vento_velocidade_ms",

            "estacao_seca",

            "focos",
            "risco"
        ]
    ]

    df_final = df_final.drop_duplicates()

    df_final = df_final.sort_values([
        "uf",
        "cidade",
        "ano",
        "mes"
    ])

    cols_numericas = (
        df_final.select_dtypes(
            include="number"
        ).columns
    )

    df_final[cols_numericas] = (
        df_final[cols_numericas]
        .round(2)
    )

    baixo = df_final[df_final["risco"] == "baixo"].sample(
        n=15000,
        random_state=42
    )

    moderado = df_final[df_final["risco"] == "moderado"]

    alto = df_final[df_final["risco"] == "alto"]

    critico = df_final[df_final["risco"] == "critico"]

    df_final = pd.concat([
        baixo,
        moderado,
        alto,
        critico
    ])

    df_final = df_final.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print("\nDistribuição dos riscos:")
    print(df_final["risco"].value_counts())

    df_final.to_csv(
        "./data/model_ready/dataset_final.csv",
        index=False
    )

    print("\nDataset final criado!")

    print(df_final.shape)

    print("\nPrévia:")

    print(df_final.head())

def classificar_risco(focos):

    if focos == 0:
        return "baixo"

    elif focos <= 5:
        return "moderado"

    elif focos <= 30:
        return "alto"

    return "critico"