from glob import glob
from pathlib import Path

import pandas as pd


def process_fire_data():

    Path("./data/fire_processed").mkdir(
        parents=True,
        exist_ok=True
    )

    estados = [
        "AMAZONAS",
        "PARÁ",
        "ACRE",
        "MATO GROSSO",
        "GOIÁS",
        "MINAS GERAIS"
    ]

    for caminho in glob("./data/fire_raw/*.csv"):

        try:

            df = pd.read_csv(
                caminho,
                encoding="latin1"
            )

            nome_arquivo = Path(caminho).stem

            colunas_utilizadas = [
                "estado",
                "municipio",
                "data_pas",
                "risco_fogo",
                "frp",
                "numero_dias_sem_chuva",
                "precipitacao"
            ]

            if not all(
                col in df.columns
                for col in colunas_utilizadas
            ):
                print(
                    f"Colunas faltando em {nome_arquivo}.csv"
                )
                continue

            df = df[
                df["estado"].isin(estados)
            ]

            df = df[colunas_utilizadas]

            df = df.rename(columns={
                "estado": "uf",
                "municipio": "cidade"
            })

            df["data"] = pd.to_datetime(
                df["data_pas"],
                format="mixed"
            )

            df["ano"] = df["data"].dt.year
            df["mes"] = df["data"].dt.month

            df = df[
                (df["risco_fogo"] != -999)
                &
                (df["numero_dias_sem_chuva"] != -999)
            ]

            df = df.dropna()

            df["uf"] = df["uf"].astype("category")
            df["cidade"] = df["cidade"].astype("category")

            df_grouped = df.groupby([
                "uf",
                "cidade",
                "ano",
                "mes"
            ], observed=True).agg({

                "frp": "mean",

                "risco_fogo": "mean",

                "numero_dias_sem_chuva": "mean",

                "precipitacao": "mean"

            }).reset_index()

            focos = df.groupby([
                "uf",
                "cidade",
                "ano",
                "mes"
            ], observed=True).size().reset_index(
                name="focos"
            )

            df_grouped = df_grouped.merge(
                focos,
                on=[
                    "uf",
                    "cidade",
                    "ano",
                    "mes"
                ]
            )

            colunas_numericas = (
                df_grouped.select_dtypes(
                    include="number"
                ).columns
            )

            df_grouped[colunas_numericas] = (
                df_grouped[colunas_numericas]
                .round(2)
            )

            df_grouped = df_grouped.sort_values([
                "uf",
                "cidade",
                "ano",
                "mes"
            ])

            df_grouped.to_csv(
                f"./data/fire_processed/{nome_arquivo}_processed.csv",
                index=False
            )

            print(
                f"Processado: {nome_arquivo}.csv → {df_grouped.shape}"
            )

        except Exception as e:

            print(
                f"Erro em {caminho}: {e}"
            )