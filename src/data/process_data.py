from glob import glob
import pandas as pd
from pathlib import Path
from src.utils.climate_rules import name_to_uf

def process_all_raw_data():
    
    Path("./data/processed").mkdir(parents=True, exist_ok=True)
    estados = ["AM", "PA", "AC", "MT", "GO", "MG"]
    colunas_utilizadas = [
        "datahora",
        "uf_nome",
        "municipio_nome",
        "longitude",
        "latitude",
        "precipitacao_mmdia",
        "temperatura_c",
        "umidade_relativa_percentual",
        "vento_velocidade_ms"
    ]

    for caminho in glob("./data/raw/*.csv"):
        try:
            df = pd.read_csv(caminho)
            nomeArquivo = Path(caminho).stem

            if not all(col in df.columns for col in colunas_utilizadas):
                print(f"Colunas faltando em {nomeArquivo}.csv — ignorado")
                continue

            df = df[colunas_utilizadas].dropna()

            df = df.rename(columns={"uf_nome": "uf", "municipio_nome": "cidade"})

            df["uf"] = df["uf"].apply(name_to_uf)

            df = df[df["uf"].isin(estados)]

            df["data"] = pd.to_datetime(df["datahora"])
            df["ano"] = df["data"].dt.year
            df["mes"] = df["data"].dt.month

            df["uf"] = df["uf"].astype("category")
            df["cidade"] = df["cidade"].astype("category")

            df = df.groupby([
                "uf",
                "cidade",
                "ano",
                "mes"
            ], observed=True).mean(numeric_only=True).reset_index()

            cols_to_round = df.select_dtypes(include="number").columns
            cols_to_round = cols_to_round.difference(["latitude", "longitude"])

            df[cols_to_round] = df[cols_to_round].round(2)

            df = df.sort_values(["uf", "cidade", "ano", "mes"])

            df.to_csv(f"./data/processed/{nomeArquivo}_processed.csv", index=False)
            print(f"Processado: {nomeArquivo}.csv → {df.shape}")
        except Exception as e:
            print(f"Erro em '{caminho}': {e}")
            continue
