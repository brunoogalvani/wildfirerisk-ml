import sys
from src.data.process_data import process_all_raw_data
from src.features.build_features import build_dataset

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py [process | features | train]")
        return

    comando = sys.argv[1]

    if comando == "process":
        process_all_raw_data()

    elif comando == "features":
        build_dataset()

    elif comando == "train":
        print("Treinando modelo...")

    else:
        print(f"Comando inválido: {comando}")

if __name__ == "__main__":
    main()