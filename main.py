import sys
from src.data.process_data import process_all_raw_data

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py [process | features | train]")
        return

    comando = sys.argv[1]

    if comando == "process":
        process_all_raw_data()

    elif comando == "features":
        print("Gerando features...")

    elif comando == "train":
        print("Treinando modelo...")

    else:
        print(f"Comando inválido: {comando}")

if __name__ == "__main__":
    main()