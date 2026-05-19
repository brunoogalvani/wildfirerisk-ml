import sys
import uvicorn

from src.data.process_data import process_all_raw_data
from src.data.process_fire_data import process_fire_data
from src.features.build_dataset import build_dataset
from src.models.train_model import train_model

def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py [process | process-fire | build | train | api]")
        return

    comando = sys.argv[1]

    if comando == "process":
        process_all_raw_data()

    elif comando == "process-fire":
        process_fire_data()

    elif comando == "build":
        build_dataset()

    elif comando == "train":
        train_model()

    elif comando == "api":
        print("Iniciando API...")
        uvicorn.run(
            "src.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )

    else:
        print(f"Comando inválido: {comando}")

if __name__ == "__main__":
    main()