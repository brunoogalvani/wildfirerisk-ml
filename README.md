# Wildfire Risk Prediction API

Projeto de Machine Learning focado na predição de risco de queimadas utilizando dados históricos de focos de incêndio e indicadores climáticos de estados brasileiros.

O projeto realiza:
- processamento de queimadas
- engenharia de features
- treinamento de modelo de Machine Learning
- disponibilização das previsões através de uma API REST com FastAPI

# Funcionalidades

- Pipeline de Machine Learning para queimadas
- Limpeza e normalização de dados
- Agregação mensal por cidade/estado
- Engenharia de features
- Classificação de risco de queimadas
- Modelo Random Forest
- API REST com FastAPI
- Documentação automática com Swagger

# Tecnologias Utilizadas

- Python
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib

# Estrutura do Projeto

```bash
wildfirerisk-ml/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── fire_raw/
│   ├── fire_processed/
│   └── model_ready/
│
├── models/
│
├── src/
│   ├── api/
│   │   └── app.py
│   │
│   ├── data/
│   │   ├── process_data.py
│   │   └── process_fire_data.py
│   │
│   ├── features/
│   │   └── build_dataset.py
│   │
│   ├── models/
│   │   └── train_model.py
│   │
│   └── utils/
│       └── climate_rules.py
│
├── main.py
└── README.md
```

# Fontes de Dados

O projeto utiliza:
- Dados históricos de focos de queimadas
- Dados climáticos contendo:
    - temperatura
    - umidade relativa
    - precipitação
    - velocidade do vento

Estados analisados:
- Amazonas (AM)
- Pará (PA)
- Acre (AC)
- Mato Grosso (MT)
- Goiás (GO)
- Minas Gerais (MG)

# Pipeline de Machine Learning

## Processamento dos Dados Climáticos

Processa os arquivos CSV climáticos brutos:

```bash
python main.py process
```

Saída:
- ```data/processed/*.csv```

## Processamento dos Dados de Queimadas

Processa os datasets de focos de incêndio:

```bash
python main.py process-fire
```

Saída:
- ```data/fire_processed/*.csv```

## Construção do Dataset Final

Realiza o merge entre os dados climáticos e os dados de queimadas, além da engenharia de features.

```bash
python main.py build
```

Saída:
- ```data/model_ready/dataset_final.csv```

Features geradas:
- temperatura
- umidade relativa
- precipitação
- velocidade do vento
- indicador de estação seca
- classificação de risco

## Treinamento do Modelo

Treina o modelo Random Forest para classificação de risco.

```bash
python main.py train
```

Saída:
- ```models/wildfire_model.pkl```

# Executando a API

Inicie a API com:

```bash
python main.py api
```

# Documentação Swagger

Após iniciar a API:

```bash
http://localhost:8000/docs
```

# Endpoint da API

### POST /wildfire-risk

Realiza a previsão do risco de queimadas.

### Body da Requisição

```json
{
  "uf": "MATO GROSSO",
  "latitude": -10.0,
  "longitude": -55.0,
  "mes": 8,
  "temperatura_c": 41,
  "umidade_relativa_percentual": 9,
  "precipitacao_mmdia": 0,
  "vento_velocidade_ms": 8
}
```

# Exemplo de Resposta

```json
{
  "risk_level": "critico",
  "description": "Critical wildfire risk",
  "probabilities": {
    "alto": 0.13,
    "baixo": 0.23,
    "critico": 0.48,
    "moderado": 0.16
  }
}
```

# Instalação

### Clonando o repositório

```bash
git clone https://github.com/brunoogalvani/wildfirerisk-ml.git
```