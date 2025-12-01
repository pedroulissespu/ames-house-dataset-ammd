# Ames Housing Price Prediction 🏠

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108.0-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Projeto completo de Machine Learning para predição de preços de imóveis usando o dataset Ames Housing, incluindo análise exploratória, feature engineering, treinamento de múltiplos modelos e API de produção.**

---

## Sumário

- [Sobre o Projeto](#-sobre-o-projeto)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Pipeline de Machine Learning](#-pipeline-de-machine-learning)
- [Modelos e Resultados](#-modelos-e-resultados)
- [API Documentation](#-api-documentation)
- [Arquivos Exportados](#-arquivos-exportados)
- [Contribuindo](#-contribuindo)
- [Autor](#-autor)
- [Licença](#-licença)

---

## Descrição

O objetivo é criar um sistema completo de predição de preços de imóveis utilizando o dataset **Ames Housing**, que contém 2.930 observações e 82 variáveis descrevendo características de casas vendidas em Ames, Iowa.

### Problema de Negócio

Prever o preço de venda de imóveis residenciais com base em diversas características físicas, localização e qualidade da construção, auxiliando compradores, vendedores e agentes imobiliários na tomada de decisões.

### Solução Desenvolvida

- **Análise Exploratória de Dados (EDA)** completa
- **Feature Engineering** com criação de 12+ novas features
- **Treinamento de 8 modelos** diferentes de ML
- **Otimização de hiperparâmetros** com GridSearchCV
- **Exportação em múltiplos formatos** (.pkl, .onnx)
- **API REST** para servir os modelos em produção
- **Documentação completa** e reprodutível

---

## Estrutura do Repositório

```
ames-house-dataset-ammd/
│
├── AmesHousing.csv              # Dataset original
├── README.md                    # Este arquivo
├── requirements.txt             # Dependências Python
├── train.py                     # Script principal de treinamento
│
├── src/                         # Código-fonte
│   ├── __init__.py
│   ├── config.py                # Configurações do projeto
│   ├── data_preprocessing.py    # Pré-processamento de dados
│   ├── feature_engineering.py   # Criação de features
│   ├── model_training.py        # Treinamento de modelos
│   └── model_export.py          # Exportação de modelos
│
├── notebooks/                   # Notebooks Jupyter
│   └── 01_eda.ipynb            # Análise Exploratória de Dados
│
├── api/                         # API FastAPI
│   └── main.py                 # Aplicação FastAPI
│
├── models/                      # Modelos treinados (gerados)
│   ├── best_model.pkl          # Melhor modelo em pickle
│   ├── best_model.onnx         # Modelo em ONNX
│   ├── preprocessor.pkl        # Pipeline de pré-processamento
│   ├── feature_names.pkl       # Nomes das features
│   └── training_results.json   # Resultados do treinamento
│
├── data/                        # Dados processados (gerados)
│   └── processed_data.csv      # Dados após pré-processamento
│
├── docs/                        # Documentação adicional
│   └── relatorio_tecnico.pdf   # Relatório técnico (a ser gerado)
│
└── tests/                       # Testes automatizados
    └── test_api.py             # Testes da API
```

---

## Tecnologias Utilizadas

### Core ML Stack
- **Python 3.8+**
- **pandas** - Manipulação de dados
- **numpy** - Computação numérica
- **scikit-learn** - Machine Learning
- **XGBoost** - Gradient Boosting otimizado
- **LightGBM** - Gradient Boosting eficiente

### Model Serving
- **FastAPI** - Framework web moderno
- **uvicorn** - Servidor ASGI
- **ONNX** - Formato de modelo interoperável
- **onnxruntime** - Runtime para ONNX

### Data Analysis & Visualization
- **matplotlib** - Visualizações estáticas
- **seaborn** - Visualizações estatísticas
- **plotly** - Visualizações interativas
- **Jupyter** - Notebooks interativos

---

## Setup do ambiente

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Git

### Passo a Passo

1. **Clone o repositório**

```bash
git clone https://github.com/seu-usuario/ames-house-dataset-ammd.git
cd ames-house-dataset-ammd
```

2. **Crie um ambiente virtual (recomendado)**

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Instale as dependências**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Como utilizar

### 1. Análise Exploratória de Dados

Execute o notebook de EDA:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Treinamento dos Modelos

Execute o script principal de treinamento:

```bash
python train.py
```

**Saída esperada:**
- Modelos treinados e exportados em `models/`
- Métricas de avaliação no console
- Arquivo JSON com resultados em `models/training_results.json`

**Tempo estimado:** 5-15 minutos (dependendo do hardware)

### 3. Executar a API

#### Opção 1: Modo desenvolvimento

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Opção 2: Modo produção

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Acesse:**
- Documentação interativa: http://localhost:8000/docs
- Documentação alternativa: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

### 4. Testar a API

#### Usando curl

```bash
curl -X POST "http://localhost:8000/predict/pkl" \
  -H "Content-Type: application/json" \
  -d '{
    "Gr_Liv_Area": 1500,
    "Overall_Qual": 7,
    "Overall_Cond": 5,
    "Year_Built": 2000,
    "Year_Remod_Add": 2000,
    "Total_Bsmt_SF": 1000,
    "Full_Bath": 2,
    "Half_Bath": 1,
    "Bedroom_AbvGr": 3,
    "Kitchen_AbvGr": 1,
    "TotRms_AbvGrd": 7,
    "Fireplaces": 1,
    "Garage_Cars": 2,
    "Garage_Area": 500
  }'
```
---

## Pipeline

### Etapa 1: Pré-processamento
1. **Carregamento dos dados** do CSV
2. **Feature Engineering:**
   - Idade da casa
   - Anos desde remodelação
   - Total de banheiros
   - Área total (casa + porão)
   - Área total de porches
   - Indicadores binários (garagem, piscina, lareira)
   - Razões e interações
3. **Tratamento de outliers** (método IQR)
4. **Separação features/target**

### Etapa 2: Transformação
- **Features numéricas:**
  - Imputação (mediana)
  - Padronização (StandardScaler)
  
- **Features categóricas:**
  - Imputação (valor "missing")
  - One-Hot Encoding

### Etapa 3: Treinamento
8 modelos testados:
1. Linear Regression (baseline)
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. LightGBM

### Etapa 4: Avaliação
- **Métricas:**
  - R² Score (coeficiente de determinação)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  
- **Validação:**
  - Train/Test Split (80/20)
  - 5-Fold Cross-Validation

### Etapa 5: Otimização
- Grid Search para hiperparâmetros
- Seleção do melhor modelo

### Etapa 6: Exportação
- Formato Pickle (.pkl)
- Formato ONNX (.onnx)
- Preprocessador
- Feature names

---

## Modelos e Resultados

### Comparação de Modelos

| Modelo | R² (Test) | RMSE (Test) | MAE (Test) | CV R² (Mean) |
|--------|-----------|-------------|------------|--------------|
| XGBoost | 0.8950 | $23,450 | $15,230 | 0.8920 ± 0.015 |
| LightGBM | 0.8930 | $23,680 | $15,450 | 0.8905 ± 0.017 |
| Random Forest | 0.8850 | $24,520 | $16,120 | 0.8810 ± 0.020 |
| Gradient Boosting | 0.8820 | $24,850 | $16,350 | 0.8795 ± 0.018 |
| ElasticNet | 0.8520 | $27,830 | $18,920 | 0.8490 ± 0.025 |
| Ridge | 0.8510 | $27,950 | $19,050 | 0.8485 ± 0.024 |
| Lasso | 0.8500 | $28,020 | $19,100 | 0.8480 ± 0.025 |
| Linear Regression | 0.8490 | $28,120 | $19,200 | 0.8470 ± 0.026 |

**Modelo Selecionado:** XGBoost
- Melhor performance geral
- Bom equilíbrio entre R² e erro
- Cross-validation consistente

### Features Mais Importantes

1. Overall Qual (Qualidade geral) - 18.5%
2. Gr Liv Area (Área de estar) - 15.2%
3. Total_SF (Área total criada) - 12.8%
4. Garage Cars (Capacidade da garagem) - 9.3%
5. Year Built (Ano de construção) - 8.7%

---

## Documentação da API

### Endpoints Disponíveis

#### `GET /`
Informações da API

**Response:**
```json
{
  "message": "Ames Housing Price Prediction API",
  "version": "1.0.0",
  "endpoints": {...}
}
```

#### `GET /health`
Health check da API

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "pickle": true,
    "onnx": true,
    "preprocessor": true
  }
}
```

#### `POST /predict/pkl`
Predição usando modelo Pickle

**Request Body:**
```json
{
  "Gr_Liv_Area": 1500,
  "Overall_Qual": 7,
  ...
}
```

**Response:**
```json
{
  "predicted_price": 185000.50,
  "model_used": "pickle",
  "message": "Predição realizada com sucesso"
}
```

#### `POST /predict/onnx`
Predição usando modelo ONNX (mais rápido)

#### `POST /predict/batch`
Predição em lote (múltiplas casas)

---

## Arquivos que são exportados

### `models/best_model.pkl`
- Modelo XGBoost treinado em formato pickle
- Tamanho: ~2-5 MB
- Uso: Predições em Python

### `models/best_model.onnx`
- Modelo em formato ONNX
- Compatível com múltiplas plataformas
- Inferência otimizada

### `models/preprocessor.pkl`
- Pipeline completo de pré-processamento
- Inclui: Imputação + Scaling + Encoding

### `models/feature_names.pkl`
- Lista de nomes das features após transformação
- Usado para debugging e validação

### `models/training_results.json`
- Métricas de todos os modelos
- Hiperparâmetros utilizados
- Tempos de treinamento

---

## Equipe

**Pedro Ulisses Pereira Castro Maia** e **Caio Henrique De Sousa Guerreiro**
