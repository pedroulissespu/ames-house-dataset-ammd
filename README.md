# Ames Housing Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Sistema de ML pra prever preço de casas usando o dataset Ames Housing. Tem desde análise dos dados até API funcionando.

---

## Quickstart

Veja o [**QUICKSTART.md**](QUICKSTART.md) com todos os passos detalhados de como usar.

---

## Índice

- [Descrição](#descrição)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Tecnologias](#tecnologias-utilizadas)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Resultados](#resultados-dos-modelos)
- [API](#documentação-da-api)
- [Documentação Adicional](#documentação-adicional)
- [Equipe](#equipe)

---

## Descrição

O objetivo é criar um sistema completo de predição de preços de imóveis utilizando o dataset **Ames Housing**, que contém 2.930 observações e 82 variáveis descrevendo características de casas vendidas em Ames, Iowa.

### Problema de Negócio

Prever o preço de venda de imóveis residenciais com base em diversas características físicas, localização e qualidade da construção. Pode ajudar compradores, vendedores e corretores na tomada de decisões.

### O que foi feito

- Análise Exploratória de Dados (EDA) completa  
- Feature Engineering com criação de 12+ novas features
- Treinamento de 8 modelos diferentes de ML
- Otimização de hiperparâmetros
- Exportação em múltiplos formatos (.pkl, .onnx)
- API REST para servir os modelos
- Documentação

---

## Estrutura do Projeto

```text
ames-house-dataset-ammd/
│
├── AmesHousing.csv              # Dataset original (2,930 casas)
├── README.md                    # Documentação principal (este arquivo)
├── QUICKSTART.md                # Guia rápido de execução
├── requirements.txt             # Dependências Python
├── train.py                     # Script de treinamento
│
├── src/                         # Código-fonte → [Ver README](src/README.md)
│   ├── config.py                # Configurações centralizadas
│   ├── data_preprocessing.py    # Pipeline de limpeza
│   ├── feature_engineering.py   # Criação de features
│   ├── model_training.py        # Treinamento de modelos
│   └── model_export.py          # Exportação (.pkl, .onnx)
│
├── notebooks/                   # Análise exploratória → [Ver README](notebooks/README.md)
│   └── 01_eda.ipynb            # Visualizações e insights
│
├── api/                         # API FastAPI → [Ver README](api/README.md)
│   └── main.py                 # 6 endpoints REST
│
├── models/                      # Modelos treinados → [Ver README](models/README.md)
│   ├── best_model.pkl          # Gradient Boosting (pickle)
│   ├── best_model.onnx         # Modelo ONNX (otimizado)
│   ├── preprocessor.pkl        # Pipeline de transformação
│   ├── feature_names.pkl       # Nomes das features
│   └── training_results.json   # Métricas de todos os modelos
│
├── docs/                        # Documentação → [Ver README](docs/README.md)
│   └── relatorio_tecnico.md    # Relatório completo do projeto
│
└── tests/                       # Suite de testes → [Ver README](tests/README.md)
    ├── test_api_health.py      # Testes básicos da API
    ├── test_api_predictions.py # Testes de predição
    ├── test_api_comprehensive.py  # Suite completa (9 testes)
    └── test_model_performance.py  # Testes do modelo
```

**Cada diretório tem seu próprio README com instruções detalhadas!**

---

## Tecnologias Utilizadas

### Core ML Stack

- **Python 3.8+**
- **pandas** - Manipulação de dados
- **numpy** - Computação numérica
- **scikit-learn 1.4.0** - Machine Learning
- **XGBoost 2.0.0** - Gradient Boosting otimizado
- **LightGBM 4.1.0** - Gradient Boosting eficiente

### Model Serving

- **FastAPI 0.110.0** - Framework web moderno
- **uvicorn** - Servidor ASGI
- **ONNX 1.16.0** - Formato de modelo interoperável
- **onnxruntime** - Runtime para ONNX

### Data Analysis & Visualization

- **matplotlib** - Visualizações estáticas
- **seaborn** - Visualizações estatísticas
- **plotly** - Visualizações interativas
- **Jupyter** - Notebooks interativos

---

## Pipeline de Desenvolvimento

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

## Resultados dos Modelos

### Comparação de Performance

| Modelo | R² (Test) | RMSE (Test) | MAE (Test) | CV R² (Mean) |
|--------|-----------|-------------|------------|--------------|
| Gradient Boosting | 0.9235 | $16,862 | $12,021 | 0.8982 ± 0.018 |
| XGBoost | 0.9212 | $17,123 | $12,093 | 0.8963 ± 0.019 |
| LightGBM | 0.9189 | $17,367 | $12,254 | 0.8958 ± 0.022 |
| Random Forest | 0.9032 | $18,970 | $13,325 | 0.8809 ± 0.020 |
| Ridge | 0.8630 | $22,572 | $13,474 | 0.8214 ± 0.139 |
| Lasso | 0.8402 | $24,379 | $13,141 | 0.8037 ± 0.148 |
| Linear Regression | 0.8282 | $25,278 | $13,450 | 0.7749 ± 0.156 |
| ElasticNet | 0.8258 | $25,450 | $15,558 | 0.8299 ± 0.086 |

**Modelo Selecionado:** Gradient Boosting

- Melhor R² no conjunto de teste (0.9235)
- Menor erro (RMSE e MAE)
- Cross-validation consistente
- Bom equilíbrio entre performance e generalização

### Features Mais Importantes

As características que mais influenciam no preço dos imóveis:

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

## Documentação Adicional

Este projeto possui documentação abrangente organizada por módulos:

### Guias Principais

- **[QUICKSTART.md](QUICKSTART.md)** - Guia passo a passo para executar o projeto do zero
- **[docs/relatorio_tecnico.md](docs/relatorio_tecnico.md)** - Relatório técnico completo do projeto

### Documentação por Módulo

- **[api/README.md](api/README.md)** - Documentação completa da API FastAPI
  - Todos os endpoints disponíveis
  - Exemplos de uso com curl e Python
  - Schemas de request/response
  
- **[src/README.md](src/README.md)** - Módulos de código-fonte
  - Fluxo de execução do pipeline
  - Documentação de cada módulo
  - Configurações disponíveis
  
- **[tests/README.md](tests/README.md)** - Suite de testes
  - Como executar os testes
  - Interpretação dos resultados
  - Adicionar novos testes
  
- **[models/README.md](models/README.md)** - Artefatos dos modelos
  - Descrição de cada arquivo gerado
  - Como carregar e usar os modelos
  - Formatos disponíveis (.pkl, .onnx)
  
- **[notebooks/README.md](notebooks/README.md)** - Análise exploratória
  - Guia do notebook EDA
  - Visualizações disponíveis
  - Como reproduzir a análise

---

## Equipe

**Pedro Ulisses Pereira Castro Maia** e **Caio Henrique De Sousa Guerreiro**
