# API - Ames Housing Price Prediction

Este diretório contém a implementação da API REST usando FastAPI.

## Arquivos

### `main.py`
Aplicação principal da API que expõe os modelos de predição através de endpoints HTTP.

**Principais componentes:**
- Carregamento automático dos modelos na inicialização
- Endpoints para predição (pickle e ONNX)
- Health check e informações dos modelos
- Validação de dados com Pydantic

## Endpoints Disponíveis

### `GET /`
Retorna informações gerais da API e lista de endpoints disponíveis.

### `GET /health`
Verifica o status de saúde da API e se os modelos estão carregados.

**Resposta:**
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

### `GET /models/info`
Retorna informações detalhadas sobre os modelos carregados.

**Resposta:**
```json
{
  "pickle_model": {
    "loaded": true,
    "type": "GradientBoostingRegressor"
  },
  "onnx_model": {
    "loaded": true
  },
  "preprocessor": {
    "loaded": true
  },
  "num_features": 328
}
```

### `POST /predict/pkl`
Faz predição usando o modelo pickle (schema simplificado).

**Limitação:** Aceita apenas as principais features (14 campos).

### `POST /predict/onnx`
Faz predição usando o modelo ONNX (mais rápido).

**Limitação:** Aceita apenas as principais features (14 campos).

### `POST /predict/raw` ⭐ RECOMENDADO
Faz predição usando dados completos do CSV.

**Vantagem:** Aceita todos os 82 campos do dataset original.

**Request Body:**
```json
{
  "MS SubClass": 20,
  "MS Zoning": "RL",
  "Lot Frontage": 141,
  "Lot Area": 31770,
  ...
}
```

**Response:**
```json
{
  "predicted_price": 214150.38,
  "model_used": "pickle",
  "message": "Predição realizada com sucesso (endpoint raw)"
}
```

### `POST /predict/batch`
Faz predições em lote para múltiplas casas.

## Como Executar

### 1. Certifique-se de que os modelos foram treinados
```bash
python train.py
```

### 2. Inicie a API

**Modo desenvolvimento (com auto-reload):**
```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

**Modo produção:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Acesse a documentação interativa

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Exemplo de Uso

### Python
```python
import requests
import pandas as pd

# Carregar dados de exemplo
df = pd.read_csv('AmesHousing.csv')
house = df.iloc[0].drop(['Order', 'PID', 'SalePrice']).to_dict()

# Fazer predição
response = requests.post(
    'http://127.0.0.1:8000/predict/raw',
    json=house
)

if response.status_code == 200:
    price = response.json()['predicted_price']
    print(f"Preço previsto: ${price:,.2f}")
```

### curl
```bash
curl -X POST "http://127.0.0.1:8000/health" -H "accept: application/json"
```

## Dependências

- FastAPI: Framework web moderno e rápido
- Uvicorn: Servidor ASGI
- Pydantic: Validação de dados
- ONNX Runtime: Para modelos ONNX (opcional)
