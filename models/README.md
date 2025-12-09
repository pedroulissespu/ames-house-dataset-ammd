# Modelos Treinados

Este diretório armazena todos os modelos treinados e artefatos relacionados.

## Arquivos Gerados

### `best_model.pkl`
Melhor modelo treinado em formato pickle.

**Características:**
- Formato: Pickle (joblib)
- Algoritmo: Gradient Boosting Regressor
- Tamanho: ~2-5 MB
- Compatibilidade: Python/scikit-learn

**Como carregar:**
```python
import joblib

model = joblib.load('models/best_model.pkl')
predictions = model.predict(X_processed)
```

### `best_model.onnx`
Modelo em formato ONNX para inferência otimizada.

**Características:**
- Formato: ONNX (Open Neural Network Exchange)
- Tamanho: ~1-3 MB
- Compatibilidade: Multi-plataforma (Python, C++, Java, JavaScript)
- Performance: Inferência mais rápida

**Como carregar:**
```python
import onnxruntime as rt

session = rt.InferenceSession('models/best_model.onnx')
input_name = session.get_inputs()[0].name
predictions = session.run(None, {input_name: X_processed.astype(np.float32)})
```

### `preprocessor.pkl`
Pipeline completo de pré-processamento.

**Conteúdo:**
- Imputadores (numério e categórico)
- StandardScaler para features numéricas
- OneHotEncoder para features categóricas
- Nomes das features após transformação

**Como usar:**
```python
import joblib

preprocessor = joblib.load('models/preprocessor.pkl')

# Transform novos dados
X_processed = preprocessor.transform(X_raw)

# Feature names
feature_names = preprocessor.feature_names_
```

### `feature_names.pkl`
Lista de nomes das features após transformação.

**Conteúdo:**
- 328 nomes de features
- Inclui features originais transformadas
- Inclui features criadas (engineered)
- Inclui features one-hot encoded

**Como usar:**
```python
import joblib

feature_names = joblib.load('models/feature_names.pkl')
print(f"Total de features: {len(feature_names)}")
print(f"Primeiras 10: {feature_names[:10]}")
```

### `training_results.json`
Resultados detalhados do treinamento de todos os modelos.

**Conteúdo:**
Para cada modelo treinado:
- `train_r2`: R² no conjunto de treino
- `test_r2`: R² no conjunto de teste
- `train_rmse`: RMSE no conjunto de treino
- `test_rmse`: RMSE no conjunto de teste
- `train_mae`: MAE no conjunto de treino
- `test_mae`: MAE no conjunto de teste
- `cv_r2_mean`: Média do R² na validação cruzada
- `cv_r2_std`: Desvio padrão do R² na validação cruzada

**Exemplo:**
```json
{
  "Gradient Boosting": {
    "train_r2": 0.9823,
    "test_r2": 0.9235,
    "train_rmse": 7783.99,
    "test_rmse": 16862.79,
    "train_mae": 6118.68,
    "test_mae": 12021.02,
    "cv_r2_mean": 0.8982,
    "cv_r2_std": 0.0179
  }
}
```

**Como usar:**
```python
import json

with open('models/training_results.json', 'r') as f:
    results = json.load(f)

for model_name, metrics in results.items():
    print(f"{model_name}: R² = {metrics['test_r2']:.4f}")
```

## Como os Modelos Foram Gerados

Os modelos são gerados automaticamente ao executar:

```bash
python train.py
```

### Processo de Treinamento

1. **Carregamento de dados**: AmesHousing.csv
2. **Feature Engineering**: Criação de 12+ novas features
3. **Pré-processamento**: Imputação, scaling, encoding
4. **Divisão**: 80% treino, 20% teste
5. **Treinamento**: 8 algoritmos diferentes
6. **Validação cruzada**: 5-fold CV
7. **Seleção**: Melhor modelo por R² no teste
8. **Exportação**: Salva em .pkl e .onnx

## Performance do Melhor Modelo

**Gradient Boosting Regressor**

| Métrica | Treino | Teste | Cross-Validation |
|---------|--------|-------|------------------|
| R² | 0.9823 | 0.9235 | 0.8982 ± 0.018 |
| RMSE | $7,784 | $16,863 | - |
| MAE | $6,119 | $12,021 | - |

**Interpretação:**
- Explica 92.35% da variação nos preços
- Erro médio de ~$16,863 (cerca de 9% do preço médio)
- Validação cruzada consistente (baixo desvio padrão)

## Usando os Modelos em Produção

### Opção 1: Via API (Recomendado)
```bash
# Iniciar API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Fazer predição
curl -X POST http://localhost:8000/predict/raw \
  -H "Content-Type: application/json" \
  -d @house_data.json
```

### Opção 2: Diretamente em Python
```python
import joblib
import pandas as pd
from src.feature_engineering import FeatureEngineer

# Carregar modelo e preprocessador
model = joblib.load('models/best_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Preparar dados
df = pd.read_csv('nova_casa.csv')

# Aplicar feature engineering
fe = FeatureEngineer()
df = fe.create_features(df)
df = fe.create_interaction_features(df)

# Preprocessar
X = preprocessor.transform(df)

# Predizer
price = model.predict(X)[0]
print(f"Preço estimado: ${price:,.2f}")
```

### Opção 3: Com ONNX (Performance)
```python
import onnxruntime as rt
import numpy as np

# Carregar sessão ONNX
session = rt.InferenceSession('models/best_model.onnx')

# Preparar dados (mesmo processo)
# ... (feature engineering e preprocessing)

# Predição
input_name = session.get_inputs()[0].name
predictions = session.run(None, {input_name: X.astype(np.float32)})[0]
```

## Versionamento

Para manter versões diferentes dos modelos:

```bash
# Copiar modelo atual
cp models/best_model.pkl models/best_model_v1.0.pkl
cp models/training_results.json models/training_results_v1.0.json

# Adicionar timestamp
cp models/best_model.pkl models/best_model_2025-12-09.pkl
```

## Retreinamento

Para retreinar os modelos com novos dados:

```bash
# 1. Adicionar novos dados ao AmesHousing.csv (ou criar novo arquivo)
# 2. Ajustar src/config.py se necessário
# 3. Executar treinamento
python train.py
```

Os modelos antigos serão sobrescritos. Faça backup se necessário!
