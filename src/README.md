# Código Fonte

Este diretório contém todo o código-fonte do projeto, organizado em módulos.

## Arquivos

### `config.py`
Configurações centralizadas do projeto.

**Conteúdo:**
- Caminhos de arquivos (dados, modelos)
- Parâmetros de treinamento (random_state, test_size)
- Configurações de validação cruzada
- Nome da variável target

**Exemplo:**
```python
from src.config import RANDOM_STATE, TEST_SIZE, MODEL_PKL_PATH
```

### `data_preprocessing.py`
Classe `DataPreprocessor` para pré-processamento de dados.

**Funcionalidades:**
- Carregamento de dados CSV
- Identificação automática de features numéricas e categóricas
- Criação de pipeline de transformação:
  - Features numéricas: Imputação (mediana) + StandardScaler
  - Features categóricas: Imputação ("missing") + OneHotEncoder
- Separação de features e target
- Tratamento de outliers (método IQR)
- Salvamento e carregamento do preprocessador

**Exemplo de uso:**
```python
from src.data_preprocessing import DataPreprocessor

prep = DataPreprocessor()
df = prep.load_data('AmesHousing.csv')
X, y = prep.split_features_target(df)

num_features, cat_features = prep.identify_feature_types(X)
prep.create_preprocessor(num_features, cat_features)

X_train_processed = prep.fit_transform(X_train, y_train)
X_test_processed = prep.transform(X_test)
```

### `feature_engineering.py`
Classe `FeatureEngineer` para criação de novas features.

**Features criadas:**

1. **House_Age:** Idade da casa (2025 - Year Built)
2. **Years_Since_Remod:** Anos desde a última remodelação
3. **Total_Bathrooms:** Total de banheiros (Full + Half*0.5 + Basement)
4. **Total_SF:** Área total (Gr Liv Area + Total Bsmt SF)
5. **Total_Porch_SF:** Área total de varandas
6. **Overall_Score:** Qualidade × Condição
7. **Lot_To_Living_Ratio:** Razão entre área do lote e área construída
8. **Has_Garage:** Indicador de presença de garagem
9. **Has_Pool:** Indicador de presença de piscina
10. **Has_Fireplace:** Indicador de presença de lareira
11. **Sale_Season:** Temporada de venda (Winter/Spring/Summer/Fall)
12. **Qual_Area_Interaction:** Overall Qual × Gr Liv Area
13. **Age_Qual_Interaction:** House Age × Overall Qual

**Exemplo de uso:**
```python
from src.feature_engineering import FeatureEngineer

fe = FeatureEngineer()
df = fe.create_features(df)
df = fe.create_interaction_features(df)
```

### `model_training.py`
Classe `ModelTrainer` para treinamento e avaliação de modelos.

**Modelos implementados:**
1. Linear Regression
2. Ridge
3. Lasso
4. ElasticNet
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. LightGBM

**Funcionalidades:**
- Treinamento de múltiplos modelos
- Validação cruzada (5-fold)
- Cálculo de métricas (R², RMSE, MAE, MAPE)
- Seleção automática do melhor modelo
- Otimização de hiperparâmetros (GridSearchCV)
- Salvamento de modelos e resultados

**Exemplo de uso:**
```python
from src.model_training import ModelTrainer

trainer = ModelTrainer(random_state=42)
results = trainer.train_models(X_train, y_train, X_test, y_test)

# Otimização (opcional)
trainer.hyperparameter_tuning(X_train, y_train)

# Salvar
trainer.save_model()
trainer.save_results('results.json')
```

### `model_export.py`
Classe `ModelExporter` para exportação de modelos.

**Formatos suportados:**
- **Pickle (.pkl):** Formato nativo do Python
- **ONNX (.onnx):** Formato interoperável multi-plataforma

**Funcionalidades:**
- Exportação para pickle
- Conversão para ONNX (com skl2onnx)
- Carregamento de modelos ONNX
- Verificação de consistência (sklearn vs ONNX)
- Predições com ONNX Runtime

**Exemplo de uso:**
```python
from src.model_export import ModelExporter

exporter = ModelExporter()

# Exportar para ONNX
onnx_path = exporter.export_to_onnx(model, X_sample)

# Carregar e verificar
onnx_session = exporter.load_onnx_model(onnx_path)
exporter.verify_onnx_export(model, onnx_session, X_test)
```

## Fluxo de Uso Típico

```python
# 1. Configuração
from src.config import RAW_DATA_FILE, RANDOM_STATE

# 2. Preprocessamento
from src.data_preprocessing import DataPreprocessor, handle_outliers
prep = DataPreprocessor()
df = prep.load_data(RAW_DATA_FILE)
df = handle_outliers(df, 'SalePrice')

# 3. Feature Engineering
from src.feature_engineering import FeatureEngineer
fe = FeatureEngineer()
df = fe.create_features(df)
df = fe.create_interaction_features(df)

# 4. Preparação
X, y = prep.split_features_target(df)
num_feat, cat_feat = prep.identify_feature_types(X)
prep.create_preprocessor(num_feat, cat_feat)

X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
X_train_proc = prep.fit_transform(X_train, y_train)
X_test_proc = prep.transform(X_test)

# 5. Treinamento
from src.model_training import ModelTrainer
trainer = ModelTrainer(random_state=RANDOM_STATE)
results = trainer.train_models(X_train_proc, y_train, X_test_proc, y_test)

# 6. Exportação
from src.model_export import ModelExporter
exporter = ModelExporter()
exporter.export_to_onnx(trainer.best_model, X_sample)
```

## Princípios de Design

- **Modularidade:** Cada arquivo tem responsabilidade única
- **Reusabilidade:** Classes podem ser usadas independentemente
- **Configurabilidade:** Parâmetros centralizados em config.py
- **Testabilidade:** Código estruturado para facilitar testes
- **Documentação:** Docstrings em todas as funções e classes
