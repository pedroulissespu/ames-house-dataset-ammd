# Guia de Início Rápido

Este guia mostra passo a passo como executar todo o projeto, desde a instalação até fazer predições com a API.

## Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Instalação](#instalação)
3. [Executar Análise Exploratória](#executar-análise-exploratória)
4. [Treinar Modelos](#treinar-modelos)
5. [Testar Modelos](#testar-modelos)
6. [Executar API](#executar-api)
7. [Testar API](#testar-api)
8. [Fazer Predições](#fazer-predições)

---

## Pré-requisitos

- **Python 3.8+** instalado
- **Git** instalado
- **pip** atualizado
- ~500MB de espaço em disco

**Verificar instalações:**
```bash
python --version  # Deve ser 3.8 ou superior
pip --version
git --version
```

---

## Instalação

### 1. Clonar o Repositório

```bash
git clone https://github.com/pedroulissespu/ames-house-dataset-ammd.git
cd ames-house-dataset-ammd
```

### 2. Criar Ambiente Virtual

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

**Confirmar ativação:**
Você deve ver `(.venv)` no início da linha de comando.

### 3. Instalar Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Tempo estimado:** 2-5 minutos

---

## Executar Análise Exploratória

### 1. Abrir Jupyter Notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Explorar o Notebook

O notebook contém:
- Visualização dos dados
- Gráficos de distribuição
- Análise de correlações
- Identificação de outliers
- Insights sobre features importantes

**Tempo estimado:** 10-15 minutos para executar tudo

---

## Treinar Modelos

### 1. Executar Script de Treinamento

```bash
python train.py
```

**O que acontece:**
1. Carrega dados do CSV (2,930 casas)
2. Cria 12+ novas features
3. Trata outliers (remove ~137 casas)
4. Divide em treino (80%) e teste (20%)
5. Treina 8 modelos diferentes
6. Faz validação cruzada (5-fold)
7. Seleciona o melhor modelo
8. Exporta para .pkl e .onnx

**Tempo estimado:** 5-15 minutos (depende do hardware)

**Saída esperada:**
```
================================================================================
AMES HOUSING PRICE PREDICTION - PIPELINE DE TREINAMENTO
================================================================================

[1/7] Carregando dados...
Shape original: (2930, 82)

[2/7] Criando novas feats...
Feats criadas. Shape atual: (2930, 94)

[3/7] Tratando outliers do target...
Removidos 137 outliers

[4/7] Preparando feats e target...
Treino: (2234, 93), Teste: (559, 93)

[5/7] Treinando modelos...
Treinando Linear Regression...
  Test R^2: 0.8282
...
Treinando Gradient Boosting...
  Test R^2: 0.9235  ← MELHOR!
...

Melhor modelo: Gradient Boosting
Test R^2: 0.9235

[7/7] Exportando modelos...
Modelo exportado para ONNX: models/best_model.onnx

================================================================================
PIPELINE CONCLUÍDO COM SUCESSO!
================================================================================
```

### 2. Verificar Arquivos Gerados

```bash
ls models/
```

**Deve conter:**
- `best_model.pkl` (modelo em pickle)
- `best_model.onnx` (modelo em ONNX)
- `preprocessor.pkl` (pipeline de pré-processamento)
- `feature_names.pkl` (nomes das features)
- `training_results.json` (métricas de todos os modelos)

---

## Testar Modelos

### 1. Testar Performance do Modelo

```bash
python tests/test_model_performance.py
```

**Testes realizados:**
- Verificação de existência dos arquivos
- Predições básicas
- Cálculo de métricas (R², RMSE, MAE)
- Consistência das predições

**Saída esperada:**
```
================================================================================
TESTES DE PERFORMANCE DO MODELO
================================================================================

[TEST] Verificando existência do modelo...
✓ Modelo encontrado

[TEST] Testando predição do modelo...
✓ 10 predições realizadas com sucesso
  Preço médio real: $180,796.00
  Preço médio previsto: $178,234.12

[TEST] Testando métricas do modelo...
✓ Métricas calculadas:
  R² Score: 0.9235
  RMSE: $16,862.79
  MAE: $12,021.02
✓ Modelo passou em todas as validações

================================================================================
✓ TODOS OS TESTES PASSARAM!
================================================================================
```

---

## Executar API

### 1. Iniciar o Servidor

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Saída esperada:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
Modelo PKL carregado de models\best_model.pkl
Modelo ONNX carregado de models\best_model.onnx
Preprocessador carregado de models\preprocessor.pkl
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Manter esta janela aberta!** A API está rodando.

### 2. Abrir Documentação Interativa

Abrir no navegador:
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

---

## Testar API

**Abrir uma NOVA janela de terminal** (manter a API rodando na outra).

### 1. Ativar Ambiente Virtual

```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 2. Executar Testes Básicos

```bash
python tests/test_api_health.py
```

**Saída esperada:**
```
================================================================================
TESTANDO API - AMES HOUSING PRICE PREDICTION
================================================================================

[1/3] Testando Health Check...
Status: 200
Resposta: {
  "status": "healthy",
  "models_loaded": {
    "pickle": true,
    "onnx": true,
    "preprocessor": true
  }
}

[3/3] Testando predição com dados do CSV...
Preço real da casa: $215,000.00
Preço previsto: $214,150.38
Diferença: $849.62
Erro percentual: 0.40%

================================================================================
TESTES CONCLUÍDOS!
================================================================================
```

### 3. Executar Testes Completos

```bash
python tests/test_api_comprehensive.py
```

**Testa:**
- Disponibilidade da API
- Todos os endpoints
- Validação de dados
- Performance (tempo de resposta)
- Acurácia (erro percentual)
- Requisições concorrentes

**Saída esperada:**
```
================================================================================
TESTES ABRANGENTES DA API
================================================================================
...
================================================================================
RESULTADO: 9 passaram, 0 falharam
================================================================================
```

---

## Fazer Predições

### Opção 1: Via Interface Web (Swagger)

1. Abrir http://127.0.0.1:8000/docs
2. Expandir `POST /predict/raw`
3. Clicar em "Try it out"
4. Copiar dados de exemplo do CSV ou usar o exemplo fornecido
5. Clicar em "Execute"
6. Ver o resultado!

### Opção 2: Via Python

Criar arquivo `test_prediction.py`:

```python
import requests
import pandas as pd
import math

# Carregar uma casa do dataset
df = pd.read_csv('AmesHousing.csv')
house = df.iloc[0].to_dict()

# Remover campos não-feature
for field in ['Order', 'PID', 'SalePrice']:
    if field in house:
        del house[field]

# Converter NaN para None
for key, value in house.items():
    if isinstance(value, float) and math.isnan(value):
        house[key] = None

# Fazer predição
response = requests.post(
    'http://127.0.0.1:8000/predict/raw',
    json=house
)

if response.status_code == 200:
    result = response.json()
    print(f"Preço previsto: ${result['predicted_price']:,.2f}")
else:
    print(f"Erro: {response.text}")
```

Executar:
```bash
python test_prediction.py
```

### Opção 3: Via curl

```bash
# Health check
curl http://127.0.0.1:8000/health

# Informações dos modelos
curl http://127.0.0.1:8000/models/info
```

---

## Troubleshooting

### Erro: "Modelo não encontrado"
**Solução:** Execute `python train.py` primeiro.

### Erro: "API não está rodando"
**Solução:** Inicie a API com `uvicorn api.main:app --host 127.0.0.1 --port 8000`

### Erro: "Módulo não encontrado"
**Solução:** 
```bash
# Verifique se está no ambiente virtual
# Deve ver (.venv) no prompt

# Reinstale dependências
pip install -r requirements.txt
```

### Erro: "Porta 8000 em uso"
**Solução:**
```bash
# Use outra porta
uvicorn api.main:app --host 127.0.0.1 --port 8001
```
