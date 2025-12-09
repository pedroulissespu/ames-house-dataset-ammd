# Análise Exploratória de Dados com Jupyter Notebook

Este diretório contém notebooks Jupyter para análise e experimentação.

## Arquivo

### `01_eda.ipynb`
Notebook de Análise Exploratória de Dados (EDA).

**Conteúdo:**
1. **Carregamento de Dados**
   - Importação do dataset
   - Visualização inicial
   - Informações básicas

2. **Análise Univariada**
   - Distribuição do target (SalePrice)
   - Distribuições de features numéricas
   - Frequências de features categóricas

3. **Análise Bivariada**
   - Correlações com o target
   - Scatter plots
   - Box plots por categoria

4. **Análise de Valores Ausentes**
   - Identificação de missing values
   - Padrões de ausência
   - Estratégias de imputação

5. **Análise de Outliers**
   - Detecção com IQR
   - Visualização
   - Decisões sobre tratamento

6. **Insights e Conclusões**
   - Features mais importantes
   - Transformações necessárias
   - Próximos passos

## Como Usar

### 1. Instalar Jupyter
```bash
pip install jupyter ipykernel ipywidgets
```

### 2. Abrir o Notebook
```bash
jupyter notebook notebooks/01_eda.ipynb
```

Ou no VS Code:
- Abrir o arquivo `01_eda.ipynb`
- Selecionar kernel Python (ambiente virtual do projeto)
- Executar células

### 3. Executar Análise
- **Executar todas as células:** `Kernel > Restart & Run All`
- **Executar célula por célula:** `Shift + Enter`
- **Adicionar nova célula:** Botão `+` ou `B` (below)

## Principais Visualizações

O notebook gera diversos gráficos:

### Distribuições
- Histograma de preços de venda
- Distribuições de áreas (Gr Liv Area, Lot Area, etc.)
- Distribuições de características (Overall Qual, Year Built, etc.)

### Correlações
- Matriz de correlação (heatmap)
- Correlações com SalePrice (barplot)
- Scatter plots de features vs preço

### Categorical Analysis
- Box plots de preço por categoria
- Bar plots de contagens
- Violin plots

### Missing Data
- Heatmap de valores ausentes
- Porcentagem de missing por feature
- Padrões de ausência

## Insights Principais

**Features mais correlacionadas com preço:**
1. Overall Qual (0.79)
2. Gr Liv Area (0.71)
3. Garage Cars (0.64)
4. Garage Area (0.62)
5. Total Bsmt SF (0.61)

**Valores ausentes significativos:**
- Alley: 93.2%
- Pool QC: 99.5%
- Fence: 80.4%
- Misc Feature: 96.3%

**Outliers detectados:**
- ~8% das casas (137 de 2,930)
- Principalmente em preços muito altos
- Tratados no pipeline de treinamento

## Dependências

O notebook usa estas bibliotecas:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
```

Todas estão no `requirements.txt`.

## Executar no VS Code

1. **Instalar extensão:**
   - Jupyter (Microsoft)
   - Python (Microsoft)

2. **Abrir notebook:**
   - Clicar em `01_eda.ipynb`

3. **Selecionar kernel:**
   - Clicar em "Select Kernel" no topo direito
   - Escolher o ambiente virtual do projeto

4. **Executar:**
   - Clicar no botão play em cada célula
   - Ou `Run All` no topo

## Exportar Resultados

### Para HTML
```bash
jupyter nbconvert --to html notebooks/01_eda.ipynb
```

### Para PDF (requer LaTeX)
```bash
jupyter nbconvert --to pdf notebooks/01_eda.ipynb
```

### Para Python Script
```bash
jupyter nbconvert --to script notebooks/01_eda.ipynb
```

## Adicionar Novos Notebooks

Para criar análises adicionais:

```bash
# Criar novo notebook
jupyter notebook

# Ou no VS Code
# Command Palette (Ctrl+Shift+P)
# > Jupyter: Create New Blank Notebook
```

**Sugestões de notebooks:**
- `02_feature_engineering.ipynb` - Experimentar novas features
- `03_model_comparison.ipynb` - Comparação visual de modelos
- `04_hyperparameter_tuning.ipynb` - Otimização detalhada
- `05_error_analysis.ipynb` - Análise de erros do modelo
