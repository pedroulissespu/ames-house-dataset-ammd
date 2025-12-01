"""
Configurações do projeto
"""
import os
from pathlib import Path

# Diretórios base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Arquivos de dados
RAW_DATA_FILE = BASE_DIR / "AmesHousing.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_data.csv"

# Arquivos de modelos
MODEL_PKL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_ONNX_PATH = MODELS_DIR / "best_model.onnx"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"

# Configurações de treinamento
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Target variable
TARGET_COLUMN = "SalePrice"

# Features categóricas e numéricas (serão detectadas automaticamente)
CATEGORICAL_FEATURES = []
NUMERICAL_FEATURES = []

# Criar diretórios se não existirem
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
