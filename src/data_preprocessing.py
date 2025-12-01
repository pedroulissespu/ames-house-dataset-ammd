"""
Módulo de pré-processamento de dados
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple
import joblib

from src.config import TARGET_COLUMN, PREPROCESSOR_PATH


class DataPreprocessor:
    """Classe para pré-processamento dos dados"""
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Carrega os dados do CSV"""
        df = pd.read_csv(filepath)
        print(f"Dados que foram carregados: {df.shape}")
        return df
    
    def split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separa features e target"""
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        # Remove colunas desnecessárias (Order e PID são apenas identificadores)
        if 'Order' in X.columns:
            X = X.drop(columns=['Order'])
        if 'PID' in X.columns:
            X = X.drop(columns=['PID'])
            
        return X, y
    
    def identify_feature_types(self, X: pd.DataFrame) -> Tuple[list, list]:
        """Identifica feats numéricas e categóricas"""
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Feats de nmúeros: {len(numerical_features)}")
        print(f"Feats de categórias: {len(categorical_features)}")
        
        return numerical_features, categorical_features
    
    def create_preprocessor(self, numerical_features: list, categorical_features: list):
        """Cria o pipeline de pré-processamento"""
        
        # Pipeline para features numéricas
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Pipeline para as features de categorias
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combinar os pipelines
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return self.preprocessor
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Ajusta e transforma os dados"""
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Salvar nomes das features
        self.feature_names = self._get_feature_names()
        
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforma os dados usando preprocessor já ajustado"""
        return self.preprocessor.transform(X)
    
    def _get_feature_names(self) -> list:
        """Obtém os nomes das features após transformação"""
        feature_names = []
        
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                    cat_features = transformer.named_steps['onehot'].get_feature_names_out(features)
                    feature_names.extend(cat_features)
        
        return feature_names
    
    def save_preprocessor(self, filepath: str = None):
        """Salva o preprocessor"""
        if filepath is None:
            filepath = PREPROCESSOR_PATH
        joblib.dump(self.preprocessor, filepath)
        print(f"preprocesaomento salvo no seguinte arquivo: {filepath}")
    
    def load_preprocessor(self, filepath: str = None):
        """Carrega o preprocessor"""
        if filepath is None:
            filepath = PREPROCESSOR_PATH
        self.preprocessor = joblib.load(filepath)
        print(f"preprocesasmento carregado de: {filepath}")
        return self.preprocessor


def handle_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Remove ou trata outliers
    
    Args:
        df: DataFrame
        column: Nome da coluna
        method: Método ('iqr' ou 'zscore')
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df_clean = df[z_scores < 3]
    
    else:
        df_clean = df
    
    removed = len(df) - len(df_clean)
    if removed > 0:
        print(f"Removidos {removed} outliers da coluna {column}")
    
    return df_clean
