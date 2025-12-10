# Módulo de treinamento de modelos

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
from typing import Dict, Tuple
import json
import warnings

from src.config import RANDOM_STATE, TEST_SIZE, CV_FOLDS, MODEL_PKL_PATH


class ModelTrainer:
    """Classe para treinamento de modelos"""
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_models(self) -> Dict:
        """Retorna dicionário com modelos a serem treinados
        
        """
        models = {
            'Linear Regression': LinearRegression(),  # baseline simples
            
            'Ridge': Ridge(random_state=self.random_state),  # funciona ok
            
            'Lasso': Lasso(random_state=self.random_state),  # feature selection automático
            
            'ElasticNet': ElasticNet(random_state=self.random_state),  # meio termo entre ridge e lasso
            
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            ),
            
            'XGBoost': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'LightGBM': LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        return models
    
    def train_models(self, X_train, y_train, X_test, y_test) -> Dict:
        """
        Treina múltiplos modelos e avalia performance
        """
        self.models = self.get_models()
        
        print("Iniciando treinamento...\n")
        
        for name, model in self.models.items():
            print(f"Treinando {name}...")
            
            # Treinar modelo
            model.fit(X_train, y_train)
            
            # Predições
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métricas
            metrics = {
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test)
            }
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=CV_FOLDS, 
                scoring='r2',
                n_jobs=-1
            )
            metrics['cv_r2_mean'] = cv_scores.mean()
            metrics['cv_r2_std'] = cv_scores.std()
            
            self.results[name] = metrics
            
            print(f"  Test R^2: {metrics['test_r2']:.4f}")
            print(f"  Test RMSE: {metrics['test_rmse']:.2f}")
            print(f"  CV R^2 (mean +- std): {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}\n")
        
        # Identificar melhor modelo
        self.best_model_name = max(self.results.keys(), 
                                   key=lambda x: self.results[x]['test_r2'])
        self.best_model = self.models[self.best_model_name]
        
        print(f"Melhor modelo: {self.best_model_name}")
        print(f"Test R^2: {self.results[self.best_model_name]['test_r2']:.4f}")
        
        return self.results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name: str = None):
        """
        Otimização de hiperparâmetros para o melhor modelo
        """
        if model_name is None:
            model_name = self.best_model_name
        
        print(f"\nOtimizando hiperparâmetros para {model_name}...")
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 50, 70],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        if model_name not in param_grids:
            print(f"Grid search não configurado para {model_name}")
            return self.models[model_name]
        
        base_model = self.get_models()[model_name]
        
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_name],
            cv=CV_FOLDS,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nMelhores parâmetros: {grid_search.best_params_}")
        print(f"Melhor CV R^2: {grid_search.best_score_:.4f}")
        
        self.best_model = grid_search.best_estimator_
        self.models[model_name] = self.best_model
        
        return self.best_model
    
    def save_model(self, filepath: str = None):
        """Salva o melhor modelo"""
        if filepath is None:
            filepath = MODEL_PKL_PATH
        
        joblib.dump(self.best_model, filepath)
        print(f"\nModelo salvo em: {filepath}")
    
    def load_model(self, filepath: str = None):
        """Carrega um modelo salvo"""
        if filepath is None:
            filepath = MODEL_PKL_PATH
        
        self.best_model = joblib.load(filepath)
        print(f"Modelo carregado de: {filepath}")
        return self.best_model
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Retorna resultados em formato DataFrame"""
        return pd.DataFrame(self.results).T.sort_values('test_r2', ascending=False)
    
    def save_results(self, filepath: str):
        """Salva resultados em JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Resultados salvos no arquivo json: {filepath}")


def evaluate_model(model, X_test, y_test) -> Dict:
    """
    Avalia um modelo e retorna métricas
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'R²': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    return metrics, y_pred
