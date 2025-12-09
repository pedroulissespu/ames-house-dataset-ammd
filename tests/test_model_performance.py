"""
Testes de performance do modelo
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.config import MODEL_PKL_PATH, PREPROCESSOR_PATH, RAW_DATA_FILE
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

def test_model_exists():
    """Verifica se o modelo existe"""
    print("\n[TEST] Verificando existência do modelo...")
    assert MODEL_PKL_PATH.exists(), "Modelo não encontrado! Execute train.py primeiro."
    print("[OK] Modelo encontrado")

def test_preprocessor_exists():
    """Verifica se o preprocessador existe"""
    print("\n[TEST] Verificando existência do preprocessador...")
    assert PREPROCESSOR_PATH.exists(), "Preprocessador não encontrado! Execute train.py primeiro."
    print("[OK] Preprocessador encontrado")

def test_model_prediction():
    """Testa predição do modelo"""
    print("\n[TEST] Testando predição do modelo...")
    
    # Carregar modelo e preprocessador
    model = joblib.load(MODEL_PKL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # Carregar dados
    prep = DataPreprocessor()
    df = prep.load_data(RAW_DATA_FILE)
    
    # Feature engineering
    fe = FeatureEngineer()
    df = fe.create_features(df)
    df = fe.create_interaction_features(df)
    
    # Separar features e target
    X, y = prep.split_features_target(df)
    
    # Pegar amostra pequena
    X_sample = X.head(10)
    y_sample = y.head(10)
    
    # Preprocessar
    X_processed = preprocessor.transform(X_sample)
    
    # Fazer predição
    predictions = model.predict(X_processed)
    
    assert len(predictions) == len(y_sample), "Número de predições não corresponde"
    assert all(p > 0 for p in predictions), "Predições devem ser positivas"
    
    print(f"[OK] {len(predictions)} predições realizadas com sucesso")
    print(f"  Preço médio real: ${y_sample.mean():,.2f}")
    print(f"  Preço médio previsto: ${predictions.mean():,.2f}")

def test_model_metrics():
    """Testa métricas do modelo em uma amostra"""
    print("\n[TEST] Testando métricas do modelo...")
    
    # Carregar modelo e preprocessador
    model = joblib.load(MODEL_PKL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # Carregar dados
    prep = DataPreprocessor()
    df = prep.load_data(RAW_DATA_FILE)
    
    # Feature engineering
    fe = FeatureEngineer()
    df = fe.create_features(df)
    df = fe.create_interaction_features(df)
    
    # Separar features e target
    X, y = prep.split_features_target(df)
    
    # Pegar amostra
    X_sample = X.head(100)
    y_sample = y.head(100)
    
    # Preprocessar
    X_processed = preprocessor.transform(X_sample)
    
    # Fazer predição
    predictions = model.predict(X_processed)
    
    # Calcular métricas
    r2 = r2_score(y_sample, predictions)
    rmse = np.sqrt(mean_squared_error(y_sample, predictions))
    mae = mean_absolute_error(y_sample, predictions)
    
    print(f"[OK] Métricas calculadas:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE: ${mae:,.2f}")
    
    # Validações
    assert r2 > 0.85, f"R² muito baixo: {r2}"
    assert rmse < 30000, f"RMSE muito alto: {rmse}"
    
    print("[OK] Modelo passou em todas as validações")

def test_model_consistency():
    """Testa consistência das predições"""
    print("\n[TEST] Testando consistência das predições...")
    
    model = joblib.load(MODEL_PKL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # Carregar dados
    prep = DataPreprocessor()
    df = prep.load_data(RAW_DATA_FILE)
    
    # Feature engineering
    fe = FeatureEngineer()
    df = fe.create_features(df)
    df = fe.create_interaction_features(df)
    
    X, y = prep.split_features_target(df)
    X_sample = X.head(5)
    
    # Preprocessar
    X_processed = preprocessor.transform(X_sample)
    
    # Fazer predições múltiplas vezes
    pred1 = model.predict(X_processed)
    pred2 = model.predict(X_processed)
    pred3 = model.predict(X_processed)
    
    # Verificar se são idênticas
    assert np.allclose(pred1, pred2), "Predições não são consistentes"
    assert np.allclose(pred2, pred3), "Predições não são consistentes"
    
    print("[OK] Predições são consistentes (determinísticas)")

if __name__ == "__main__":
    print("=" * 80)
    print("TESTES DE PERFORMANCE DO MODELO")
    print("=" * 80)
    
    try:
        test_model_exists()
        test_preprocessor_exists()
        test_model_prediction()
        test_model_metrics()
        test_model_consistency()
        
        print("\n" + "="*80)
        print("[SUCESSO] TODOS OS TESTES PASSARAM!")
        print("="*80)
    except AssertionError as e:
        print(f"\n[ERRO] TESTE FALHOU: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERRO] Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
