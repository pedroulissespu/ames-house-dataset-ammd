"""
Script principal de treinamento do pipeline completo
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent))

from src.config import (
    RAW_DATA_FILE, RANDOM_STATE, TEST_SIZE, 
    MODELS_DIR, TARGET_COLUMN
)
from src.data_preprocessing import DataPreprocessor, handle_outliers
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_export import ModelExporter, export_full_pipeline


def main():
    """Executa o pipeline completo de treinamento"""
    
    print("="*80)
    print("AMES HOUSING PRICE PREDICTION - PIPELINE DE TREINAMENTO")
    print("="*80)
    
    # 1. CARREGAR DADOS
    print("\n[1/7] Carregando dados...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(RAW_DATA_FILE)
    
    print(f"Shape original: {df.shape}")
    print(f"Valores ausentes:\n{df.isnull().sum().sum()} no total")
    
    # 2. FEATURE ENGINEERING
    print("\n[2/7] Criando novas feats...")
    engineer = FeatureEngineer()
    df = engineer.create_features(df)
    df = engineer.create_interaction_features(df)
    
    # 3. TRATAR OUTLIERS DO TARGET
    print("\n[3/7] Tratando outliers do target...")
    df_clean = handle_outliers(df, TARGET_COLUMN, method='iqr')
    
    # 4. SEPARAR FEATURES E TARGET
    print("\n[4/7] Preparando feats e target...")
    X, y = preprocessor.split_features_target(df_clean)
    
    # Identificar tipos de features
    numerical_features, categorical_features = preprocessor.identify_feature_types(X)
    
    # Criar preprocessador
    preprocessor.create_preprocessor(numerical_features, categorical_features)
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    
    # Transformar dados
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Shape após pré-processamento: {X_train_processed.shape}")
    
    # Salvar preprocessador
    preprocessor.save_preprocessor()
    
    # 5. TREINAR MODELOS
    print("\n[5/7] Treinando modelos...")
    print("-"*80)
    
    trainer = ModelTrainer(random_state=RANDOM_STATE)
    results = trainer.train_models(
        X_train_processed, y_train,
        X_test_processed, y_test
    )
    
    # Mostrar resultados
    print("\n" + "="*80)
    print("RESULTADOS DOS MODELOS")
    print("="*80)
    results_df = trainer.get_results_dataframe()
    print(results_df[['test_r2', 'test_rmse', 'test_mae', 'cv_r2_mean']].to_string())
    
    # Salvar resultados
    trainer.save_results(MODELS_DIR / "training_results.json")
    
    # 6. OTIMIZAÇÃO DE HIPERPARÂMETROS (opcional)
    print("\n[6/7] Otimizando hiperparâmetros do melhor modelo...")
    
    # trainer.hyperparameter_tuning(X_train_processed, y_train)
    # Caso não precise da otimização desabilitada, pode descomentar a linha de código acima
    # Deixei comentada pois eu estou com pressa para testar, podemos revisitar esse campo depois
    # Vou deixar um to-do
    # TODO : testar com a otimização habilitada
    
    # 7. EXPORTAR MODELOS
    print("\n[7/7] Exportando modelos...")
    
    # Salvar modelo pickle
    trainer.save_model()
    
    # Exportar para ONNX
    exporter = ModelExporter()
    
    # Amostra para ONNX
    X_sample = X_test_processed[:10]
    
    onnx_path = exporter.export_to_onnx(
        trainer.best_model,
        X_sample
    )
    
    # Verificar exportação ONNX
    if onnx_path:
        onnx_session = exporter.load_onnx_model(onnx_path)
        exporter.verify_onnx_export(
            trainer.best_model,
            onnx_session,
            X_test_processed[:100]
        )
    
    # Salvar feature names
    feature_names_path = MODELS_DIR / "feature_names.pkl"
    joblib.dump(preprocessor.feature_names, feature_names_path)
    print(f"Feature names salvas em: {feature_names_path}")
    
    # 8. RESUMO FINAL
    print("\n" + "="*80)
    print("RESUMO DO TREINAMENTO")
    print("="*80)
    print(f"Melhor Modelo: {trainer.best_model_name}")
    print(f"R^2 no conjunto de teste: {results[trainer.best_model_name]['test_r2']:.4f}")
    print(f"RMSE no conjunto de teste: ${results[trainer.best_model_name]['test_rmse']:,.2f}")
    print(f"MAE no conjunto de teste: ${results[trainer.best_model_name]['test_mae']:,.2f}")
    print(f"\nCross-Validation R^2 (mean +-std): {results[trainer.best_model_name]['cv_r2_mean']:.4f} ± {results[trainer.best_model_name]['cv_r2_std']:.4f}")
    
    print("\n" + "="*80)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*80)
    print(f"\nArquivos gerados em: {MODELS_DIR}")
    print("- best_model.pkl")
    print("- best_model.onnx (se compatível)")
    print("- preprocessor.pkl")
    print("- feature_names.pkl")
    print("- training_results.json")

if __name__ == "__main__":
    main()
