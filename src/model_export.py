"""
Módulo para exportação de modelos em diferentes formatos
"""
import joblib
import numpy as np

# Importações ONNX opcionais
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnxruntime as rt
    ONNX_AVAILABLE = True
except (ImportError, AttributeError, Exception) as e:
    ONNX_AVAILABLE = False
    print(f"AVISO: ONNX não disponível: {type(e).__name__}")
    print("Exportação ONNX será desabilitada (não afeta o treinamento)")

from src.config import MODEL_PKL_PATH, MODEL_ONNX_PATH


class ModelExporter:
    """Classe para exportar modelos em diferentes formatos"""
    
    @staticmethod
    def export_to_pickle(model, filepath: str = None):
        """
        Exporta modelo para formato pickle (.pkl)
        """
        if filepath is None:
            filepath = MODEL_PKL_PATH
        
        joblib.dump(model, filepath)
        print(f"Modelo exportado para pickle: {filepath}")
        
        return filepath
    
    @staticmethod
    def export_to_onnx(model, X_sample, filepath: str = None):
        """
        Exporta modelo para formato ONNX (.onnx)
        
        Args:
            model: Modelo scikit-learn treinado
            X_sample: Amostra de dados de entrada para inferir shape
            filepath: Caminho para salvar o arquivo
        """
        if not ONNX_AVAILABLE:
            print("ERRO: ONNX não está disponível. Pulando exportação ONNX.")
            return None
            
        if filepath is None:
            filepath = MODEL_ONNX_PATH
        
        # Definir tipo de entrada
        n_features = X_sample.shape[1]
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        
        # Converter para ONNX
        try:
            onnx_model = convert_sklearn(
                model,
                initial_types=initial_type,
                target_opset=12
            )
            
            # Salvar
            with open(filepath, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"Modelo exportado para ONNX: {filepath}")
            print(f"Número de features: {n_features}")
            
            return filepath
            
        except Exception as e:
            print(f"ERRO: Falha ao exportar para ONNX: {e}")
            print("Alguns modelos podem não ser compatíveis com ONNX.")
            return None
    
    @staticmethod
    def load_onnx_model(filepath: str = None):
        """
        Carrega modelo ONNX
        """
        if not ONNX_AVAILABLE:
            print("ERRO: ONNX não está disponível.")
            return None
            
        if filepath is None:
            filepath = MODEL_ONNX_PATH
        
        sess = rt.InferenceSession(filepath)
        print(f"Modelo ONNX carregado com sucesso: {filepath}")
        
        return sess
    
    @staticmethod
    def predict_onnx(sess, X):
        """
        Faz predição usando modelo ONNX
        
        Args:
            sess: Sessão ONNX Runtime
            X: Dados de entrada (numpy array)
        """
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        
        # Garantir que X é float32
        X = X.astype(np.float32)
        
        pred_onnx = sess.run([label_name], {input_name: X})[0]
        
        return pred_onnx
    
    @staticmethod
    def verify_onnx_export(sklearn_model, onnx_session, X_test, tolerance=1e-4):
        """
        Verifica se o modelo ONNX produz os mesmos resultados que o scikit-learn
        
        Args:
            sklearn_model: Modelo scikit-learn original
            onnx_session: Sessão ONNX Runtime
            X_test: Dados de teste
            tolerance: Tolerância para diferenças
        """
        # Predição com scikit-learn
        y_sklearn = sklearn_model.predict(X_test)
        
        # Predição com ONNX
        y_onnx = ModelExporter.predict_onnx(onnx_session, X_test).flatten()
        
        # Calcular diferença
        diff = np.abs(y_sklearn - y_onnx)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print("\nVerificação de exportação ONNX:")
        print(f"Diferença máxima: {max_diff:.6f}")
        print(f"Diferença média: {mean_diff:.6f}")
        
        if max_diff < tolerance:
            print("Exportação ONNX verificada com sucesso!")
            return True
        else:
            print("Diferenças significativas detectadas")
            return False


def export_full_pipeline(model, preprocessor, feature_names, base_path: str):
    """
    Exporta modelo completo com preprocessador
    
    Args:
        model: Modelo treinado
        preprocessor: Pipeline de pré-processamento
        feature_names: Lista de nomes das features
        base_path: Diretório base para salvar
    """
    from pathlib import Path
    
    base_path = Path(base_path)
    
    # Salvar modelo
    model_path = base_path / "best_model.pkl"
    joblib.dump(model, model_path)
    print(f"Modelo salvo: {model_path}")
    
    # Salvar preprocessador
    preprocessor_path = base_path / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessador salvo: {preprocessor_path}")
    
    # Salvar feature names
    features_path = base_path / "feature_names.pkl"
    joblib.dump(feature_names, features_path)
    print(f"Feature names salvas no arquivo: {features_path}")
    
    print("\nPipeline completo exportado com sucesso!")
