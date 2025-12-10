"""
API FastAPI para servir os modelos de predição de preço de imóveis
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from pathlib import Path

try:
    import onnxruntime as rt
    ONNX_AVAILABLE = True
except (ImportError, AttributeError, Exception) as e:
    ONNX_AVAILABLE = False
    print(f"ONNX não disponível: {type(e).__name__}")
    print(",Endpoints ONNX serão desabilitados.")

# Importar configurações
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.config import MODEL_PKL_PATH, MODEL_ONNX_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH
from src.feature_engineering import FeatureEngineer  # Importar feature engineering

app = FastAPI(
    title="Ames Housing Price Prediction API",
    description="API para predição de preços de imóveis usando ML",
    version="1.0.0"
)

# Carregar modelos e preprocessador no início
model_pkl = None
model_onnx = None
preprocessor = None
feature_names = None


@app.on_event("startup")
async def load_models():
    """Carrega os modelos na inicialização"""
    global model_pkl, model_onnx, preprocessor, feature_names
    
    try:
        # Carregar modelo pickle
        if MODEL_PKL_PATH.exists():
            model_pkl = joblib.load(MODEL_PKL_PATH)
            print(f"Modelo PKL carregado de {MODEL_PKL_PATH}")
        
        # Carregar modelo ONNX (apenas se disponível)
        if ONNX_AVAILABLE and MODEL_ONNX_PATH.exists():
            model_onnx = rt.InferenceSession(str(MODEL_ONNX_PATH))
            print(f"Modelo ONNX carregado de {MODEL_ONNX_PATH}")
        elif not ONNX_AVAILABLE:
            print("ONNX não disponível - endpoints ONNX desabilitados")
        
        # Carregar preprocessador
        if PREPROCESSOR_PATH.exists():
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print(f"Preprocessador carregado de {PREPROCESSOR_PATH}")
        
        # Carregar feature names
        if FEATURE_NAMES_PATH.exists():
            feature_names = joblib.load(FEATURE_NAMES_PATH)
            print(f"Feature names carregadas")
        
        if not any([model_pkl, model_onnx]):
            print("Nenhum modelo foi carregado! Execute train.py primeiro.")
        
    except Exception as e:
        print(f"Erro ao carregar modelos: {e}")


class HouseFeatures(BaseModel):
    """Schema de entrada pra API
    
    """
    # features principais (comecei por essas)
    Gr_Liv_Area: int = Field(..., description="Área de estar acima do solo (pés quadrados)")
    Overall_Qual: int = Field(..., ge=1, le=10, description="Qualidade geral do material e acabamento")
    Overall_Cond: int = Field(..., ge=1, le=10, description="Condição geral")
    Year_Built: int = Field(..., ge=1800, le=2025, description="Ano de construção")
    Year_Remod_Add: int = Field(..., description="Ano de remodelação")
    Total_Bsmt_SF: int = Field(..., description="Área total do porão (pés quadrados)")
    Full_Bath: int = Field(..., description="Número de banheiros completos")
    Half_Bath: int = Field(..., description="Número de lavabos")
    Bedroom_AbvGr: int = Field(..., description="Número de quartos acima do térreo")
    Kitchen_AbvGr: int = Field(..., description="Número de cozinhas")
    TotRms_AbvGrd: int = Field(..., description="Total de cômodos acima do térreo")
    Fireplaces: int = Field(..., description="Número de lareiras")
    Garage_Cars: int = Field(..., description="Capacidade da garagem em carros")
    Garage_Area: int = Field(..., description="Área da garagem (pés quadrados)")
    
    class Config:
        schema_extra = {
            "example": {
                "Gr_Liv_Area": 1500,
                "Overall_Qual": 7,
                "Overall_Cond": 5,
                "Year_Built": 2000,
                "Year_Remod_Add": 2000,
                "Total_Bsmt_SF": 1000,
                "Full_Bath": 2,
                "Half_Bath": 1,
                "Bedroom_AbvGr": 3,
                "Kitchen_AbvGr": 1,
                "TotRms_AbvGrd": 7,
                "Fireplaces": 1,
                "Garage_Cars": 2,
                "Garage_Area": 500
            }
        }


class PredictionResponse(BaseModel):
    """Schema de resposta da predição"""
    predicted_price: float
    model_used: str
    message: str


@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "Ames Housing Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_pkl": "/predict/pkl",
            "predict_onnx": "/predict/onnx",
            "models_info": "/models/info"
        }
    }


@app.get("/health")
async def health_check():
    """Verifica o status da API e modelos"""
    return {
        "status": "healthy",
        "models_loaded": {
            "pickle": model_pkl is not None,
            "onnx": model_onnx is not None,
            "preprocessor": preprocessor is not None
        }
    }


@app.get("/models/info")
async def models_info():
    """Retorna informações sobre os modelos"""
    info = {
        "pickle_model": {
            "loaded": model_pkl is not None,
            "type": str(type(model_pkl).__name__) if model_pkl else None
        },
        "onnx_model": {
            "loaded": model_onnx is not None,
        },
        "preprocessor": {
            "loaded": preprocessor is not None,
        }
    }
    
    if feature_names:
        info["num_features"] = len(feature_names)
    
    return info


@app.post("/predict/pkl", response_model=PredictionResponse)
async def predict_pkl(features: HouseFeatures):
    """
    Faz predição usando o modelo pickle
    """
    if model_pkl is None:
        raise HTTPException(status_code=503, detail="Modelo pickle não está carregado")
    
    try:
        # Converter para DataFrame
        features_dict = features.dict()
        df = pd.DataFrame([features_dict])
        
        # Pré-processar se necessário
        if preprocessor:
            X = preprocessor.transform(df)
        else:
            X = df.values
        
        # Predição
        prediction = model_pkl.predict(X)[0]
        
        return PredictionResponse(
            predicted_price=float(prediction),
            model_used="pickle",
            message="Predição realizada com sucesso"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@app.post("/predict/onnx", response_model=PredictionResponse)
async def predict_onnx(features: HouseFeatures):
    """
    Faz predição usando o modelo ONNX (se disponível)
    """
    if not ONNX_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="ONNX não está disponível. Use o endpoint /predict/pkl"
        )
    
    if model_onnx is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo ONNX não está carregado. Execute train.py primeiro."
        )
    
    try:
        # Converter para DataFrame
        features_dict = features.dict()
        df = pd.DataFrame([features_dict])
        
        # Pré-processar se necessário
        if preprocessor:
            X = preprocessor.transform(df)
        else:
            X = df.values
        
        # Garantir float32
        X = X.astype(np.float32)
        
        # Predição ONNX
        input_name = model_onnx.get_inputs()[0].name
        label_name = model_onnx.get_outputs()[0].name
        prediction = model_onnx.run([label_name], {input_name: X})[0][0]
        
        return PredictionResponse(
            predicted_price=float(prediction),
            model_used="onnx",
            message="Predição realizada com sucesso usando ONNX"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(houses: List[HouseFeatures]):
    """
    Faz predição em lote usando o modelo pickle
    """
    if model_pkl is None:
        raise HTTPException(status_code=503, detail="Modelo pickle não está carregado")
    
    try:
        # Converter para DataFrame
        features_list = [house.dict() for house in houses]
        df = pd.DataFrame(features_list)
        
        # Pré-processar se necessário
        if preprocessor:
            X = preprocessor.transform(df)
        else:
            X = df.values
        
        # Predição
        predictions = model_pkl.predict(X)
        
        # Criar respostas
        responses = [
            PredictionResponse(
                predicted_price=float(pred),
                model_used="pickle",
                message="Predição realizada com sucesso"
            )
            for pred in predictions
        ]
        
        return responses
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição em lote: {str(e)}")


@app.post("/predict/raw", response_model=PredictionResponse)
async def predict_raw(data: Dict):
    """
    Faz predição usando dados brutos (formato flexível - aceita qualquer estrutura do CSV)
    Use este endpoint para enviar dados diretamente do dataset.
    """
    if model_pkl is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado. Execute train.py primeiro.")
    
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessador não carregado")
    
    try:
        # Converter para DataFrame
        df = pd.DataFrame([data])
        
        # APLICAR FEATURE ENGINEERING (igual ao treinamento)
        fe = FeatureEngineer()
        df = fe.create_features(df)
        df = fe.create_interaction_features(df)  # Adicionar features de interação
        
        # Pré-processar
        X = preprocessor.transform(df)
        
        # Predição
        prediction = model_pkl.predict(X)[0]
        
        return PredictionResponse(
            predicted_price=float(prediction),
            model_used="pickle",
            message="Predição realizada com sucesso (endpoint raw)"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
