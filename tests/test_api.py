"""
Testes para a API FastAPI
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Adicionar path
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


def test_root():
    """Testa endpoint raiz"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_check():
    """Testa health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_models_info():
    """Testa informações dos modelos"""
    response = client.get("/models/info")
    assert response.status_code == 200
    data = response.json()
    assert "pickle_model" in data


def test_predict_pkl():
    """Testa predição com modelo pickle"""
    payload = {
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
    
    response = client.post("/predict/pkl", json=payload)
    
    # Se o modelo estiver carregado
    if response.status_code == 200:
        data = response.json()
        assert "predicted_price" in data
        assert "model_used" in data
        assert data["model_used"] == "pickle"
    # Se o modelo não estiver carregado
    elif response.status_code == 503:
        assert True  # Esperado quando modelo não está carregado
    else:
        pytest.fail(f"Unexpected status code: {response.status_code}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
