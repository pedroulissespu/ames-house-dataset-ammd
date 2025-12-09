"""
Testes abrangentes da API
Testa todos os endpoints, tratamento de erros, performance
"""
import requests
import pandas as pd
import json
import math
import time
import sys
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def test_api_available():
    """Verifica se a API está disponível"""
    print("\n[TEST] Verificando disponibilidade da API...")
    try:
        response = requests.get(BASE_URL, timeout=5)
        assert response.status_code == 200, f"API retornou status {response.status_code}"
        print("[OK] API está disponível")
        return True
    except requests.ConnectionError:
        print("[ERRO] API não está rodando!")
        print("  Execute: uvicorn api.main:app --host 127.0.0.1 --port 8000")
        return False

def test_root_endpoint():
    """Testa endpoint raiz"""
    print("\n[TEST] Testando GET /...")
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data
    print("[OK] Endpoint raiz OK")

def test_health_endpoint():
    """Testa health check"""
    print("\n[TEST] Testando GET /health...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"]["pickle"] == True
    assert data["models_loaded"]["preprocessor"] == True
    print("[OK] Health check OK")

def test_models_info_endpoint():
    """Testa endpoint de informações dos modelos"""
    print("\n[TEST] Testando GET /models/info...")
    response = requests.get(f"{BASE_URL}/models/info")
    assert response.status_code == 200
    data = response.json()
    assert data["pickle_model"]["loaded"] == True
    assert data["num_features"] > 0
    print(f"[OK] Models info OK (Features: {data['num_features']})")

def test_predict_raw_valid():
    """Testa predição com dados válidos"""
    print("\n[TEST] Testando POST /predict/raw (dados válidos)...")
    
    # Carregar dados reais
    df = pd.read_csv('AmesHousing.csv')
    sample = df.iloc[0].to_dict()
    
    # Remover campos não-feature
    for field in ['Order', 'PID', 'SalePrice']:
        if field in sample:
            del sample[field]
    
    # Converter NaN para None
    for key, value in sample.items():
        if isinstance(value, float) and math.isnan(value):
            sample[key] = None
    
    response = requests.post(f"{BASE_URL}/predict/raw", json=sample)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0
    print(f"[OK] Predição válida OK (Preço: ${data['predicted_price']:,.2f})")

def test_predict_raw_invalid_data():
    """Testa predição com dados inválidos"""
    print("\n[TEST] Testando POST /predict/raw (dados inválidos)...")
    
    # Enviar dados vazios
    response = requests.post(f"{BASE_URL}/predict/raw", json={})
    assert response.status_code in [400, 422, 500]
    print("[OK] Rejeição de dados inválidos OK")

def test_predict_raw_missing_fields():
    """Testa predição com campos faltando"""
    print("\n[TEST] Testando POST /predict/raw (campos faltando)...")
    
    incomplete_data = {
        "Gr Liv Area": 1500,
        "Overall Qual": 7
    }
    
    response = requests.post(f"{BASE_URL}/predict/raw", json=incomplete_data)
    # API deve retornar erro
    assert response.status_code in [400, 422, 500]
    print("[OK] Validação de campos faltando OK")

def test_predict_performance():
    """Testa performance de predições"""
    print("\n[TEST] Testando performance de predições...")
    
    df = pd.read_csv('AmesHousing.csv')
    sample = df.iloc[0].to_dict()
    
    for field in ['Order', 'PID', 'SalePrice']:
        if field in sample:
            del sample[field]
    
    for key, value in sample.items():
        if isinstance(value, float) and math.isnan(value):
            sample[key] = None
    
    # Fazer 10 predições e medir tempo
    times = []
    for _ in range(10):
        start = time.time()
        response = requests.post(f"{BASE_URL}/predict/raw", json=sample)
        end = time.time()
        assert response.status_code == 200
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)
    
    print(f"[OK] Performance OK")
    print(f"  Tempo médio: {avg_time*1000:.2f}ms")
    print(f"  Tempo mínimo: {min_time*1000:.2f}ms")
    print(f"  Tempo máximo: {max_time*1000:.2f}ms")
    
    assert avg_time < 1.0, "Tempo médio muito alto (>1s)"

def test_predict_accuracy():
    """Testa acurácia das predições"""
    print("\n[TEST] Testando acurácia das predições...")
    
    df = pd.read_csv('AmesHousing.csv')
    samples = df.sample(n=20, random_state=42)
    
    errors = []
    
    for _, row in samples.iterrows():
        actual_price = row['SalePrice']
        sample = row.to_dict()
        
        for field in ['Order', 'PID', 'SalePrice']:
            if field in sample:
                del sample[field]
        
        for key, value in sample.items():
            if isinstance(value, float) and math.isnan(value):
                sample[key] = None
        
        response = requests.post(f"{BASE_URL}/predict/raw", json=sample)
        if response.status_code == 200:
            predicted_price = response.json()['predicted_price']
            error_pct = abs(predicted_price - actual_price) / actual_price * 100
            errors.append(error_pct)
    
    avg_error = sum(errors) / len(errors)
    max_error = max(errors)
    min_error = min(errors)
    
    print(f"[OK] Acurácia testada em {len(errors)} amostras")
    print(f"  Erro médio: {avg_error:.2f}%")
    print(f"  Erro mínimo: {min_error:.2f}%")
    print(f"  Erro máximo: {max_error:.2f}%")
    
    assert avg_error < 15.0, f"Erro médio muito alto: {avg_error}%"

def test_concurrent_requests():
    """Testa requisições concorrentes"""
    print("\n[TEST] Testando requisições concorrentes...")
    
    df = pd.read_csv('AmesHousing.csv')
    sample = df.iloc[0].to_dict()
    
    for field in ['Order', 'PID', 'SalePrice']:
        if field in sample:
            del sample[field]
    
    for key, value in sample.items():
        if isinstance(value, float) and math.isnan(value):
            sample[key] = None
    
    # Fazer múltiplas requisições em paralelo
    import concurrent.futures
    
    def make_request():
        response = requests.post(f"{BASE_URL}/predict/raw", json=sample)
        return response.status_code == 200
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    success_rate = sum(results) / len(results) * 100
    print(f"[OK] Requisições concorrentes OK (Taxa de sucesso: {success_rate:.0f}%)")
    assert success_rate >= 90, "Taxa de sucesso muito baixa"

def run_all_tests():
    """Executa todos os testes"""
    print("=" * 80)
    print("TESTES ABRANGENTES DA API")
    print("=" * 80)
    
    # Verificar se API está disponível
    if not test_api_available():
        print("\n[ERRO] TESTES INTERROMPIDOS: API não está rodando")
        return False
    
    tests = [
        test_root_endpoint,
        test_health_endpoint,
        test_models_info_endpoint,
        test_predict_raw_valid,
        test_predict_raw_invalid_data,
        test_predict_raw_missing_fields,
        test_predict_performance,
        test_predict_accuracy,
        test_concurrent_requests,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[ERRO] FALHOU: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERRO] ERRO: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTADO: {passed} passaram, {failed} falharam")
    print("=" * 80)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
