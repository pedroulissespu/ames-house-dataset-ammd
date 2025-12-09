"""
Script para testar a API localmente
"""
import requests
import pandas as pd
import json

# URL da API
BASE_URL = "http://127.0.0.1:8000"

print("=" * 80)
print("TESTANDO API - AMES HOUSING PRICE PREDICTION")
print("=" * 80)

# 1. Testar Health Check
print("\n[1/3] Testando Health Check...")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"ERRO: {e}")
    exit(1)

# 2. Testar Info dos Modelos
print("\n[2/3] Testando info dos modelos...")
try:
    response = requests.get(f"{BASE_URL}/models/info")
    print(f"Status: {response.status_code}")
    print(f"Resposta: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"ERRO: {e}")

# 3. Testar Predição com dados reais
print("\n[3/3] Testando predição com dados do CSV...")
try:
    # Carregar uma linha do CSV
    df = pd.read_csv('AmesHousing.csv')
    
    # Pegar primeira casa (sem o SalePrice)
    sample = df.iloc[0].to_dict()
    
    # Remover campos que não são features
    if 'Order' in sample:
        del sample['Order']
    if 'PID' in sample:
        del sample['PID']
    if 'SalePrice' in sample:
        actual_price = sample['SalePrice']
        del sample['SalePrice']
    
    # Converter NaN para None (compatível com JSON)
    import math
    for key, value in sample.items():
        if isinstance(value, float) and math.isnan(value):
            sample[key] = None
    
    print(f"\nPreço real da casa: ${actual_price:,.2f}")
    
    # Fazer predição
    response = requests.post(
        f"{BASE_URL}/predict/raw",
        json=sample,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        predicted_price = result['predicted_price']
        print(f"Preço previsto: ${predicted_price:,.2f}")
        print(f"Diferença: ${abs(predicted_price - actual_price):,.2f}")
        print(f"Erro percentual: {abs(predicted_price - actual_price) / actual_price * 100:.2f}%")
        print(f"\nResposta completa: {json.dumps(result, indent=2)}")
    else:
        print(f"ERRO {response.status_code}: {response.text}")
        
except Exception as e:
    print(f"ERRO: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TESTES CONCLUÍDOS!")
print("=" * 80)
