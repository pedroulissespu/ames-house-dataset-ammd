"""
Script simples para testar a API com dados reais do dataset
"""
import requests
import json
import pandas as pd
from pathlib import Path

# Carregar dados reais
data_path = Path(__file__).parent / "AmesHousing.csv"
df = pd.read_csv(data_path)

# Pegar uma amostra real (primeira casa)
sample = df.iloc[0].to_dict()

# Remover colunas não necessárias
sample.pop('Order', None)
sample.pop('PID', None)
sample.pop('SalePrice', None)

# Converter NaN para None (compatível com JSON)
sample = {k: (None if pd.isna(v) else v) for k, v in sample.items()}

print("=" * 80)
print("TESTE DA API - PREDIÇÃO COM DADOS REAIS")
print("=" * 80)
print(f"\nUsando amostra da linha 0 do dataset")
print(f"Preço real: ${df.iloc[0]['SalePrice']:,.2f}")
print("\n" + "=" * 80)

# Fazer predição
url = "http://localhost:8000/predict/raw"  # Usando endpoint raw que aceita dados do CSV

try:
    response = requests.post(url, json=sample)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ PREDIÇÃO BEM-SUCEDIDA!")
        print(f"\nPreço previsto: ${result['predicted_price']:,.2f}")
        print(f"Preço real:     ${df.iloc[0]['SalePrice']:,.2f}")
        print(f"Diferença:      ${abs(result['predicted_price'] - df.iloc[0]['SalePrice']):,.2f}")
        print(f"Modelo usado:   {result['model_used']}")
    else:
        print(f"❌ ERRO {response.status_code}")
        print(response.json())
        
except requests.exceptions.ConnectionError:
    print("❌ Não foi possível conectar à API")
    print("Certifique-se de que está rodando:")
    print("  cd api && uvicorn main:app --reload")
except Exception as e:
    print(f"❌ Erro: {e}")

print("=" * 80)
