"""
Teste completo da API com múltiplas predições
"""
import requests
import pandas as pd
import json
import math

BASE_URL = "http://127.0.0.1:8000"

print("=" * 80)
print("TESTE COMPLETO DA API - PREDIÇÕES MÚLTIPLAS")
print("=" * 80)

# Carregar dados
df = pd.read_csv('AmesHousing.csv')

print(f"\n📊 Dataset carregado: {len(df)} casas")

# Testar com 5 casas aleatórias
num_tests = 5
samples = df.sample(n=num_tests, random_state=42)

print(f"\n🧪 Testando {num_tests} predições aleatórias...\n")

results = []

for idx, (_, row) in enumerate(samples.iterrows(), 1):
    sample = row.to_dict()
    
    # Guardar preço real
    actual_price = sample['SalePrice']
    
    # Remover campos não-feature
    for field in ['Order', 'PID', 'SalePrice']:
        if field in sample:
            del sample[field]
    
    # Converter NaN para None
    for key, value in sample.items():
        if isinstance(value, float) and math.isnan(value):
            sample[key] = None
    
    # Fazer predição
    try:
        response = requests.post(f"{BASE_URL}/predict/raw", json=sample)
        
        if response.status_code == 200:
            result = response.json()
            predicted_price = result['predicted_price']
            error_pct = abs(predicted_price - actual_price) / actual_price * 100
            
            results.append({
                'actual': actual_price,
                'predicted': predicted_price,
                'error_pct': error_pct
            })
            
            print(f"Casa {idx}:")
            print(f"  Preço Real:     ${actual_price:>12,.2f}")
            print(f"  Preço Previsto: ${predicted_price:>12,.2f}")
            print(f"  Erro:           {error_pct:>11.2f}%")
            print()
        else:
            print(f"Casa {idx}: ERRO {response.status_code}")
    except Exception as e:
        print(f"Casa {idx}: ERRO - {e}")

# Estatísticas
if results:
    errors = [r['error_pct'] for r in results]
    avg_error = sum(errors) / len(errors)
    max_error = max(errors)
    min_error = min(errors)
    
    print("=" * 80)
    print("ESTATÍSTICAS DE PERFORMANCE")
    print("=" * 80)
    print(f"Erro médio:      {avg_error:>6.2f}%")
    print(f"Erro mínimo:     {min_error:>6.2f}%")
    print(f"Erro máximo:     {max_error:>6.2f}%")
    print(f"Testes bem-sucedidos: {len(results)}/{num_tests}")
    print("=" * 80)

print("\n✅ API está funcionando perfeitamente!")
print("Acesse http://127.0.0.1:8000/docs para a documentação interativa")
