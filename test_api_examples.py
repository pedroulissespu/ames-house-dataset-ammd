"""
Exemplos de uso da API de Predição de Preços
"""
import requests
import json


# URL base da API (ajuste se necessário)
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Verifica se a API está funcionando"""
    print("=" * 70)
    print("1. TESTE DE HEALTH CHECK")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")


def test_models_info():
    """Obtém informações sobre os modelos carregados"""
    print("=" * 70)
    print("2. INFORMAÇÕES DOS MODELOS")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/models/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}\n")


def test_single_prediction():
    """Testa predição individual"""
    print("=" * 70)
    print("3. PREDIÇÃO INDIVIDUAL")
    print("=" * 70)
    
    # Dados de exemplo de uma casa
    house_data = {
        "Gr_Liv_Area": 1500,        # Área de estar: 1500 pés quadrados
        "Overall_Qual": 7,           # Qualidade: 7/10
        "Overall_Cond": 5,           # Condição: 5/10
        "Year_Built": 2000,          # Construída em 2000
        "Year_Remod_Add": 2000,      # Sem remodelação
        "Total_Bsmt_SF": 1000,       # Porão: 1000 pés quadrados
        "Full_Bath": 2,              # 2 banheiros completos
        "Half_Bath": 1,              # 1 lavabo
        "Bedroom_AbvGr": 3,          # 3 quartos
        "Kitchen_AbvGr": 1,          # 1 cozinha
        "TotRms_AbvGrd": 7,          # 7 cômodos totais
        "Fireplaces": 1,             # 1 lareira
        "Garage_Cars": 2,            # Garagem para 2 carros
        "Garage_Area": 500           # Área da garagem: 500 pés quadrados
    }
    
    print("Dados da casa:")
    print(json.dumps(house_data, indent=2))
    print()
    
    # Fazer predição com modelo pickle
    response = requests.post(f"{BASE_URL}/predict/pkl", json=house_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPredição com Pickle:")
        print(f"  Preço previsto: ${result['predicted_price']:,.2f}")
        print(f"  Modelo usado: {result['model_used']}")
        print(f"  Mensagem: {result['message']}")
    else:
        print(f"Erro: {response.text}")
    
    print()


def test_multiple_predictions():
    """Testa predição de múltiplas casas"""
    print("=" * 70)
    print("4. PREDIÇÃO EM LOTE (3 CASAS)")
    print("=" * 70)
    
    houses = [
        {
            "Gr_Liv_Area": 1200,
            "Overall_Qual": 5,
            "Overall_Cond": 6,
            "Year_Built": 1980,
            "Year_Remod_Add": 1995,
            "Total_Bsmt_SF": 800,
            "Full_Bath": 1,
            "Half_Bath": 1,
            "Bedroom_AbvGr": 2,
            "Kitchen_AbvGr": 1,
            "TotRms_AbvGrd": 5,
            "Fireplaces": 0,
            "Garage_Cars": 1,
            "Garage_Area": 300
        },
        {
            "Gr_Liv_Area": 2000,
            "Overall_Qual": 8,
            "Overall_Cond": 7,
            "Year_Built": 2005,
            "Year_Remod_Add": 2005,
            "Total_Bsmt_SF": 1500,
            "Full_Bath": 3,
            "Half_Bath": 1,
            "Bedroom_AbvGr": 4,
            "Kitchen_AbvGr": 1,
            "TotRms_AbvGrd": 9,
            "Fireplaces": 2,
            "Garage_Cars": 3,
            "Garage_Area": 700
        },
        {
            "Gr_Liv_Area": 1800,
            "Overall_Qual": 7,
            "Overall_Cond": 6,
            "Year_Built": 1998,
            "Year_Remod_Add": 2010,
            "Total_Bsmt_SF": 1200,
            "Full_Bath": 2,
            "Half_Bath": 1,
            "Bedroom_AbvGr": 3,
            "Kitchen_AbvGr": 1,
            "TotRms_AbvGrd": 8,
            "Fireplaces": 1,
            "Garage_Cars": 2,
            "Garage_Area": 600
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=houses)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        results = response.json()
        print(f"\nPredições para {len(results)} casas:")
        for i, result in enumerate(results, 1):
            print(f"\n  Casa {i}:")
            print(f"    Preço previsto: ${result['predicted_price']:,.2f}")
    else:
        print(f"Erro: {response.text}")
    
    print()


def test_onnx_prediction():
    """Testa predição com modelo ONNX"""
    print("=" * 70)
    print("5. PREDIÇÃO COM ONNX (MAIS RÁPIDO)")
    print("=" * 70)
    
    house_data = {
        "Gr_Liv_Area": 1600,
        "Overall_Qual": 8,
        "Overall_Cond": 6,
        "Year_Built": 2010,
        "Year_Remod_Add": 2010,
        "Total_Bsmt_SF": 1100,
        "Full_Bath": 2,
        "Half_Bath": 1,
        "Bedroom_AbvGr": 3,
        "Kitchen_AbvGr": 1,
        "TotRms_AbvGrd": 7,
        "Fireplaces": 1,
        "Garage_Cars": 2,
        "Garage_Area": 550
    }
    
    # Predição com ONNX
    response = requests.post(f"{BASE_URL}/predict/onnx", json=house_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPredição com ONNX:")
        print(f"  Preço previsto: ${result['predicted_price']:,.2f}")
        print(f"  Modelo usado: {result['model_used']}")
    else:
        print(f"Erro: {response.text}")
    
    print()


def compare_models():
    """Compara predições entre pickle e ONNX"""
    print("=" * 70)
    print("6. COMPARAÇÃO PICKLE vs ONNX")
    print("=" * 70)
    
    house_data = {
        "Gr_Liv_Area": 1750,
        "Overall_Qual": 7,
        "Overall_Cond": 5,
        "Year_Built": 2003,
        "Year_Remod_Add": 2015,
        "Total_Bsmt_SF": 1200,
        "Full_Bath": 2,
        "Half_Bath": 1,
        "Bedroom_AbvGr": 3,
        "Kitchen_AbvGr": 1,
        "TotRms_AbvGrd": 8,
        "Fireplaces": 1,
        "Garage_Cars": 2,
        "Garage_Area": 520
    }
    
    # Predição com pickle
    response_pkl = requests.post(f"{BASE_URL}/predict/pkl", json=house_data)
    
    # Predição com ONNX
    response_onnx = requests.post(f"{BASE_URL}/predict/onnx", json=house_data)
    
    if response_pkl.status_code == 200 and response_onnx.status_code == 200:
        price_pkl = response_pkl.json()['predicted_price']
        price_onnx = response_onnx.json()['predicted_price']
        
        print(f"Preço previsto (Pickle): ${price_pkl:,.2f}")
        print(f"Preço previsto (ONNX):   ${price_onnx:,.2f}")
        print(f"Diferença absoluta:      ${abs(price_pkl - price_onnx):,.2f}")
        print(f"Diferença percentual:    {abs(price_pkl - price_onnx) / price_pkl * 100:.6f}%")
    else:
        print("Erro ao fazer comparação")
    
    print()


def main():
    """Executa todos os testes"""
    print("\n" + "=" * 70)
    print(" TESTES DA API DE PREDIÇÃO DE PREÇOS DE IMÓVEIS")
    print("=" * 70)
    print()
    
    try:
        # 1. Health check
        test_health_check()
        
        # 2. Informações dos modelos
        test_models_info()
        
        # 3. Predição individual
        test_single_prediction()
        
        # 4. Predição em lote
        test_multiple_predictions()
        
        # 5. Predição com ONNX
        test_onnx_prediction()
        
        # 6. Comparação de modelos
        compare_models()
        
        print("=" * 70)
        print("TODOS OS TESTES CONCLUÍDOS COM SUCESSO! ✓")
        print("=" * 70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERRO: Não foi possível conectar à API")
        print("Certifique-se de que a API está rodando em http://localhost:8000")
        print("\nPara iniciar a API, execute:")
        print("  cd api")
        print("  uvicorn main:app --reload")
    
    except Exception as e:
        print(f"\n❌ ERRO: {e}")


if __name__ == "__main__":
    main()
