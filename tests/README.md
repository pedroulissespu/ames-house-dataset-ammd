# TESTS - Testes Automatizados

Este diretório contém todos os testes automatizados do projeto.

## Arquivos de Teste

### `test_api_health.py`
Testes básicos de saúde da API.

**O que testa:**
- Health check endpoint
- Informações dos modelos
- Predição básica com dados reais do CSV

**Como executar:**
```bash
python tests/test_api_health.py
```

**Pré-requisito:** API deve estar rodando
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### `test_api_predictions.py`
Testes de predições múltiplas.

**O que testa:**
- Predições de 5 casas aleatórias
- Cálculo de estatísticas de erro
- Performance geral do modelo

**Como executar:**
```bash
python tests/test_api_predictions.py
```

**Pré-requisito:** API deve estar rodando

### `test_api_comprehensive.py` TESTE COMPLETO
Testes da API completo.

**O que testa:**
- [OK] Disponibilidade da API
- [OK] Todos os endpoints (GET /, /health, /models/info)
- [OK] Predições com dados válidos
- [OK] Validação de dados inválidos
- [OK] Tratamento de campos faltando
- [OK] Performance (tempo de resposta)
- [OK] Acurácia (erro percentual)
- [OK] Requisições concorrentes

**Como executar:**
```bash
python tests/test_api_comprehensive.py
```

**Saída esperada:**
```
================================================================================
TESTES ABRANGENTES DA API
================================================================================
[TEST] Verificando disponibilidade da API...
[OK] API está disponível
[TEST] Testando GET /...
[OK] Endpoint raiz OK
...
================================================================================
RESULTADO: 9 passaram, 0 falharam
================================================================================
```

### `test_model_performance.py`
Testes do modelo treinado.

**O que testa:**
- [OK] Existência do modelo e preprocessador
- [OK] Predições básicas
- [OK] Métricas (R², RMSE, MAE)
- [OK] Consistência (determinismo)

**Como executar:**
```bash
python tests/test_model_performance.py
```

**Não requer API rodando** - testa o modelo diretamente.

**Saída esperada:**
```
================================================================================
TESTES DE PERFORMANCE DO MODELO
================================================================================
[TEST] Verificando existência do modelo...
[OK] Modelo encontrado
[TEST] Testando predição do modelo...
[OK] 10 predições realizadas com sucesso
...
================================================================================
[OK] TODOS OS TESTES PASSARAM!
================================================================================
```

### `test_api.py`
Testes da API usando pytest.

**Como executar com pytest:**
```bash
pytest tests/test_api.py -v
```

## Executando Todos os Testes

### Opção 1: Manualmente
```bash
# 1. Testar modelo (sem API)
python tests/test_model_performance.py

# 2. Iniciar API em outra janela
uvicorn api.main:app --host 127.0.0.1 --port 8000

# 3. Em outra janela, testar API
python tests/test_api_comprehensive.py
python tests/test_api_health.py
python tests/test_api_predictions.py
```

### Opção 2: Com pytest (se instalado)
```bash
pytest tests/ -v
```

## Estrutura de um Teste

Todos os testes seguem o padrão:

```python
def test_nome_do_teste():
    """Descrição do que o teste faz"""
    print("\n[TEST] Testando...")
    
    # Arrange (preparar)
    dados = preparar_dados()
    
    # Act (executar)
    resultado = funcao_testada(dados)
    
    # Assert (validar)
    assert resultado == esperado, "Mensagem de erro"
    
    print("[OK] Teste passou")
```

## Cobertura de Testes

### API
- [OK] Health check
- [OK] Endpoints de informação
- [OK] Predições válidas
- [OK] Validação de erros
- [OK] Performance
- [OK] Acurácia
- [OK] Concorrência

### Modelo
- [OK] Existência de arquivos
- [OK] Predições básicas
- [OK] Métricas de performance
- [OK] Consistência/Determinismo

## Interpretação dos Resultados

### Testes da API

**Sucesso:**
```
[OK] API está disponível
[OK] Health check OK
[OK] Predição válida OK (Preço: $214,150.38)
```

**Falha:**
```
[ERRO] FALHOU: assertion error message
[ERRO] ERRO: detailed error traceback
```

### Métricas Esperadas

**Modelo:**
- R² > 0.90 (ótimo)
- RMSE < $20,000 (bom)
- MAE < $15,000 (bom)

**API:**
- Erro médio < 5% (ótimo)
- Tempo médio < 100ms (excelente)
- Taxa de sucesso > 95% (esperado)
