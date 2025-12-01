#!/bin/bash

# Script de execução do projeto

echo "================================================"
echo "  Ames Housing Price Prediction"
echo "================================================"
echo ""

# Verificar se o ambiente virtual está ativo
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Ambiente virtual não está ativo!"
    echo "Execute: source venv/bin/activate"
    exit 1
fi

# Menu de opções
echo "Escolha uma opção:"
echo ""
echo "1) Treinar modelos"
echo "2) Executar API (desenvolvimento)"
echo "3) Executar API (produção)"
echo "4) Executar testes"
echo "5) Abrir Jupyter Notebook"
echo "6) Instalar dependências"
echo ""

read -p "Opção [1-6]: " option

case $option in
    1)
        echo ""
        echo "🚀 Iniciando treinamento dos modelos..."
        python train.py
        ;;
    2)
        echo ""
        echo "🌐 Iniciando API em modo desenvolvimento..."
        cd api && uvicorn main:app --reload --host 0.0.0.0 --port 8000
        ;;
    3)
        echo ""
        echo "🌐 Iniciando API em modo produção..."
        cd api && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
        ;;
    4)
        echo ""
        echo "🧪 Executando testes..."
        pytest tests/ -v
        ;;
    5)
        echo ""
        echo "📓 Abrindo Jupyter Notebook..."
        jupyter notebook notebooks/
        ;;
    6)
        echo ""
        echo "📦 Instalando dependências..."
        pip install --upgrade pip
        pip install -r requirements.txt
        echo "✓ Dependências instaladas!"
        ;;
    *)
        echo "Opção inválida!"
        exit 1
        ;;
esac
