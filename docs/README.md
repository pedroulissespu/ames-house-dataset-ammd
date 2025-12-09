# Documentação

Este diretório contém toda a documentação do projeto.

## Arquivos

### `relatorio_tecnico.md`
Relatório técnico completo do projeto em Markdown.

**Conteúdo:**
- Introdução e motivação
- Descrição do dataset
- Metodologia aplicada
- Análise exploratória de dados
- Feature engineering detalhado
- Resultados dos modelos
- Comparação entre algoritmos
- Conclusões e aprendizados

**Visualizar:**
```bash
# No VS Code
code docs/relatorio_tecnico.md

# Ou converter para HTML
markdown docs/relatorio_tecnico.md > relatorio.html
```

### `relatorio_tecnico.pdf`
Versão em PDF do relatório técnico.

**Como gerar/atualizar:**
Veja `GERAR_PDF.md` para instruções.

### `GERAR_PDF.md`
Instruções para gerar o PDF do relatório técnico.

**Métodos disponíveis:**
1. Pandoc (recomendado)
2. VS Code com extensão Markdown PDF
3. Ferramentas online

**Conteúdo:**
- Resumo dos testes realizados
- Resultados de cada teste
- Estatísticas de performance
- Métricas de acurácia
- Exemplos de uso

**Quando foi gerado:**
Este relatório foi gerado automaticamente após os testes da API.

## Estrutura da Documentação

```
docs/
├── relatorio_tecnico.md    # Relatório principal (Markdown)
├── relatorio_tecnico.pdf   # Relatório principal (PDF)
├── GERAR_PDF.md            # Instruções para gerar PDF
```

## Como Usar

### Ler a Documentação

**Relatório Técnico:**
```bash
# Markdown (editável)
cat docs/relatorio_tecnico.md

# PDF (leitura)
open docs/relatorio_tecnico.pdf  # macOS
start docs/relatorio_tecnico.pdf  # Windows
xdg-open docs/relatorio_tecnico.pdf  # Linux
```

### Atualizar Documentação

1. **Editar Markdown:**
   ```bash
   code docs/relatorio_tecnico.md
   ```

2. **Gerar PDF:**
   Siga as instruções em `GERAR_PDF.md`

3. **Commit das mudanças:**
   ```bash
   git add docs/
   git commit -m "Atualizar documentação"
   ```

## Seções do Relatório Técnico

1. **Resumo Executivo**
   - Visão geral do projeto
   - Principais resultados

2. **Introdução**
   - Contexto e motivação
   - Objetivos
   - Descrição do dataset

3. **Metodologia**
   - Pipeline de desenvolvimento
   - Ferramentas utilizadas
   - Métricas de avaliação

4. **Análise Exploratória**
   - Distribuições
   - Correlações
   - Valores ausentes

5. **Feature Engineering**
   - Features criadas
   - Justificativas
   - Impacto

6. **Pré-processamento**
   - Tratamento de outliers
   - Transformações
   - Pipeline

7. **Modelagem**
   - Modelos testados
   - Resultados comparativos
   - Seleção do melhor

8. **Otimização**
   - Hiperparâmetros
   - Grid search
   - Resultados

9. **Exportação**
   - Formatos
   - Verificação
   - Artefatos

10. **API de Produção**
    - Arquitetura
    - Endpoints
    - Exemplos

11. **Reprodução**
    - Instruções passo a passo
    - Dependências
    - Comandos

12. **Conclusões**
    - Resultados alcançados
    - Aprendizados
    - Próximos passos

## Estatísticas do Projeto

**Dataset:**
- 2,930 casas
- 82 features originais
- 328 features após processamento

**Modelos:**
- 8 algoritmos testados
- Melhor: Gradient Boosting
- R² = 0.9235

**API:**
- 6+ endpoints
- FastAPI + Uvicorn
- Suporta pickle e ONNX

**Testes:**
- 100% de cobertura da API
- Erro médio < 5%
- Performance < 100ms

## Figuras e Gráficos

O relatório técnico menciona várias visualizações. Para gerar os gráficos:

```bash
# Abrir notebook de EDA
jupyter notebook notebooks/01_eda.ipynb

# Executar todas as células
# Gráficos serão gerados inline
```

## Exportação para Outros Formatos

### HTML
```bash
pandoc docs/relatorio_tecnico.md -o relatorio.html
```

### DOCX
```bash
pandoc docs/relatorio_tecnico.md -o relatorio.docx
```

### LaTeX
```bash
pandoc docs/relatorio_tecnico.md -o relatorio.tex
```
