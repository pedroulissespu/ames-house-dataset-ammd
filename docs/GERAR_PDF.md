# Instruções para Gerar o Relatório Técnico em PDF

## Opção 1: Usar Pandoc (Recomendado)

### Instalação do Pandoc

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install pandoc texlive-latex-base texlive-fonts-recommended texlive-latex-extra
```

**Mac (com Homebrew):**
```bash
brew install pandoc basictex
```

**Windows:**
Baixe o instalador em: https://pandoc.org/installing.html

### Gerar PDF

```bash
cd docs/
pandoc relatorio_tecnico.md -o relatorio_tecnico.pdf --pdf-engine=xelatex -V geometry:margin=1in
```

## Opção 2: Usar Markdown to PDF (VS Code)

1. Instale a extensão "Markdown PDF" no VS Code
2. Abra `docs/relatorio_tecnico.md`
3. Pressione `Ctrl+Shift+P` (ou `Cmd+Shift+P` no Mac)
4. Digite "Markdown PDF: Export (pdf)"
5. Aguarde a geração do PDF

## Opção 3: Usar Grip + Navegador

```bash
# Instalar grip
pip install grip

# Visualizar no navegador
grip docs/relatorio_tecnico.md

# Acesse http://localhost:6419
# Use "Imprimir" > "Salvar como PDF" do navegador
```

## Opção 4: Converter Online

1. Acesse: https://www.markdowntopdf.com/
2. Faça upload do arquivo `relatorio_tecnico.md`
3. Clique em "Convert"
4. Baixe o PDF gerado

## Customização do PDF

Para melhorar a aparência do PDF com Pandoc:

```bash
pandoc relatorio_tecnico.md -o relatorio_tecnico.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V documentclass=article \
  -V colorlinks=true \
  --toc \
  --number-sections \
  --highlight-style=tango
```

Opções:
- `--toc`: Adiciona índice
- `--number-sections`: Numera seções
- `--highlight-style`: Estilo de código
- `-V geometry:margin=1in`: Margens
- `-V fontsize=11pt`: Tamanho da fonte

## Resultado Esperado

O PDF gerado terá aproximadamente:
- 25-30 páginas
- Tabelas formatadas
- Blocos de código destacados
- Estrutura de seções clara
- Pronto para entrega acadêmica

---

**Dica:** Para melhores resultados, use a Opção 1 (Pandoc) com as customizações sugeridas.
