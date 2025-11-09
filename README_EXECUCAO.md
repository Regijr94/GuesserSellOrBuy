# 📈 Sistema de Análise Fundamentalista de Ações

## 🚀 INSTALAÇÃO E EXECUÇÃO RÁPIDA

### Opção 1: Instalação Automática (Recomendado)

1. **Execute o instalador:**
   ```bash
   python instalar.py
   ```

2. **O script fará tudo automaticamente:**
   - Instala todas as dependências
   - Verifica a instalação
   - Oferece executar o sistema

### Opção 2: Instalação Manual

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute o sistema:**
   ```bash
   streamlit run analisador_acoes_completo.py
   ```

3. **Acesse no navegador:**
   ```
   http://localhost:8501
   ```

## 📋 Arquivos do Projeto

```
📁 pasta-do-projeto/
├── analisador_acoes_completo.py    # Sistema principal (ÚNICO ARQUIVO NECESSÁRIO)
├── requirements.txt                # Dependências
├── instalar.py                     # Instalador automático
└── README_EXECUCAO.md             # Este guia
```

## 🎯 Como Usar

1. **Abra o sistema** no navegador
2. **Digite uma ação** na barra lateral (ex: PETR4, VALE3, ITUB4)
3. **Selecione o período** de análise (1, 2 ou 5 anos)
4. **Clique em "Analisar Ação"**
5. **Veja os resultados:**
   - Preço atual e variação
   - Recomendação (COMPRAR/VENDER/MANTER)
   - Análise de risco (BAIXO/MÉDIO/ALTO)
   - Gráficos interativos
   - Indicadores fundamentalistas

## 🔧 Solução de Problemas

### ❌ Erro "ModuleNotFoundError"
```bash
# Reinstale as dependências
pip install --upgrade -r requirements.txt
```

### ❌ Erro "Port 8501 is already in use"  
```bash
# Use outra porta
streamlit run analisador_acoes_completo.py --server.port 8502
```

### ❌ Dados não carregam
- Verifique sua conexão com a internet
- Teste com ações populares: PETR4, VALE3, ITUB4
- Aguarde alguns segundos para carregamento

### ❌ Python não encontrado
1. Instale o Python 3.7+ do site oficial: https://python.org
2. Marque a opção "Add to PATH" durante instalação
3. Reinicie o terminal/prompt

## 📊 Exemplos de Ações para Testar

### 🏦 Bancos
- ITUB4 (Itaú)
- BBDC4 (Bradesco)  
- BBAS3 (Banco do Brasil)

### 🛢️ Petróleo
- PETR4 (Petrobrás)
- PETR3 (Petrobrás)

### ⛏️ Mineração
- VALE3 (Vale)
- GGBR4 (Gerdau)

### 🍺 Consumo
- ABEV3 (Ambev)
- JBSS3 (JBS)

### 🏭 Indústria
- WEGE3 (WEG)
- SUZB3 (Suzano)

## 💡 Recursos do Sistema

- ✅ **Dados em Tempo Real** via Yahoo Finance API
- ✅ **Análise Fundamentalista** com 6+ indicadores
- ✅ **Avaliação de Risco** quantitativa
- ✅ **Recomendações Inteligentes** com justificativas
- ✅ **Gráficos Interativos** com Plotly
- ✅ **Interface Responsiva** com Streamlit
- ✅ **Cache Inteligente** para performance

## 🆘 Suporte

Se encontrar problemas:

1. **Verifique os requisitos:**
   - Python 3.7 ou superior
   - Conexão com internet
   - Todas as dependências instaladas

2. **Teste o exemplo básico:**
   ```python
   import yfinance as yf
   print(yf.Ticker("PETR4.SA").info['regularMarketPrice'])
   ```

3. **Reinstale tudo:**
   ```bash
   pip uninstall -y streamlit yfinance pandas numpy plotly
   python instalar.py
   ```

---
**🎉 Sistema pronto para uso! Boa análise!**
