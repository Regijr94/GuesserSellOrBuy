
# 📈 Analisador de Ações (Fundamentalista + Técnico)

Aplicação Streamlit que combina análise fundamentalista, avaliação de risco e indicadores técnicos para ações brasileiras (Yahoo Finance).

## 🚀 Destaques

- **Fundamentalista**: P/L, P/VP, Dividend Yield, ROE, Dívida/Patrimônio, Margem Líquida (pontuação ponderada e normalizada).
- **Risco**: Volatilidade anualizada, Sharpe, Sortino, VaR/CVaR 95%, Max Drawdown, Beta com comentários automáticos.
- **Técnico**: Médias móveis (SMA21/50/200), EMA9, MACD, RSI, ADX e score de tendência (compra / manter / venda).
- **Recomendação híbrida**: Combinação ponderada (45% fundamentos, 35% técnico, 20% risco) com justificativas e confiança.
- **Interface rica**: Gráficos Plotly, indicadores tabulares, cartões resumidos e histórico interativo.

## 🧩 Arquitetura

- `YFinanceProvider`: fonte de dados com cache de 5 minutos (`st.cache_data`).
- `EstrategiaFundamentalista`: scoring modular por indicador e justificativas detalhadas.
- `AvaliadorRisco`: consolida métricas estatísticas e gera score de 0 a 100 com comentários.
- `AnaliseTecnicaIndicadores`: utiliza `pandas-ta` para calcular tendência e momentum.
- `AnalisadorAcoes`: orquestra as camadas e gera a recomendação final composta.

## 🛠️ Pré-requisitos

- Python 3.12+
- Conta Streamlit Cloud (para deploy)

## 📦 Instalação Local

```bash
git clone git@github.com:Regijr94/GuesserSellOrBuy.git
cd GuesserSellOrBuy
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_analise_acoes.py
```

## ☁️ Deploy na Streamlit Cloud

1. Faça o push deste diretório para `git@github.com:Regijr94/GuesserSellOrBuy.git`.
2. No painel da Streamlit Cloud, clique em **New app** e selecione o repositório/branch `main`.
3. Informe `app_analise_acoes.py` como arquivo principal.
4. Confirme (as dependências são lidas de `requirements.txt`).

Pronto! A aplicação será disponibilizada com atualizações automáticas a cada novo push.

## 📁 Estrutura Sugerida do Repositório

```
GuesserSellOrBuy/
├── app_analise_acoes.py
├── requirements.txt
├── README.md
└── .gitignore
```

Arquivos auxiliares (`exemplo_uso.py`, scripts de experimentos) podem ser adicionados conforme necessidade, desde que mantidos fora do diretório `.streamlit` (ignorado por padrão).

## 🧪 Uso

1. Digite o ticker (ex.: `PETR4`, `VALE3`, `ITUB4`).
2. Selecione o período (1, 2 ou 5 anos).
3. Clique em **Analisar Ação**.
4. Explore métricas, gráficos, comentários e justificativas da recomendação.

## ⚠️ Observações

- Dados dependem da disponibilidade do Yahoo Finance.
- Tickers brasileiros precisam do sufixo `.SA` (adicionado automaticamente).
- Cache de 5 minutos evita excesso de chamadas à API.
- Indicadores técnicos exigem histórico suficiente (até 200 candles diários).

## 📬 Contato

Abra uma issue no repositório para dúvidas e sugestões.
