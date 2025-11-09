
#!/usr/bin/env python3
"""
SISTEMA DE ANÁLISE FUNDAMENTALISTA DE AÇÕES - VERSÃO UNIFICADA
Análise completa com POO, APIs e interface Streamlit
Autor: Sistema AI - 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
# from scipy import stats  # Usado apenas no sistema avançado
import warnings
warnings.filterwarnings('ignore')

# Importar sistema avançado
try:
    from analise_avancada import (
        AnalisadorRiscoAvancado, PreditorPrecosML, SistemaRecomendacaoAvancado,
        AnaliseRiscoAvancada, PredicaoPreco, RecomendacaoAvancada
    )
    SISTEMA_AVANCADO_DISPONIVEL = True
except ImportError:
    SISTEMA_AVANCADO_DISPONIVEL = False
    st.warning("⚠️ Sistema avançado não disponível. Instale scikit-learn para funcionalidades ML.")

# Sistema de Deep Learning
try:
    from deep_learning_system import SistemaDeepLearning, BacktestHistorico
    from transfer_learning_system import SistemaAvancadoDeepLearning, ValidacaoCruzadaTemporal
    DEEP_LEARNING_DISPONIVEL = True
except ImportError:
    DEEP_LEARNING_DISPONIVEL = False

# ===== CONFIGURAÇÃO DA PÁGINA STREAMLIT =====
st.set_page_config(
    page_title="📈 Analisador de Ações - POO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CONFIGURAÇÕES DO SISTEMA =====
class Config:
    """Configurações centralizadas do sistema"""

    # Dados financeiros
    CACHE_TTL = 300  # 5 minutos
    TAXA_LIVRE_RISCO = 0.1375  # CDI aproximado

    # Pesos para análise fundamentalista
    PESOS = {
        'pe_ratio': 0.25,
        'pb_ratio': 0.20,
        'dividend_yield': 0.15,
        'roe': 0.20,
        'debt_to_equity': 0.10,
        'profit_margin': 0.10
    }

    # Thresholds para classificação
    SCORE_COMPRA = 8.0
    SCORE_MANUTENCAO = 6.0
    SCORE_NEUTRO = 4.0

    # Ações populares brasileiras
    ACOES_POPULARES = [
        'PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3', 
        'WEGE3', 'MGLU3', 'RENT3', 'GGBR4', 'USIM5'
    ]

# ===== ESTRUTURAS DE DADOS =====

@dataclass
class DadosFinanceiros:
    """Dados financeiros básicos de uma ação"""
    ticker: str
    preco_atual: float
    variacao_dia: float
    volume: int
    market_cap: float
    timestamp: datetime

@dataclass
class AvaliacaoRisco:
    """Avaliação de risco de um investimento"""
    ticker: str
    volatilidade: float
    sharpe_ratio: float
    beta: float
    var_95: float
    max_drawdown: float
    classificacao_risco: str

# ===== CLASSES PRINCIPAIS =====

class FonteDados(ABC):
    """Interface para provedores de dados financeiros"""

    @abstractmethod
    def obter_dados_basicos(self, ticker: str) -> Optional[DadosFinanceiros]:
        pass

    @abstractmethod
    def obter_historico_precos(self, ticker: str, periodo: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def obter_indicadores_fundamentais(self, ticker: str) -> Dict:
        pass

class YFinanceProvider(FonteDados):
    """Provedor de dados usando Yahoo Finance"""

    def __init__(self):
        self.cache = {}
        self.cache_timestamp = {}

    def _formatar_ticker_brasileiro(self, ticker: str) -> str:
        """Adiciona .SA para ações brasileiras"""
        if not ticker.endswith('.SA'):
            return f"{ticker}.SA"
        return ticker
    
    def limpar_cache(self, ticker: str = None):
        """Limpa o cache para um ticker específico ou todo o cache"""
        if ticker:
            if ticker in self.cache:
                del self.cache[ticker]
            if ticker in self.cache_timestamp:
                del self.cache_timestamp[ticker]
        else:
            self.cache.clear()
            self.cache_timestamp.clear()

    @st.cache_data(ttl=Config.CACHE_TTL)
    def obter_dados_basicos(_self, ticker: str) -> Optional[DadosFinanceiros]:
        """Obtém dados básicos da ação"""
        try:
            ticker_formatado = _self._formatar_ticker_brasileiro(ticker)
            stock = yf.Ticker(ticker_formatado)
            info = stock.info
            hist = stock.history(period='2d')

            if hist.empty:
                return None

            preco_atual = hist['Close'].iloc[-1]
            preco_anterior = hist['Close'].iloc[-2] if len(hist) > 1 else preco_atual
            variacao_dia = ((preco_atual - preco_anterior) / preco_anterior) * 100

            return DadosFinanceiros(
                ticker=ticker,
                preco_atual=float(preco_atual),
                variacao_dia=float(variacao_dia),
                volume=int(hist['Volume'].iloc[-1]) if hist['Volume'].iloc[-1] > 0 else 0,
                market_cap=info.get('marketCap', 0),
                timestamp=datetime.now()
            )
        except Exception as e:
            st.error(f"Erro ao obter dados básicos para {ticker}: {e}")
            return None

    @st.cache_data(ttl=Config.CACHE_TTL)
    def obter_historico_precos(_self, ticker: str, periodo: str = '1y') -> pd.DataFrame:
        """Obtém histórico de preços"""
        try:
            ticker_formatado = _self._formatar_ticker_brasileiro(ticker)
            stock = yf.Ticker(ticker_formatado)
            hist = stock.history(period=periodo)
            return hist
        except Exception as e:
            st.error(f"Erro ao obter histórico para {ticker}: {e}")
            return pd.DataFrame()

    @st.cache_data(ttl=Config.CACHE_TTL)
    def obter_indicadores_fundamentais(_self, ticker: str) -> Dict:
        """Obtém indicadores fundamentalistas com validação"""
        try:
            ticker_formatado = _self._formatar_ticker_brasileiro(ticker)
            stock = yf.Ticker(ticker_formatado)
            info = stock.info

            # Validar e filtrar dados
            dados_validados = {}
            
            # P/L com validação
            pe = info.get('trailingPE')
            if pe and 0 < pe < 1000:
                dados_validados['pe_ratio'] = pe
            
            # P/VP com validação
            pb = info.get('priceToBook')
            if pb and 0 < pb < 50:
                dados_validados['pb_ratio'] = pb
            
            # ROE com validação
            roe = info.get('returnOnEquity')
            if roe and -1 < roe < 10:
                dados_validados['roe'] = roe
            
            # Dividend Yield com validação
            dy = info.get('dividendYield')
            if dy and 0 <= dy <= 0.5:  # 0% a 50%
                dados_validados['dividend_yield'] = dy * 100
            
            # Beta com validação
            beta = info.get('beta')
            if beta and 0 < beta < 5:
                dados_validados['beta'] = beta
            
            # EPS com validação
            eps = info.get('trailingEps')
            if eps and eps != 0:
                dados_validados['eps'] = eps
            
            # Book Value com validação
            book_value = info.get('bookValue')
            if book_value and book_value > 0:
                dados_validados['book_value'] = book_value
            
            # Market Cap com validação
            market_cap = info.get('marketCap')
            if market_cap and market_cap > 0:
                dados_validados['market_cap'] = market_cap
            
            # Debt to Equity
            debt_to_equity = info.get('debtToEquity')
            if debt_to_equity and debt_to_equity >= 0:
                dados_validados['debt_to_equity'] = debt_to_equity
            
            # Profit Margin
            profit_margin = info.get('profitMargins')
            if profit_margin and -1 < profit_margin < 1:
                dados_validados['profit_margin'] = profit_margin

            return dados_validados
        except Exception as e:
            st.error(f"Erro ao obter indicadores para {ticker}: {e}")
            return {}

class CalculadoraIndicadores:
    """Classe para cálculos estatísticos e de risco"""

    @staticmethod
    def calcular_volatilidade(precos: pd.Series, janela: int = 252) -> float:
        """Calcula volatilidade anualizada"""
        retornos = precos.pct_change().dropna()
        if len(retornos) < 2:
            return 0.0
        volatilidade = retornos.std() * np.sqrt(janela)
        return float(volatilidade)

    @staticmethod
    def calcular_sharpe_ratio(precos: pd.Series, taxa_livre_risco: float = Config.TAXA_LIVRE_RISCO) -> float:
        """Calcula índice Sharpe"""
        retornos = precos.pct_change().dropna()
        if len(retornos) < 2:
            return 0.0

        retorno_medio = retornos.mean() * 252
        volatilidade = retornos.std() * np.sqrt(252)

        if volatilidade == 0:
            return 0.0

        sharpe = (retorno_medio - taxa_livre_risco) / volatilidade
        return float(sharpe)

    @staticmethod
    def calcular_var_95(precos: pd.Series) -> float:
        """Calcula Value at Risk 95%"""
        retornos = precos.pct_change().dropna()
        if len(retornos) < 2:
            return 0.0
        var_95 = np.percentile(retornos, 5)
        return float(var_95 * 100)

    @staticmethod
    def calcular_max_drawdown(precos: pd.Series) -> float:
        """Calcula máximo drawdown"""
        if len(precos) < 2:
            return 0.0
        peak = precos.expanding(min_periods=1).max()
        drawdown = (precos / peak - 1.0)
        max_dd = drawdown.min()
        return float(max_dd * 100)

class EstrategiaFundamentalista:
    """Estratégia de análise baseada em indicadores fundamentalistas"""

    def __init__(self):
        self.pesos = Config.PESOS

    def analisar(self, dados: Dict) -> Dict:
        """Analisa indicadores e retorna score e recomendação com validação"""
        score = 0
        detalhes = {}
        indicadores_validos = 0

        # Análise P/L
        pe = dados.get('pe_ratio')
        if pe and pe > 0:
            if pe < 10:
                score_pe, analise_pe = 10, 'Excelente'
            elif pe < 15:
                score_pe, analise_pe = 8, 'Bom'
            elif pe < 25:
                score_pe, analise_pe = 5, 'Razoável'
            else:
                score_pe, analise_pe = 2, 'Alto'
            score += score_pe * self.pesos['pe_ratio']
            detalhes['pe_analise'] = analise_pe
            indicadores_validos += 1

        # Análise P/VP
        pb = dados.get('pb_ratio')
        if pb and pb > 0:
            if pb < 1:
                score_pb, analise_pb = 10, 'Subvalorizada'
            elif pb < 2:
                score_pb, analise_pb = 8, 'Boa'
            elif pb < 3:
                score_pb, analise_pb = 5, 'Razoável'
            else:
                score_pb, analise_pb = 2, 'Cara'
            score += score_pb * self.pesos['pb_ratio']
            detalhes['pb_analise'] = analise_pb
            indicadores_validos += 1

        # Análise ROE
        roe = dados.get('roe')
        if roe and roe > 0:
            roe_percent = roe * 100
            if roe_percent > 15:
                score_roe, analise_roe = 10, 'Excelente'
            elif roe_percent > 10:
                score_roe, analise_roe = 8, 'Bom'
            elif roe_percent > 5:
                score_roe, analise_roe = 5, 'Razoável'
            else:
                score_roe, analise_roe = 2, 'Baixo'
            score += score_roe * self.pesos['roe']
            detalhes['roe_analise'] = analise_roe
            indicadores_validos += 1

        # Análise Dividend Yield
        dy = dados.get('dividend_yield', 0)
        if dy and 0 < dy < 50:  # Filtro para dados válidos
            if dy > 6:
                score_dy, analise_dy = 10, 'Alto'
            elif dy > 4:
                score_dy, analise_dy = 8, 'Bom'
            elif dy > 2:
                score_dy, analise_dy = 5, 'Médio'
            else:
                score_dy, analise_dy = 2, 'Baixo'
            score += score_dy * self.pesos['dividend_yield']
            detalhes['dy_analise'] = analise_dy
            indicadores_validos += 1

        # Normalizar score baseado no número de indicadores válidos
        if indicadores_validos > 0:
            # Ajustar score para refletir a confiabilidade
            fator_confiabilidade = indicadores_validos / 4.0  # 4 indicadores principais
            score_normalizado = score * fator_confiabilidade
        else:
            score_normalizado = 0

        # Determinar confiabilidade
        if indicadores_validos >= 3:
            confiabilidade = 'ALTA'
        elif indicadores_validos >= 2:
            confiabilidade = 'MÉDIA'
        else:
            confiabilidade = 'BAIXA'

        # Determinação da recomendação
        if score_normalizado >= Config.SCORE_COMPRA:
            recomendacao = 'COMPRAR'
            justificativa = 'Indicadores fundamentalistas muito favoráveis'
        elif score_normalizado >= Config.SCORE_MANUTENCAO:
            recomendacao = 'MANTER'
            justificativa = 'Indicadores fundamentalistas favoráveis'
        elif score_normalizado >= Config.SCORE_NEUTRO:
            recomendacao = 'NEUTRO'
            justificativa = 'Indicadores fundamentalistas neutros'
        else:
            recomendacao = 'VENDER'
            justificativa = 'Indicadores fundamentalistas desfavoráveis'

        # Adicionar aviso se confiabilidade baixa
        if confiabilidade == 'BAIXA':
            justificativa += ' (⚠️ Poucos indicadores disponíveis)'

        return {
            'recomendacao': recomendacao,
            'score': score_normalizado,
            'justificativa': justificativa,
            'detalhes': detalhes,
            'indicadores_validos': indicadores_validos,
            'confiabilidade': confiabilidade
        }

class AvaliadorRisco:
    """Classe para avaliação de risco de investimentos"""

    def __init__(self):
        self.calculadora = CalculadoraIndicadores()

    def avaliar_risco(self, ticker: str, historico_precos: pd.DataFrame, beta: Optional[float] = None) -> AvaliacaoRisco:
        """Avalia risco completo de um ativo"""

        if historico_precos.empty:
            return AvaliacaoRisco(
                ticker=ticker, volatilidade=0, sharpe_ratio=0, beta=1.0,
                var_95=0, max_drawdown=0, classificacao_risco="INDETERMINADO"
            )

        precos = historico_precos['Close']

        # Cálculos de risco
        volatilidade = self.calculadora.calcular_volatilidade(precos)
        sharpe_ratio = self.calculadora.calcular_sharpe_ratio(precos)
        var_95 = self.calculadora.calcular_var_95(precos)
        max_drawdown = self.calculadora.calcular_max_drawdown(precos)
        beta_final = beta if beta is not None else 1.0

        # Classificação de risco baseada em múltiplos fatores
        score_risco = 0

        # Volatilidade (40% do score)
        if volatilidade < 0.15:
            score_risco += 40
        elif volatilidade < 0.25:
            score_risco += 25
        elif volatilidade < 0.35:
            score_risco += 15
        else:
            score_risco += 5

        # Beta (30% do score)
        if abs(beta_final - 1) < 0.2:
            score_risco += 30
        elif abs(beta_final - 1) < 0.5:
            score_risco += 20
        else:
            score_risco += 10

        # Sharpe Ratio (20% do score)
        if sharpe_ratio > 1:
            score_risco += 20
        elif sharpe_ratio > 0:
            score_risco += 15
        elif sharpe_ratio > -0.5:
            score_risco += 10
        else:
            score_risco += 5

        # Max Drawdown (10% do score)
        if abs(max_drawdown) < 10:
            score_risco += 10
        elif abs(max_drawdown) < 20:
            score_risco += 7
        elif abs(max_drawdown) < 30:
            score_risco += 4
        else:
            score_risco += 2

        # Classificação final
        if score_risco >= 80:
            classificacao = "BAIXO"
        elif score_risco >= 60:
            classificacao = "MÉDIO"
        else:
            classificacao = "ALTO"

        return AvaliacaoRisco(
            ticker=ticker,
            volatilidade=volatilidade,
            sharpe_ratio=sharpe_ratio,
            beta=beta_final,
            var_95=var_95,
            max_drawdown=max_drawdown,
            classificacao_risco=classificacao
        )

class AnalisadorAcoes:
    """Sistema principal de análise de ações"""

    def __init__(self, fonte_dados: FonteDados):
        self.fonte_dados = fonte_dados
        self.estrategia = EstrategiaFundamentalista()
        self.avaliador_risco = AvaliadorRisco()

    def analisar_acao(self, ticker: str, periodo: str = '1y') -> Dict:
        """Análise completa de uma ação"""
        resultado = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'sucesso': False
        }

        try:
            # 1. Obter dados básicos
            dados_basicos = self.fonte_dados.obter_dados_basicos(ticker)
            if not dados_basicos:
                resultado['erro'] = 'Não foi possível obter dados básicos'
                return resultado

            # 2. Obter indicadores fundamentalistas
            indicadores = self.fonte_dados.obter_indicadores_fundamentais(ticker)
            if not indicadores:
                resultado['erro'] = 'Não foi possível obter indicadores fundamentalistas'
                return resultado

            # 3. Obter histórico para análise de risco
            historico = self.fonte_dados.obter_historico_precos(ticker, periodo)

            # 4. Análise fundamentalista
            analise_fund = self.estrategia.analisar(indicadores)

            # 5. Análise de risco
            beta = indicadores.get('beta')
            avaliacao_risco = self.avaliador_risco.avaliar_risco(ticker, historico, beta)

            # 6. Compilar resultado
            resultado.update({
                'sucesso': True,
                'dados_basicos': dados_basicos,
                'indicadores_fundamentais': indicadores,
                'analise_fundamentalista': analise_fund,
                'avaliacao_risco': avaliacao_risco,
                'historico': historico,
                'recomendacao_final': self._recomendacao_final(analise_fund, avaliacao_risco)
            })

        except Exception as e:
            resultado['erro'] = f'Erro na análise: {str(e)}'

        return resultado

    def _recomendacao_final(self, analise_fund: Dict, avaliacao_risco: AvaliacaoRisco) -> Dict:
        """Combina análise fundamentalista com avaliação de risco"""

        recom_fund = analise_fund['recomendacao']
        risco = avaliacao_risco.classificacao_risco

        # Matriz de decisão
        if recom_fund == 'COMPRAR':
            if risco == 'BAIXO':
                final = 'COMPRA FORTE'
                justificativa = 'Excelentes fundamentos com baixo risco'
            elif risco == 'MÉDIO':
                final = 'COMPRAR'
                justificativa = 'Bons fundamentos com risco moderado'
            else:
                final = 'CAUTELA'
                justificativa = 'Bons fundamentos mas risco elevado'
        elif recom_fund == 'MANTER':
            if risco == 'BAIXO':
                final = 'MANTER'
                justificativa = 'Fundamentos razoáveis com baixo risco'
            elif risco == 'MÉDIO':
                final = 'MANTER'
                justificativa = 'Fundamentos e risco equilibrados'
            else:
                final = 'REDUZIR POSIÇÃO'
                justificativa = 'Fundamentos razoáveis mas risco elevado'
        else:  # VENDER ou NEUTRO
            if risco == 'ALTO':
                final = 'VENDA FORTE'
                justificativa = 'Fundamentos fracos e alto risco'
            else:
                final = 'VENDER'
                justificativa = 'Fundamentos desfavoráveis'

        # Calcular confiança
        confianca_base = analise_fund['score'] / 10 * 100
        if risco == 'BAIXO':
            confianca_base *= 1.1
        elif risco == 'ALTO':
            confianca_base *= 0.8

        if confianca_base >= 80:
            confianca = 'ALTA'
        elif confianca_base >= 60:
            confianca = 'MÉDIA'
        else:
            confianca = 'BAIXA'

        return {
            'recomendacao': final,
            'justificativa': justificativa,
            'confianca': confianca
        }

# ===== UTILITÁRIOS =====

def formatar_moeda(valor: float) -> str:
    """Formata valor em moeda brasileira"""
    return f"R$ {valor:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

def formatar_percentual(valor: float) -> str:
    """Formata valor como percentual"""
    return f"{valor:.2f}%"

def obter_cor_recomendacao(recomendacao: str) -> str:
    """Retorna cor baseada na recomendação"""
    if "COMPRA" in recomendacao:
        return "#28a745"
    elif "VEND" in recomendacao:
        return "#dc3545"
    else:
        return "#ffc107"

# ===== INTERFACE STREAMLIT =====

def exibir_tela_inicial():
    """Tela inicial com informações do sistema"""
    st.markdown("""
    ## 🎯 Sistema de Análise Fundamentalista de Ações

    **Desenvolvido com Programação Orientada a Objetos (POO)**

    ### 🧠 **Sistema de Deep Learning com Backtest Histórico**
    - **✨ Atualização em Tempo Real**: Digite um ticker e os dados são atualizados automaticamente
    - **🧠 Deep Learning Nativo**: LSTM + CNN + Ensemble para predições avançadas
    - **🔄 Transfer Learning**: Conhecimento transferido entre diferentes ações
    - **📊 Validação Cruzada Temporal**: Backtest robusto com múltiplos folds
    - **🤖 Machine Learning**: Random Forest, Gradient Boosting e Ridge
    - **📈 Análise Quantitativa**: VaR condicional, CVaR, Hurst Exponent, Calmar Ratio
    - **🎯 Recomendações Multi-Fatoriais**: Combina fundamentos, técnica, momentum e risco
    - **🔍 Validação Avançada**: Sistema de confiabilidade e probabilidade de sucesso
    - **📈 Backtest Histórico**: Treinamento com dados de 2015-2024
    - **🎮 Interface Unificada**: Tudo em uma única experiência

    ### 🔍 **Funcionalidades**

    #### 📊 Análise Fundamentalista
    - **P/L (Preço/Lucro)**: Avalia se a ação está cara ou barata
    - **P/VP (Preço/Valor Patrimonial)**: Compara preço com valor contábil
    - **ROE (Return on Equity)**: Mede a eficiência da empresa
    - **Dividend Yield**: Retorno em dividendos

    #### ⚡ Avaliação de Risco
    - **Volatilidade**: Medida de instabilidade dos preços
    - **Índice Sharpe**: Relação risco-retorno
    - **VaR (Value at Risk)**: Perda máxima esperada
    - **Maximum Drawdown**: Maior queda histórica

    #### 🤖 Recomendações Inteligentes
    - **Sistema de Scoring**: Combina múltiplos indicadores
    - **Matriz de Decisão**: Considera fundamentos + risco
    - **Níveis de Confiança**: Alta, Média, Baixa

    ### 💡 **Como Usar**
    1. **Digite um ticker** na barra lateral (ex: PETR4, VALE3, ITUB4)
    2. **Os dados são atualizados automaticamente** quando você digita
    3. **Selecione o período** de análise (1, 2 ou 5 anos)
    4. **Veja a análise avançada** com ML e técnicas quantitativas automaticamente
    5. **Use os botões** para atualizar ou limpar cache quando necessário

    **👈 Digite uma ação na barra lateral para começar!**
    """)

def exibir_metricas_principais(dados: DadosFinanceiros, recom_final: Dict, risco: AvaliacaoRisco):
    """Exibe métricas principais em cards"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cor_variacao = "inverse" if dados.variacao_dia < 0 else "normal"
        st.metric(
            "Preço Atual",
            formatar_moeda(dados.preco_atual),
            formatar_percentual(dados.variacao_dia),
            delta_color=cor_variacao
        )

    with col2:
        cor_emoji = "🟢" if "COMPRA" in recom_final['recomendacao'] else "🔴" if "VEND" in recom_final['recomendacao'] else "🟡"
        st.metric(
            "Recomendação",
            f"{cor_emoji} {recom_final['recomendacao']}"
        )

    with col3:
        cor_risco = "🟢" if risco.classificacao_risco == "BAIXO" else "🟡" if risco.classificacao_risco == "MÉDIO" else "🔴"
        st.metric(
            "Risco",
            f"{cor_risco} {risco.classificacao_risco}"
        )

    with col4:
        st.metric(
            "Confiança",
            f"📈 {recom_final['confianca']}"
        )

def criar_grafico_precos(historico: pd.DataFrame, ticker: str):
    """Cria gráfico de evolução de preços"""
    if historico.empty:
        st.warning("Dados históricos não disponíveis")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historico.index,
        y=historico['Close'],
        mode='lines',
        name='Preço de Fechamento',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.update_layout(
        title=f'Evolução do Preço - {ticker}',
        xaxis_title='Data',
        yaxis_title='Preço (R$)',
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

def criar_grafico_retornos(historico: pd.DataFrame):
    """Cria gráfico de distribuição dos retornos"""
    if historico.empty:
        st.warning("Dados históricos não disponíveis")
        return

    retornos = historico['Close'].pct_change().dropna() * 100

    if len(retornos) == 0:
        st.warning("Não há dados suficientes para análise de retornos")
        return

    fig = px.histogram(
        retornos, 
        nbins=30, 
        title='Distribuição dos Retornos Diários (%)',
        labels={'value': 'Retorno Diário (%)', 'count': 'Frequência'}
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def criar_gauge_score(score: float):
    """Cria gauge do score fundamentalista"""
    score_normalizado = min(score, 10) / 10 * 100

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score_normalizado,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Score Fundamentalista (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 40], 'color': "#ffcccc"},
                {'range': [40, 60], 'color': "#fff3cd"},
                {'range': [60, 80], 'color': "#d4edda"},
                {'range': [80, 100], 'color': "#d1ecf1"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def exibir_tabelas_analise(indicadores: Dict, risco: AvaliacaoRisco, analise_fund: Dict):
    """Exibe tabelas com indicadores e análises"""
    col_esquerda, col_direita = st.columns(2)

    with col_esquerda:
        st.markdown("### 🔍 Indicadores Fundamentalistas")

        # Preparar dados para tabela
        dados_tabela = []

        if indicadores.get('pe_ratio'):
            dados_tabela.append(['P/L (Preço/Lucro)', f"{indicadores['pe_ratio']:.2f}"])
        if indicadores.get('pb_ratio'):
            dados_tabela.append(['P/VP (Preço/VP)', f"{indicadores['pb_ratio']:.2f}"])
        if indicadores.get('dividend_yield'):
            dados_tabela.append(['Dividend Yield', formatar_percentual(indicadores['dividend_yield'])])
        if indicadores.get('roe'):
            dados_tabela.append(['ROE', formatar_percentual(indicadores['roe']*100)])
        if indicadores.get('beta'):
            dados_tabela.append(['Beta', f"{indicadores['beta']:.2f}"])
        if indicadores.get('eps'):
            dados_tabela.append(['EPS (Lucro/Ação)', formatar_moeda(indicadores['eps'])])

        if dados_tabela:
            df_indicadores = pd.DataFrame(dados_tabela, columns=['Indicador', 'Valor'])
            st.dataframe(df_indicadores, use_container_width=True, hide_index=True)
        else:
            st.warning("Indicadores fundamentalistas não disponíveis")

        # Score fundamentalista
        st.markdown("### 📊 Score Fundamentalista")
        criar_gauge_score(analise_fund['score'])

    with col_direita:
        st.markdown("### ⚠️ Análise de Risco")

        # Métricas de risco
        dados_risco = [
            ['Volatilidade Anual', f"{risco.volatilidade:.2f}"],
            ['Índice Sharpe', f"{risco.sharpe_ratio:.2f}"],
            ['VaR 95%', formatar_percentual(risco.var_95)],
            ['Max Drawdown', formatar_percentual(risco.max_drawdown)],
            ['Beta', f"{risco.beta:.2f}"],
            ['Classificação', risco.classificacao_risco]
        ]

        df_risco = pd.DataFrame(dados_risco, columns=['Métrica', 'Valor'])
        st.dataframe(df_risco, use_container_width=True, hide_index=True)

        # Gráfico de retornos
        st.markdown("### 📊 Distribuição dos Retornos")

def exibir_recomendacao_final(recom_final: Dict, analise_fund: Dict):
    """Exibe a recomendação final com destaque"""

    # Determinar cor de fundo baseada na recomendação
    if "COMPRA" in recom_final['recomendacao']:
        cor_fundo = "#d4edda"
        cor_texto = "#155724"
        icone = "🟢"
    elif "VEND" in recom_final['recomendacao']:
        cor_fundo = "#f8d7da"
        cor_texto = "#721c24"
        icone = "🔴"
    else:
        cor_fundo = "#fff3cd"
        cor_texto = "#856404"
        icone = "🟡"

    # Adicionar informação de confiabilidade
    confiabilidade = analise_fund.get('confiabilidade', 'MÉDIA')
    indicadores_validos = analise_fund.get('indicadores_validos', 0)
    
    st.markdown(f"""
    <div style="
        background-color: {cor_fundo};
        color: {cor_texto};
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {cor_texto};
        margin: 20px 0;
    ">
        <h3>{icone} {recom_final['recomendacao']}</h3>
        <p><strong>💡 Justificativa:</strong> {recom_final['justificativa']}</p>
        <p><strong>🎯 Nível de Confiança:</strong> {recom_final['confianca']}</p>
        <p><strong>📊 Score Fundamentalista:</strong> {analise_fund['score']:.2f}/10</p>
        <p><strong>🔍 Confiabilidade dos Dados:</strong> {confiabilidade} ({indicadores_validos}/4 indicadores)</p>
    </div>
    """, unsafe_allow_html=True)

def executar_analise_deep_learning(ticker: str, periodo: str) -> Dict:
    """Executa análise com deep learning e backtest histórico"""
    
    resultado = {
        'ticker': ticker,
        'timestamp': datetime.now(),
        'sucesso': False,
        'tipo_analise': 'DEEP_LEARNING'
    }
    
    try:
        # Inicializar sistema de deep learning
        sistema_dl = SistemaDeepLearning(ticker)
        
        # Configurar sistema
        X_train, X_test, y_train, y_test = sistema_dl.configurar_sistema(
            janela_temporal=60,
            split_temporal=0.8
        )
        
        # Treinar sistema (com menos épocas para interface)
        resultados_treinamento = sistema_dl.treinar_sistema(
            X_train, y_train, X_test, y_test,
            epochs=30, batch_size=32
        )
        
        # Obter dados básicos
        fonte_dados = YFinanceProvider()
        dados_basicos = fonte_dados.obter_dados_basicos(ticker)
        indicadores = fonte_dados.obter_indicadores_fundamentais(ticker)
        historico = fonte_dados.obter_historico_precos(ticker, periodo)
        
        if not dados_basicos or historico.empty:
            resultado['erro'] = 'Dados insuficientes para análise'
            return resultado
        
        # Fazer predições
        predicoes = sistema_dl.fazer_predicoes(X_test[-10:])  # Últimas 10 predições
        metricas = sistema_dl.obter_metricas()
        relatorio = sistema_dl.gerar_relatorio()
        
        # Compilar resultado
        resultado.update({
            'sucesso': True,
            'dados_basicos': dados_basicos,
            'indicadores_fundamentais': indicadores,
            'historico': historico,
            'deep_learning': {
                'metricas': metricas,
                'relatorio': relatorio,
                'predicoes': predicoes.flatten().tolist(),
                'qualidade_modelo': relatorio['qualidade_modelo']
            }
        })
        
    except Exception as e:
        resultado['erro'] = f'Erro na análise de deep learning: {str(e)}'
    
    return resultado

def executar_analise_avancada(ticker: str, periodo: str) -> Dict:
    """Executa análise avançada com ML e técnicas quantitativas"""
    
    resultado = {
        'ticker': ticker,
        'timestamp': datetime.now(),
        'sucesso': False,
        'tipo_analise': 'AVANCADA'
    }
    
    try:
        # Inicializar componentes avançados
        analisador_risco = AnalisadorRiscoAvancado()
        preditor_ml = PreditorPrecosML()
        sistema_recomendacao = SistemaRecomendacaoAvancado()
        
        # Obter dados
        fonte_dados = YFinanceProvider()
        dados_basicos = fonte_dados.obter_dados_basicos(ticker)
        indicadores = fonte_dados.obter_indicadores_fundamentais(ticker)
        historico = fonte_dados.obter_historico_precos(ticker, periodo)
        
        if not dados_basicos or historico.empty:
            resultado['erro'] = 'Dados insuficientes para análise avançada'
            return resultado
        
        # Análise de risco avançada
        analise_risco = analisador_risco.analisar_risco_completo(ticker, historico)
        
        # Predição com ML
        predicao = preditor_ml.prever_precos(ticker, historico)
        
        # Recomendação avançada
        recomendacao = sistema_recomendacao.gerar_recomendacao(
            ticker, indicadores, analise_risco, predicao, historico
        )
        
        # Compilar resultado
        resultado.update({
            'sucesso': True,
            'dados_basicos': dados_basicos,
            'indicadores_fundamentais': indicadores,
            'analise_risco_avancada': analise_risco,
            'predicao_ml': predicao,
            'recomendacao_avancada': recomendacao,
            'historico': historico
        })
        
    except Exception as e:
        resultado['erro'] = f'Erro na análise avançada: {str(e)}'
    
    return resultado

def exibir_secao_backtest(ticker: str, periodo: str):  # pylint: disable=unused-argument
    """Exibe seção de backtest (simulada)"""
    
    st.markdown("### 📈 Análise de Performance Histórica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚠️ **Importante sobre Backtest**")
        st.info("""
        **Por que precisamos de backtest?**
        
        - ✅ **Validação de Estratégias**: Testa se as recomendações realmente funcionam
        - ✅ **Otimização de Parâmetros**: Ajusta pesos e thresholds baseado em dados históricos
        - ✅ **Análise de Risco**: Avalia drawdowns e volatilidade real
        - ✅ **Confiança nas Recomendações**: Reduz falsos positivos
        
        **Problemas do sistema atual:**
        - ❌ Dados do Yahoo Finance podem estar desatualizados
        - ❌ Indicadores não são calculados, apenas copiados
        - ❌ Não há validação histórica das estratégias
        - ❌ Pesos dos indicadores são arbitrários
        """)
    
    with col2:
        st.markdown("#### 🚀 **Sistema de Backtest Implementado**")
        st.success("""
        **Melhorias implementadas:**
        
        - ✅ **Validação de Dados**: Filtros para valores inconsistentes
        - ✅ **Cálculo de Confiabilidade**: Baseado em indicadores disponíveis
        - ✅ **Sistema de Backtest**: Arquivo `sistema_backtest.py` criado
        - ✅ **Indicadores Corrigidos**: Validação e normalização
        
        **Para ativar o backtest completo:**
        1. Execute `python sistema_backtest.py`
        2. Veja análise histórica real
        3. Valide performance das estratégias
        """)
    
    # Simular alguns dados de backtest
    st.markdown("#### 📊 Simulação de Métricas Históricas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Retorno Anualizado", "12.5%", "2.3%")
    
    with col2:
        st.metric("Sharpe Ratio", "1.2", "0.1")
    
    with col3:
        st.metric("Max Drawdown", "-8.5%", "-1.2%")
    
    with col4:
        st.metric("Win Rate", "65%", "5%")
    
    st.warning("⚠️ **Dados simulados** - Execute o sistema de backtest completo para dados reais!")

def exibir_analise_deep_learning(resultado: Dict, periodo: str) -> None:
    """Exibe resultados da análise de deep learning"""
    
    st.markdown("### 🧠 Análise com Deep Learning")
    
    # Informações básicas
    dados = resultado['dados_basicos']
    dl_data = resultado['deep_learning']
    metricas = dl_data['metricas']
    relatorio = dl_data['relatorio']
    
    # Cards principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Preço Atual",
            value=f"R$ {dados.preco_atual:.2f}",
            delta=f"{dados.variacao_dia:.2f}%"
        )
    
    with col2:
        st.metric(
            label="🎯 R² Score",
            value=f"{metricas['r2_score']:.3f}",
            help="Qualidade do modelo (0-1)"
        )
    
    with col3:
        st.metric(
            label="📊 RMSE",
            value=f"{metricas['rmse']:.4f}",
            help="Erro quadrático médio"
        )
    
    with col4:
        qualidade = dl_data['qualidade_modelo']
        cor = "#28a745" if qualidade in ["EXCELENTE", "BOM"] else "#ffc107" if qualidade == "REGULAR" else "#dc3545"
        st.markdown(f"""
        <div style="background-color: {cor}; padding: 10px; border-radius: 5px; text-align: center;">
            <strong style="color: white;">{qualidade}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Informações do modelo
    st.markdown("#### 🤖 Informações do Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **📈 Modelos Utilizados:**
        - {', '.join(relatorio['modelos_ensemble'])}
        
        **🎯 Target:** {relatorio['target']}
        
        **📊 Features:** {len(relatorio['features_utilizadas'])} indicadores
        """)
    
    with col2:
        st.markdown(f"""
        **⚖️ Pesos do Ensemble:**
        - LSTM: {relatorio['pesos_ensemble'].get('LSTM', 0):.1%}
        - CNN: {relatorio['pesos_ensemble'].get('CNN', 0):.1%}
        
        **📅 Treinado em:** {relatorio['timestamp']}
        """)
    
    # Predições
    st.markdown("#### 🔮 Predições do Modelo")
    
    predicoes = dl_data['predicoes']
    if predicoes:
        # Calcular preços futuros baseados nas predições
        preco_atual = dados.preco_atual
        preco_medio_predito = preco_atual * (1 + np.mean(predicoes))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="📈 Preço Médio Predito",
                value=f"R$ {preco_medio_predito:.2f}",
                delta=f"{((preco_medio_predito / preco_atual) - 1) * 100:.2f}%"
            )
        
        with col2:
            st.metric(
                label="📊 Volatilidade Predita",
                value=f"{np.std(predicoes) * 100:.2f}%",
                help="Desvio padrão das predições"
            )
        
        with col3:
            tendencia = "ALTA" if np.mean(predicoes) > 0 else "BAIXA"
            cor_tendencia = "#28a745" if tendencia == "ALTA" else "#dc3545"
            st.markdown(f"""
            <div style="background-color: {cor_tendencia}; padding: 10px; border-radius: 5px; text-align: center;">
                <strong style="color: white;">Tendência: {tendencia}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Recomendações baseadas em deep learning
    st.markdown("#### 💡 Recomendações Baseadas em Deep Learning")
    
    if metricas['r2_score'] > 0.5:
        if np.mean(predicoes) > 0.02:  # Retorno > 2%
            recomendacao = "COMPRAR"
            cor = "#28a745"
            justificativa = "Modelo confiável prediz alta probabilidade de ganhos"
        elif np.mean(predicoes) < -0.02:  # Retorno < -2%
            recomendacao = "VENDER"
            cor = "#dc3545"
            justificativa = "Modelo confiável prediz alta probabilidade de perdas"
        else:
            recomendacao = "MANTER"
            cor = "#ffc107"
            justificativa = "Modelo prediz movimentos neutros"
    else:
        recomendacao = "AGUARDAR"
        cor = "#6c757d"
        justificativa = "Modelo com baixa confiabilidade - aguardar mais dados"
    
    st.markdown(f"""
    <div style="background-color: {cor}; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">{recomendacao}</h3>
        <p style="color: white; margin: 5px 0 0 0;">{justificativa}</p>
    </div>
    """, unsafe_allow_html=True)

def exibir_analise_avancada(resultado: Dict, ticker: str, periodo: str):  # pylint: disable=unused-argument
    """Exibe análise avançada com ML e técnicas quantitativas"""
    
    dados = resultado['dados_basicos']
    # indicadores = resultado['indicadores_fundamentais']  # Usado apenas em algumas seções
    analise_risco = resultado['analise_risco_avancada']
    predicao = resultado['predicao_ml']
    recomendacao = resultado['recomendacao_avancada']
    historico = resultado['historico']
    
    # Indicador de análise avançada
    st.success("🚀 **Análise Avançada Ativa** - ML + Técnicas Quantitativas")
    
    # Métricas principais avançadas
    st.markdown("### 📊 Métricas Principais Avançadas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cor_variacao = "inverse" if dados.variacao_dia < 0 else "normal"
        st.metric(
            "Preço Atual",
            formatar_moeda(dados.preco_atual),
            formatar_percentual(dados.variacao_dia),
            delta_color=cor_variacao
        )
    
    with col2:
        cor_emoji = "🟢" if "COMPRA" in recomendacao.recomendacao else "🔴" if "VEND" in recomendacao.recomendacao else "🟡"
        st.metric(
            "Recomendação ML",
            f"{cor_emoji} {recomendacao.recomendacao}"
        )
    
    with col3:
        st.metric(
            "Score Final",
            f"{recomendacao.score_final:.1f}/10"
        )
    
    with col4:
        st.metric(
            "Prob. Sucesso",
            f"{recomendacao.probabilidade_sucesso:.0f}%"
        )
    
    # Análise de risco avançada
    st.markdown("### ⚠️ Análise de Risco Avançada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Métricas de Risco Quantitativas")
        
        dados_risco = [
            ['VaR 95%', f"{analise_risco.var_95:.2f}%"],
            ['CVaR 95%', f"{analise_risco.cvar_95:.2f}%"],
            ['Max Drawdown', f"{analise_risco.max_drawdown:.2f}%"],
            ['Calmar Ratio', f"{analise_risco.calmar_ratio:.2f}"],
            ['Sortino Ratio', f"{analise_risco.sortino_ratio:.2f}"],
            ['Omega Ratio', f"{analise_risco.omega_ratio:.2f}"],
            ['Tail Ratio', f"{analise_risco.tail_ratio:.2f}"],
            ['Risk Score', f"{analise_risco.risk_score:.1f}/100"],
            ['Risk Rating', analise_risco.risk_rating]
        ]
        
        df_risco = pd.DataFrame(dados_risco, columns=['Métrica', 'Valor'])
        st.dataframe(df_risco, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 📊 Análise Estatística")
        
        dados_estat = [
            ['Skewness', f"{analise_risco.skewness:.3f}"],
            ['Kurtosis', f"{analise_risco.kurtosis:.3f}"],
            ['Hurst Exponent', f"{analise_risco.hurst_exponent:.3f}"],
            ['Regime Volatilidade', analise_risco.volatility_regime],
            ['Jarque-Bera p-value', f"{analise_risco.jarque_bera_pvalue:.3f}"],
            ['Common Sense Ratio', f"{analise_risco.common_sense_ratio:.3f}"]
        ]
        
        df_estat = pd.DataFrame(dados_estat, columns=['Métrica', 'Valor'])
        st.dataframe(df_estat, use_container_width=True, hide_index=True)
        
        # Interpretação do Hurst Exponent
        hurst = analise_risco.hurst_exponent
        if hurst > 0.6:
            st.info("🔄 **Tendência**: Hurst > 0.6 indica comportamento de tendência")
        elif hurst < 0.4:
            st.info("🔄 **Mean Reversion**: Hurst < 0.4 indica reversão à média")
        else:
            st.info("🔄 **Aleatório**: Hurst ≈ 0.5 indica comportamento aleatório")
    
    # Predições com ML
    st.markdown("### 🤖 Predições com Machine Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Previsões de Preço")
        
        previsoes_data = [
            ['1 Mês', f"R$ {predicao.previsao_1m:.2f}", f"{predicao.confianca_1m:.0f}%"],
            ['3 Meses', f"R$ {predicao.previsao_3m:.2f}", f"{predicao.confianca_3m:.0f}%"],
            ['6 Meses', f"R$ {predicao.previsao_6m:.2f}", f"{predicao.confianca_6m:.0f}%"],
            ['12 Meses', f"R$ {predicao.previsao_12m:.2f}", f"{predicao.confianca_12m:.0f}%"]
        ]
        
        df_previsoes = pd.DataFrame(previsoes_data, columns=['Horizonte', 'Previsão', 'Confiança'])
        st.dataframe(df_previsoes, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 🎯 Qualidade do Modelo")
        
        st.metric("Modelo Usado", predicao.modelo_usado)
        st.metric("R² Score", f"{predicao.r2_score:.3f}")
        st.metric("RMSE", f"R$ {predicao.rmse:.2f}")
        
        # Interpretação do R²
        if predicao.r2_score > 0.7:
            st.success("✅ **Excelente**: R² > 0.7 indica modelo muito confiável")
        elif predicao.r2_score > 0.5:
            st.warning("⚠️ **Bom**: R² > 0.5 indica modelo moderadamente confiável")
        else:
            st.error("❌ **Baixo**: R² < 0.5 indica modelo pouco confiável")
    
    # Recomendação avançada
    st.markdown("### 💡 Recomendação Avançada Multi-Fatorial")
    
    # Determinar cor baseada na recomendação
    if "COMPRA" in recomendacao.recomendacao:
        cor_fundo = "#d4edda"
        cor_texto = "#155724"
        icone = "🟢"
    elif "VEND" in recomendacao.recomendacao:
        cor_fundo = "#f8d7da"
        cor_texto = "#721c24"
        icone = "🔴"
    else:
        cor_fundo = "#fff3cd"
        cor_texto = "#856404"
        icone = "🟡"
    
    st.markdown(f"""
    <div style="
        background-color: {cor_fundo};
        color: {cor_texto};
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {cor_texto};
        margin: 20px 0;
    ">
        <h3>{icone} {recomendacao.recomendacao}</h3>
        <p><strong>💡 Justificativa:</strong> {recomendacao.justificativa}</p>
        <p><strong>🎯 Confiança:</strong> {recomendacao.confianca:.0f}%</p>
        <p><strong>📊 Score Final:</strong> {recomendacao.score_final:.1f}/10</p>
        <p><strong>🎲 Probabilidade de Sucesso:</strong> {recomendacao.probabilidade_sucesso:.0f}%</p>
        <p><strong>⏰ Horizonte Ótimo:</strong> {recomendacao.horizonte_otimo}</p>
        <p><strong>🔑 Fatores Chave:</strong> {', '.join(recomendacao.fatores_chave)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Scores detalhados
    st.markdown("### 📊 Breakdown dos Scores")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Score Fundamentalista", f"{recomendacao.score_fundamentalista:.1f}/10")
    
    with col2:
        st.metric("Score Técnico", f"{recomendacao.score_tecnico:.1f}/10")
    
    with col3:
        st.metric("Score Momentum", f"{recomendacao.score_momentum:.1f}/10")
    
    with col4:
        st.metric("Score Risco", f"{recomendacao.score_risco:.1f}/10")
    
    # Níveis de operação
    st.markdown("### 🎯 Níveis de Operação")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Stop Loss", f"R$ {recomendacao.stop_loss:.2f}")
    
    with col2:
        st.metric("Take Profit", f"R$ {recomendacao.take_profit:.2f}")
    
    # Gráficos
    st.markdown("### 📈 Análise Gráfica")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Evolução do Preço")
        criar_grafico_precos(historico, ticker)
    
    with col2:
        st.markdown("#### Distribuição dos Retornos")
        criar_grafico_retornos(historico)

def main():
    """Função principal da aplicação"""

    st.title("📈 Sistema de Análise Fundamentalista de Ações")
    
    # Indicador de sistema avançado
    if DEEP_LEARNING_DISPONIVEL:
        st.markdown("### 🧠 Deep Learning + ML + Técnicas Quantitativas")
        st.success("✅ **Sistema de Deep Learning Ativo** - LSTM, CNN, Transfer Learning, Validação Cruzada e muito mais!")
    elif SISTEMA_AVANCADO_DISPONIVEL:
        st.markdown("### 🚀 Análise Avançada com ML + Técnicas Quantitativas")
        st.success("✅ **Sistema Avançado Ativo** - Machine Learning, VaR, CVaR, Hurst Exponent e muito mais!")
    else:
        st.markdown("### 🤖 Análise baseada em POO e APIs em tempo real")
        st.warning("⚠️ **Sistema Básico** - Instale scikit-learn para análise avançada")

    # Inicializar variáveis de sessão
    if 'ticker_atual' not in st.session_state:
        st.session_state.ticker_atual = "PETR4"
    if 'ultimo_ticker_analisado' not in st.session_state:
        st.session_state.ultimo_ticker_analisado = None
    if 'resultado_analise' not in st.session_state:
        st.session_state.resultado_analise = None
    if 'periodo_atual' not in st.session_state:
        st.session_state.periodo_atual = "1y"

    # Sidebar
    st.sidebar.title("🔧 Configurações")
    st.sidebar.markdown("---")

    # Input da ação com detecção automática de mudança
    st.sidebar.markdown("**📝 Digite o ticker da ação:**")
    ticker_input = st.sidebar.text_input(
        "Ticker", 
        value=st.session_state.ticker_atual,
        help="Digite apenas o código da ação (ex: PETR4, VALE3). Os dados serão atualizados automaticamente.",
        key="ticker_input",
        placeholder="Ex: PETR4, VALE3, ITUB4",
        label_visibility="collapsed"
    ).upper().strip()
    
    # Indicador de status
    if st.session_state.ultimo_ticker_analisado == st.session_state.ticker_atual:
        st.sidebar.success(f"✅ Dados atualizados para {st.session_state.ticker_atual}")
    elif st.session_state.ticker_atual:
        st.sidebar.info(f"🔄 Atualizando dados para {st.session_state.ticker_atual}...")

    # Detectar mudança no ticker
    ticker_mudou = ticker_input != st.session_state.ticker_atual
    if ticker_mudou and ticker_input:
        st.session_state.ticker_atual = ticker_input
        st.session_state.ultimo_ticker_analisado = None  # Força nova análise
        st.rerun()

    # Botões de ações populares
    st.sidebar.markdown("**📊 Ações Populares:**")
    cols_sidebar = st.sidebar.columns(2)
    for i, acao in enumerate(Config.ACOES_POPULARES):
        col_idx = i % 2
        with cols_sidebar[col_idx]:
            if st.button(acao, key=f"btn_{acao}", width="stretch"):
                st.session_state.ticker_atual = acao
                st.session_state.ultimo_ticker_analisado = None  # Força nova análise
                st.rerun()

    # Período de análise
    periodo_analise = st.sidebar.selectbox(
        "📅 Período para análise:",
        ["1y", "2y", "5y"],
        index=0,
        help="Período histórico para cálculos de risco"
    )
    
    # Sistema avançado integrado
    if DEEP_LEARNING_DISPONIVEL:
        st.sidebar.success("🧠 Deep Learning Disponível")
        st.sidebar.info("💡 LSTM + CNN + Transfer Learning ativos automaticamente")
    elif SISTEMA_AVANCADO_DISPONIVEL:
        st.sidebar.success("🚀 Análise Avançada Disponível")
        st.sidebar.info("💡 ML + Técnicas Quantitativas ativas automaticamente")
    else:
        st.sidebar.warning("⚠️ Análise Básica")
        st.sidebar.info("💡 Instale scikit-learn para análise avançada")

    # Detectar mudança no período
    periodo_mudou = periodo_analise != st.session_state.periodo_atual
    if periodo_mudou:
        st.session_state.periodo_atual = periodo_analise
        st.session_state.ultimo_ticker_analisado = None  # Força nova análise
        st.rerun()

    # Verificar se precisa executar nova análise
    ticker_atual = st.session_state.ticker_atual
    precisa_analisar = (
        ticker_atual and 
        (st.session_state.ultimo_ticker_analisado != ticker_atual or 
         st.session_state.periodo_atual != periodo_analise)
    )

    # Botões de controle
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Atualizar", type="secondary", width="stretch"):
            st.session_state.ultimo_ticker_analisado = None
            st.rerun()
    
    with col2:
        if st.button("🗑️ Limpar Cache", type="secondary", width="stretch"):
            # Limpar cache do Streamlit
            st.cache_data.clear()
            st.session_state.ultimo_ticker_analisado = None
            st.session_state.resultado_analise = None
            st.success("Cache limpo com sucesso!")
            st.rerun()

    # Executar análise automaticamente quando necessário
    if precisa_analisar and ticker_atual:
        st.markdown(f"## 📊 Análise: {ticker_atual}")

        with st.spinner(f"🔄 Analisando {ticker_atual}... Aguarde..."):
            try:
                if DEEP_LEARNING_DISPONIVEL:
                    # Usar sistema de deep learning (mais avançado)
                    resultado = executar_analise_deep_learning(ticker_atual, periodo_analise)
                elif SISTEMA_AVANCADO_DISPONIVEL:
                    # Usar sistema avançado (fallback)
                    resultado = executar_analise_avancada(ticker_atual, periodo_analise)
                else:
                    # Usar sistema básico (fallback)
                    analisador = AnalisadorAcoes(YFinanceProvider())
                    resultado = analisador.analisar_acao(ticker_atual, periodo_analise)

                if resultado['sucesso']:
                    # Salvar resultado na sessão
                    st.session_state.resultado_analise = resultado
                    st.session_state.ultimo_ticker_analisado = ticker_atual
                    
                    if resultado.get('tipo_analise') == 'DEEP_LEARNING':
                        # Exibir análise de deep learning
                        exibir_analise_deep_learning(resultado, periodo_analise)
                    elif resultado.get('tipo_analise') == 'AVANCADA':
                        # Exibir análise avançada
                        exibir_analise_avancada(resultado, ticker_atual, periodo_analise)
                    else:
                        # Exibir análise básica
                        dados = resultado['dados_basicos']
                        indicadores = resultado['indicadores_fundamentais']
                        analise_fund = resultado['analise_fundamentalista']
                        risco = resultado['avaliacao_risco']
                        recom_final = resultado['recomendacao_final']
                        historico = resultado['historico']

                        # Exibir métricas principais
                        exibir_metricas_principais(dados, recom_final, risco)

                        # Gráfico de preços
                        st.markdown("### 📈 Evolução do Preço")
                        criar_grafico_precos(historico, ticker_atual)

                        # Tabelas de análise
                        exibir_tabelas_analise(indicadores, risco, analise_fund)

                        # Gráfico de retornos na coluna direita
                        _, col_direita = st.columns(2)
                        with col_direita:
                            criar_grafico_retornos(historico)

                        # Recomendação final
                        st.markdown("### 💡 Recomendação Final")
                        exibir_recomendacao_final(recom_final, analise_fund)

                        # Seção de backtest
                        exibir_secao_backtest(ticker_atual, periodo_analise)

                    # Detalhes expandíveis
                    with st.expander("🔧 Detalhes Técnicos da Análise"):
                        st.markdown("#### Detalhes da Análise Fundamentalista")
                        for key, value in analise_fund.get('detalhes', {}).items():
                            st.write(f"**{key.replace('_', ' ').title()}**: {value}")

                        st.markdown("#### Informações Adicionais")
                        st.write(f"**⏰ Timestamp**: {resultado['timestamp'].strftime('%d/%m/%Y %H:%M:%S')}")
                        st.write(f"**📊 Volume**: {dados.volume:,}")
                        if dados.market_cap > 0:
                            st.write(f"**💰 Market Cap**: R$ {dados.market_cap/1_000_000:.0f} milhões")
                        st.write(f"**📅 Período Analisado**: {periodo_analise}")
                        if not historico.empty:
                            st.write(f"**📈 Dados Históricos**: {len(historico)} dias")
                        
                        # Informações de validação
                        st.markdown("#### Validação de Dados")
                        st.write(f"**🔍 Confiabilidade**: {analise_fund.get('confiabilidade', 'MÉDIA')}")
                        st.write(f"**📊 Indicadores Válidos**: {analise_fund.get('indicadores_validos', 0)}/4")
                        
                        # Listar indicadores disponíveis
                        indicadores_disponiveis = []
                        if indicadores.get('pe_ratio'): indicadores_disponiveis.append('P/L')
                        if indicadores.get('pb_ratio'): indicadores_disponiveis.append('P/VP')
                        if indicadores.get('roe'): indicadores_disponiveis.append('ROE')
                        if indicadores.get('dividend_yield'): indicadores_disponiveis.append('Dividend Yield')
                        
                        st.write(f"**✅ Indicadores Disponíveis**: {', '.join(indicadores_disponiveis) if indicadores_disponiveis else 'Nenhum'}")

                else:
                    st.error(f"❌ {resultado.get('erro', 'Erro desconhecido na análise')}")
                    st.info("💡 Verifique se o ticker está correto e tente novamente.")

            except Exception as e:
                st.error(f"❌ Erro inesperado: {str(e)}")
                st.info("💡 Tente novamente ou verifique sua conexão com a internet.")

    # Exibir análise salva se disponível e não precisa de nova análise
    elif st.session_state.resultado_analise and not precisa_analisar:
        resultado = st.session_state.resultado_analise
        ticker_atual = st.session_state.ultimo_ticker_analisado
        
        st.markdown(f"## 📊 Análise: {ticker_atual}")
        
        dados = resultado['dados_basicos']
        indicadores = resultado['indicadores_fundamentais']
        analise_fund = resultado['analise_fundamentalista']
        risco = resultado['avaliacao_risco']
        recom_final = resultado['recomendacao_final']
        historico = resultado['historico']

        # Exibir métricas principais
        exibir_metricas_principais(dados, recom_final, risco)

        # Gráfico de preços
        st.markdown("### 📈 Evolução do Preço")
        criar_grafico_precos(historico, ticker_atual)

        # Tabelas de análise
        exibir_tabelas_analise(indicadores, risco, analise_fund)

        # Gráfico de retornos na coluna direita
        _, col_direita = st.columns(2)
        with col_direita:
            criar_grafico_retornos(historico)

        # Recomendação final
        st.markdown("### 💡 Recomendação Final")
        exibir_recomendacao_final(recom_final, analise_fund)

        # Detalhes expandíveis
        with st.expander("🔧 Detalhes Técnicos da Análise"):
            st.markdown("#### Detalhes da Análise Fundamentalista")
            for key, value in analise_fund.get('detalhes', {}).items():
                st.write(f"**{key.replace('_', ' ').title()}**: {value}")

            st.markdown("#### Informações Adicionais")
            st.write(f"**⏰ Timestamp**: {resultado['timestamp'].strftime('%d/%m/%Y %H:%M:%S')}")
            st.write(f"**📊 Volume**: {dados.volume:,}")
            if dados.market_cap > 0:
                st.write(f"**💰 Market Cap**: R$ {dados.market_cap/1_000_000:.0f} milhões")
            st.write(f"**📅 Período Analisado**: {st.session_state.periodo_atual}")
            if not historico.empty:
                st.write(f"**📈 Dados Históricos**: {len(historico)} dias")

    else:
        # Tela inicial
        exibir_tela_inicial()

        # Informações adicionais
        st.markdown("---")
        with st.expander("ℹ️ Sobre o Sistema"):
            st.markdown("""
            ### 🛠️ Tecnologias Utilizadas
            - **Python 3.7+** com POO (Programação Orientada a Objetos)
            - **Streamlit** para interface web interativa
            - **YFinance** para dados em tempo real do Yahoo Finance  
            - **Plotly** para gráficos interativos
            - **Pandas/NumPy** para análise de dados

            ### 📊 Metodologia
            - **Análise Fundamentalista**: Score ponderado baseado em P/L, P/VP, ROE, Dividend Yield
            - **Avaliação de Risco**: Volatilidade, Sharpe, VaR, Maximum Drawdown, Beta
            - **Matriz de Decisão**: Combina fundamentos com perfil de risco
            - **Recomendações**: 6 níveis de recomendação com justificativas

            ### ⚡ Características
            - ✅ Dados em tempo real via API
            - ✅ Cache inteligente (5 minutos)
            - ✅ Análise de ações brasileiras (B3)
            - ✅ Interface responsiva e intuitiva
            - ✅ Cálculos estatísticos avançados
            """)

# ===== EXECUÇÃO PRINCIPAL =====

if __name__ == "__main__":
    main()
