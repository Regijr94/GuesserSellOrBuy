# Criando versão unificada e simplificada do sistema
sistema_unificado = '''
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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

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
    
    def _formatar_ticker_brasileiro(self, ticker: str) -> str:
        """Adiciona .SA para ações brasileiras"""
        if not ticker.endswith('.SA'):
            return f"{ticker}.SA"
        return ticker
    
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
        """Obtém indicadores fundamentalistas"""
        try:
            ticker_formatado = _self._formatar_ticker_brasileiro(ticker)
            stock = yf.Ticker(ticker_formatado)
            info = stock.info
            
            return {
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'roe': info.get('returnOnEquity'),
                'debt_to_equity': info.get('debtToEquity'),
                'profit_margin': info.get('profitMargins'),
                'beta': info.get('beta'),
                'eps': info.get('trailingEps'),
                'book_value': info.get('bookValue'),
                'market_cap': info.get('marketCap')
            }
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
        """Analisa indicadores e retorna score e recomendação"""
        score = 0
        detalhes = {}
        
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
        
        # Determinação da recomendação
        if score >= Config.SCORE_COMPRA:
            recomendacao = 'COMPRAR'
            justificativa = 'Indicadores fundamentalistas muito favoráveis'
        elif score >= Config.SCORE_MANUTENCAO:
            recomendacao = 'MANTER'
            justificativa = 'Indicadores fundamentalistas favoráveis'
        elif score >= Config.SCORE_NEUTRO:
            recomendacao = 'NEUTRO'
            justificativa = 'Indicadores fundamentalistas neutros'
        else:
            recomendacao = 'VENDER'
            justificativa = 'Indicadores fundamentalistas desfavoráveis'
        
        return {
            'recomendacao': recomendacao,
            'score': score,
            'justificativa': justificativa,
            'detalhes': detalhes
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
    
    **👈 Selecione uma ação na barra lateral para começar!**
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
    </div>
    """, unsafe_allow_html=True)

def main():
    """Função principal da aplicação"""
    
    st.title("📈 Sistema de Análise Fundamentalista de Ações")
    st.markdown("### 🤖 Análise baseada em POO e APIs em tempo real")
    
    # Sidebar
    st.sidebar.title("🔧 Configurações")
    st.sidebar.markdown("---")
    
    # Input da ação
    ticker_input = st.sidebar.text_input(
        "Digite o ticker da ação:", 
        value="PETR4",
        help="Digite apenas o código da ação (ex: PETR4, VALE3)"
    ).upper().strip()
    
    # Botões de ações populares
    st.sidebar.markdown("**📊 Ações Populares:**")
    cols_sidebar = st.sidebar.columns(2)
    for i, acao in enumerate(Config.ACOES_POPULARES):
        col_idx = i % 2
        with cols_sidebar[col_idx]:
            if st.button(acao, key=f"btn_{acao}", use_container_width=True):
                st.session_state.ticker_selecionado = acao
    
    # Período de análise
    periodo_analise = st.sidebar.selectbox(
        "📅 Período para análise:",
        ["1y", "2y", "5y"],
        index=0,
        help="Período histórico para cálculos de risco"
    )
    
    # Verificar ticker selecionado
    ticker_atual = getattr(st.session_state, 'ticker_selecionado', ticker_input)
    if not ticker_atual:
        ticker_atual = ticker_input
    
    # Botão de análise
    if st.sidebar.button("🚀 Analisar Ação", type="primary", use_container_width=True):
        if ticker_atual:
            st.session_state.executar_analise = True
            st.session_state.ticker_para_analise = ticker_atual
            st.session_state.periodo_para_analise = periodo_analise
    
    # Executar análise se solicitado
    if getattr(st.session_state, 'executar_analise', False):
        ticker = st.session_state.ticker_para_analise
        periodo = st.session_state.periodo_para_analise
        
        st.markdown(f"## 📊 Análise: {ticker}")
        
        with st.spinner(f"🔄 Analisando {ticker}... Aguarde..."):
            try:
                # Criar analisador e executar análise
                analisador = AnalisadorAcoes(YFinanceProvider())
                resultado = analisador.analisar_acao(ticker, periodo)
                
                if resultado['sucesso']:
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
                    criar_grafico_precos(historico, ticker)
                    
                    # Tabelas de análise
                    exibir_tabelas_analise(indicadores, risco, analise_fund)
                    
                    # Gráfico de retornos na coluna direita
                    col_esquerda, col_direita = st.columns(2)
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
                        st.write(f"**📅 Período Analisado**: {periodo}")
                        if not historico.empty:
                            st.write(f"**📈 Dados Históricos**: {len(historico)} dias")
                
                else:
                    st.error(f"❌ {resultado.get('erro', 'Erro desconhecido na análise')}")
                    st.info("💡 Verifique se o ticker está correto e tente novamente.")
            
            except Exception as e:
                st.error(f"❌ Erro inesperado: {str(e)}")
                st.info("💡 Tente novamente ou verifique sua conexão com a internet.")
        
        # Resetar flag de análise
        st.session_state.executar_analise = False
    
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
'''

# Salvar arquivo unificado
with open('analisador_acoes_completo.py', 'w', encoding='utf-8') as f:
    f.write(sistema_unificado)

print("✅ Sistema unificado criado!")
print("📝 Arquivo: analisador_acoes_completo.py")