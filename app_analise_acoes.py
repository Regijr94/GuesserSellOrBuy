
import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURAÇÃO DA PÁGINA =====
st.set_page_config(
    page_title="📈 Analisador de Ações - POO",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== IMPORTAR AS CLASSES CRIADAS ANTERIORMENTE =====

@dataclass
class DadosFinanceiros:
    ticker: str
    preco_atual: float
    variacao_dia: float
    volume: int
    market_cap: float
    timestamp: datetime

@dataclass
class AvaliacaoRisco:
    ticker: str
    volatilidade: float
    sharpe_ratio: float
    beta: float
    var_95: float
    max_drawdown: float
    downside_deviation: float
    sortino_ratio: float
    cvar_95: float
    score_risco: float
    classificacao_risco: str
    comentarios: List[str]


@st.cache_data(ttl=300)
def _cached_dados_basicos(ticker: str, ticker_formatado: str) -> Optional[DadosFinanceiros]:
    try:
        stock = yf.Ticker(ticker_formatado)
        info = stock.info
        hist = stock.history(period='2d')

        if hist.empty:
            return None

        preco_atual = hist['Close'].iloc[-1]
        preco_anterior = hist['Close'].iloc[-2] if len(hist) > 1 else preco_atual
        variacao_dia = ((preco_atual - preco_anterior) / preco_anterior) * 100 if preco_anterior != 0 else 0.0

        return DadosFinanceiros(
            ticker=ticker,
            preco_atual=float(preco_atual),
            variacao_dia=float(variacao_dia),
            volume=int(hist['Volume'].iloc[-1]),
            market_cap=float(info.get('marketCap', 0) or 0),
            timestamp=datetime.now()
        )
    except Exception:
        return None


@st.cache_data(ttl=300)
def _cached_historico_precos(ticker_formatado: str, periodo: str) -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker_formatado)
        hist = stock.history(period=periodo)
        return hist
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _cached_indicadores_fundamentais(ticker_formatado: str) -> Dict:
    try:
        stock = yf.Ticker(ticker_formatado)
        info = stock.info or {}
        historico = stock.history(period='1y')

        preco_referencia = _safe_float(info.get('currentPrice')) or _safe_float(info.get('regularMarketPrice'))
        if (preco_referencia is None) and not historico.empty:
            preco_referencia = float(historico['Close'].iloc[-1])

        dividend_yield_info = _safe_float(info.get('dividendYield'))
        dividend_yield = dividend_yield_info * 100 if dividend_yield_info is not None else None

        if (dividend_yield is None or dividend_yield == 0) and preco_referencia:
            dividend_series = stock.dividends
            if dividend_series is not None and not dividend_series.empty:
                cutoff = datetime.now() - timedelta(days=365)
                dividendos_12m = dividend_series[dividend_series.index >= cutoff].sum()
                if dividendos_12m and dividendos_12m > 0:
                    dividend_yield = float(dividendos_12m / preco_referencia * 100)

        pe_ratio = _safe_float(info.get('trailingPE'))
        pb_ratio = _safe_float(info.get('priceToBook'))
        roe = _safe_float(info.get('returnOnEquity'))
        debt_to_equity = _safe_float(info.get('debtToEquity'))
        profit_margin = _safe_float(info.get('profitMargins'))
        beta = _safe_float(info.get('beta'))
        eps = _safe_float(info.get('trailingEps'))
        book_value = _safe_float(info.get('bookValue'))
        market_cap = _safe_float(info.get('marketCap'))

        return {
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'dividend_yield': dividend_yield,
            'roe': roe,
            'debt_to_equity': debt_to_equity,
            'profit_margin': profit_margin,
            'beta': beta,
            'eps': eps,
            'book_value': book_value,
            'market_cap': market_cap,
            'preco_referencia': preco_referencia
        }
    except Exception:
        return {}


def _safe_float(value: Optional[float]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class AnaliseTecnica:
    sinal: str
    tendencia: str
    score: float
    detalhes: Dict[str, float]
    comentario: str
    eventos: List[str]

class FonteDados(ABC):
    @abstractmethod
    def obter_dados_basicos(self, ticker: str) -> DadosFinanceiros:
        pass

    @abstractmethod
    def obter_historico_precos(self, ticker: str, periodo: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def obter_indicadores_fundamentais(self, ticker: str) -> Dict:
        pass

class YFinanceProvider(FonteDados):
    def _formatar_ticker_brasileiro(self, ticker: str) -> str:
        if not ticker.endswith('.SA'):
            return f"{ticker}.SA"
        return ticker

    def obter_dados_basicos(self, ticker: str) -> Optional[DadosFinanceiros]:
        ticker_formatado = self._formatar_ticker_brasileiro(ticker)
        return _cached_dados_basicos(ticker, ticker_formatado)

    def obter_historico_precos(self, ticker: str, periodo: str = '1y') -> pd.DataFrame:
        ticker_formatado = self._formatar_ticker_brasileiro(ticker)
        return _cached_historico_precos(ticker_formatado, periodo)

    def obter_indicadores_fundamentais(self, ticker: str) -> Dict:
        ticker_formatado = self._formatar_ticker_brasileiro(ticker)
        return _cached_indicadores_fundamentais(ticker_formatado)

class CalculadoraIndicadores:
    @staticmethod
    def calcular_volatilidade(precos: pd.Series, janela: int = 252) -> float:
        retornos = precos.pct_change().dropna()
        volatilidade = retornos.std() * np.sqrt(janela)
        return float(volatilidade)

    @staticmethod
    def calcular_sharpe_ratio(precos: pd.Series, taxa_livre_risco: float = 0.1375) -> float:
        retornos = precos.pct_change().dropna()
        retorno_medio = retornos.mean() * 252
        volatilidade = retornos.std() * np.sqrt(252)

        if volatilidade == 0:
            return 0.0

        sharpe = (retorno_medio - taxa_livre_risco) / volatilidade
        return float(sharpe)

    @staticmethod
    def calcular_var_95(precos: pd.Series) -> float:
        retornos = precos.pct_change().dropna()
        var_95 = np.percentile(retornos, 5)
        return float(var_95 * 100)

    @staticmethod
    def calcular_max_drawdown(precos: pd.Series) -> float:
        peak = precos.expanding(min_periods=1).max()
        drawdown = (precos / peak - 1.0)
        max_dd = drawdown.min()
        return float(max_dd * 100)

    @staticmethod
    def calcular_downside_deviation(precos: pd.Series, taxa_livre_risco: float = 0.1375) -> float:
        retornos = precos.pct_change().dropna()
        if retornos.empty:
            return 0.0
        taxa_diaria = taxa_livre_risco / 252
        retornos_excesso = retornos - taxa_diaria
        downside = retornos_excesso[retornos_excesso < 0]
        if downside.empty:
            return 0.0
        downside_deviation = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
        return float(downside_deviation)

    @staticmethod
    def calcular_sortino_ratio(precos: pd.Series, taxa_livre_risco: float = 0.1375) -> float:
        retornos = precos.pct_change().dropna()
        if retornos.empty:
            return 0.0
        retorno_medio = retornos.mean() * 252
        downside_deviation = CalculadoraIndicadores.calcular_downside_deviation(precos, taxa_livre_risco)
        if downside_deviation == 0:
            return 0.0
        sortino = (retorno_medio - taxa_livre_risco) / downside_deviation
        return float(sortino)

    @staticmethod
    def calcular_cvar_95(precos: pd.Series) -> float:
        retornos = precos.pct_change().dropna()
        if retornos.empty:
            return 0.0
        var_95 = np.percentile(retornos, 5)
        perdas_extremas = retornos[retornos <= var_95]
        if perdas_extremas.empty:
            return float(var_95 * 100)
        cvar = perdas_extremas.mean()
        return float(cvar * 100)

class EstrategiaFundamentalista:
    def __init__(self):
        self.pesos = {
            'pe_ratio': 0.25,
            'pb_ratio': 0.20,
            'dividend_yield': 0.15,
            'roe': 0.20,
            'debt_to_equity': 0.10,
            'profit_margin': 0.10
        }

    def analisar(self, dados: Dict) -> Dict:
        def disponivel(valor) -> bool:
            return valor is not None and not pd.isna(valor)

        detalhes = {}
        score_acumulado = 0.0
        peso_utilizado = 0.0

        pe = dados.get('pe_ratio')
        if disponivel(pe) and pe > 0:
            if pe < 10:
                pontos = 10
                avaliacao = 'Excelente'
            elif pe < 15:
                pontos = 8
                avaliacao = 'Bom'
            elif pe < 25:
                pontos = 5
                avaliacao = 'Razoável'
            else:
                pontos = 2
                avaliacao = 'Alto'
            score_acumulado += pontos * self.pesos['pe_ratio']
            peso_utilizado += self.pesos['pe_ratio']
            detalhes['P/L'] = f"{pe:.2f} • {avaliacao}"
        else:
            detalhes['P/L'] = 'Sem dados suficientes'

        pb = dados.get('pb_ratio')
        if disponivel(pb) and pb > 0:
            if pb < 1:
                pontos = 10
                avaliacao = 'Subvalorizada'
            elif pb < 2:
                pontos = 8
                avaliacao = 'Boa'
            elif pb < 3:
                pontos = 5
                avaliacao = 'Razoável'
            else:
                pontos = 2
                avaliacao = 'Cara'
            score_acumulado += pontos * self.pesos['pb_ratio']
            peso_utilizado += self.pesos['pb_ratio']
            detalhes['P/VP'] = f"{pb:.2f} • {avaliacao}"
        else:
            detalhes['P/VP'] = 'Sem dados suficientes'

        dividend_yield = dados.get('dividend_yield')
        if disponivel(dividend_yield):
            if dividend_yield >= 6:
                pontos = 10
                avaliacao = 'Excelente'
            elif dividend_yield >= 4:
                pontos = 8
                avaliacao = 'Atrativo'
            elif dividend_yield >= 2:
                pontos = 5
                avaliacao = 'Moderado'
            elif dividend_yield > 0:
                pontos = 3
                avaliacao = 'Baixo'
            else:
                pontos = 1
                avaliacao = 'Inexistente'
            score_acumulado += pontos * self.pesos['dividend_yield']
            peso_utilizado += self.pesos['dividend_yield']
            detalhes['Dividend Yield'] = f"{dividend_yield:.2f}% • {avaliacao}"
        else:
            detalhes['Dividend Yield'] = 'Sem dados suficientes'

        roe = dados.get('roe')
        if disponivel(roe):
            roe_percent = roe * 100
            if roe_percent > 18:
                pontos = 10
                avaliacao = 'Excelente'
            elif roe_percent > 12:
                pontos = 8
                avaliacao = 'Bom'
            elif roe_percent > 6:
                pontos = 5
                avaliacao = 'Razoável'
            else:
                pontos = 2
                avaliacao = 'Baixo'
            score_acumulado += pontos * self.pesos['roe']
            peso_utilizado += self.pesos['roe']
            detalhes['ROE'] = f"{roe_percent:.2f}% • {avaliacao}"
        else:
            detalhes['ROE'] = 'Sem dados suficientes'

        debt_to_equity = dados.get('debt_to_equity')
        if disponivel(debt_to_equity) and debt_to_equity >= 0:
            if debt_to_equity < 0.5:
                pontos = 10
                avaliacao = 'Endividamento muito baixo'
            elif debt_to_equity < 1.0:
                pontos = 8
                avaliacao = 'Estrutura saudável'
            elif debt_to_equity < 2.0:
                pontos = 5
                avaliacao = 'Endividamento moderado'
            else:
                pontos = 2
                avaliacao = 'Endividamento elevado'
            score_acumulado += pontos * self.pesos['debt_to_equity']
            peso_utilizado += self.pesos['debt_to_equity']
            detalhes['Dívida/Patrimônio'] = f"{debt_to_equity:.2f} • {avaliacao}"
        else:
            detalhes['Dívida/Patrimônio'] = 'Sem dados suficientes'

        profit_margin = dados.get('profit_margin')
        if disponivel(profit_margin):
            profit_percent = profit_margin * 100
            if profit_percent > 20:
                pontos = 10
                avaliacao = 'Margem muito alta'
            elif profit_percent > 12:
                pontos = 8
                avaliacao = 'Margem robusta'
            elif profit_percent > 6:
                pontos = 5
                avaliacao = 'Margem adequada'
            elif profit_percent > 0:
                pontos = 3
                avaliacao = 'Margem apertada'
            else:
                pontos = 1
                avaliacao = 'Margem negativa'
            score_acumulado += pontos * self.pesos['profit_margin']
            peso_utilizado += self.pesos['profit_margin']
            detalhes['Margem Líquida'] = f"{profit_percent:.2f}% • {avaliacao}"
        else:
            detalhes['Margem Líquida'] = 'Sem dados suficientes'

        score_normalizado = score_acumulado / peso_utilizado if peso_utilizado else 0

        if score_normalizado >= 8.5:
            recomendacao = 'COMPRAR'
            justificativa = 'Indicadores fundamentalistas excelentes, com múltiplos atrativos.'
        elif score_normalizado >= 6.5:
            recomendacao = 'MANTER'
            justificativa = 'Indicadores fundamentalistas sólidos, porém com alguns pontos de atenção.'
        elif score_normalizado >= 4.5:
            recomendacao = 'NEUTRO'
            justificativa = 'Indicadores mistos, equilíbrio entre prós e contras.'
        else:
            recomendacao = 'VENDER'
            justificativa = 'Indicadores fundamentalistas fracos ou deteriorando.'

        return {
            'recomendacao': recomendacao,
            'score': score_normalizado,
            'justificativa': justificativa,
            'detalhes': detalhes,
            'peso_utilizado': peso_utilizado
        }

class AvaliadorRisco:
    def __init__(self):
        self.calculadora = CalculadoraIndicadores()

    def avaliar_risco(self, ticker: str, historico_precos: pd.DataFrame, 
                     beta: Optional[float] = None) -> AvaliacaoRisco:
        precos = historico_precos['Close']

        volatilidade = self.calculadora.calcular_volatilidade(precos)
        sharpe_ratio = self.calculadora.calcular_sharpe_ratio(precos)
        sortino_ratio = self.calculadora.calcular_sortino_ratio(precos)
        var_95 = self.calculadora.calcular_var_95(precos)
        cvar_95 = self.calculadora.calcular_cvar_95(precos)
        max_drawdown = self.calculadora.calcular_max_drawdown(precos)
        downside_deviation = self.calculadora.calcular_downside_deviation(precos)
        beta_final = beta if beta is not None else 1.0

        comentarios: List[str] = []
        score_risco = 0.0

        # Volatilidade anualizada (em decimal)
        if volatilidade < 0.18:
            score_risco += 25
            comentarios.append("Volatilidade controlada no período analisado.")
        elif volatilidade < 0.28:
            score_risco += 18
            comentarios.append("Volatilidade moderada, atenção a movimentos bruscos.")
        elif volatilidade < 0.38:
            score_risco += 10
            comentarios.append("Volatilidade elevada indica risco maior.")
        else:
            score_risco += 5
            comentarios.append("Volatilidade muito alta, risco significativo.")

        # Sharpe ratio
        if sharpe_ratio > 1.0:
            score_risco += 18
            comentarios.append("Índice Sharpe acima de 1, retorno ajustado ao risco positivo.")
        elif sharpe_ratio > 0.4:
            score_risco += 12
            comentarios.append("Índice Sharpe moderado, retorno ajustado ao risco aceitável.")
        elif sharpe_ratio > 0:
            score_risco += 8
            comentarios.append("Índice Sharpe baixo, retorno ajustado ao risco limitado.")
        else:
            score_risco += 4
            comentarios.append("Índice Sharpe negativo, risco não compensado pelo retorno.")

        # Sortino ratio (foco em quedas)
        if sortino_ratio > 1.2:
            score_risco += 12
            comentarios.append("Sortino elevado aponta boas recompensas para o risco negativo assumido.")
        elif sortino_ratio > 0.5:
            score_risco += 8
            comentarios.append("Sortino moderado, quedas controladas.")
        elif sortino_ratio > 0:
            score_risco += 5
            comentarios.append("Sortino baixo, retornos não compensam quedas.")
        else:
            score_risco += 2
            comentarios.append("Sortino negativo, quedas pesam mais que ganhos.")

        # Drawdown
        if abs(max_drawdown) < 12:
            score_risco += 12
            comentarios.append("Drawdown limitado, proteção relativa contra quedas prolongadas.")
        elif abs(max_drawdown) < 22:
            score_risco += 8
            comentarios.append("Drawdown moderado, quedas relevantes observadas.")
        elif abs(max_drawdown) < 32:
            score_risco += 5
            comentarios.append("Drawdown elevado, quedas expressivas.")
        else:
            score_risco += 2
            comentarios.append("Drawdown extremo, atenção ao risco de perdas profundas.")

        # CVaR 95% (esperado negativo)
        if cvar_95 > -2:
            score_risco += 11
            comentarios.append("CVaR controlado, perdas extremas limitadas.")
        elif cvar_95 > -4:
            score_risco += 8
            comentarios.append("CVaR moderado, perdas extremas na média.")
        elif cvar_95 > -6:
            score_risco += 5
            comentarios.append("CVaR elevado, perdas extremas relevantes.")
        else:
            score_risco += 2
            comentarios.append("CVaR crítico, quedas extremas muito agressivas.")

        # Beta (sensibilidade ao mercado)
        if abs(beta_final - 1) < 0.15:
            score_risco += 12
            comentarios.append("Beta próximo de 1, risco alinhado ao mercado.")
        elif abs(beta_final - 1) < 0.4:
            score_risco += 9
            comentarios.append("Beta moderado, leve descasamento com o mercado.")
        else:
            score_risco += 5
            comentarios.append("Beta distante de 1, risco de mercado diferenciado.")

        if score_risco >= 75:
            classificacao = "BAIXO"
        elif score_risco >= 55:
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
            downside_deviation=downside_deviation,
            sortino_ratio=sortino_ratio,
            cvar_95=cvar_95,
            score_risco=score_risco,
            classificacao_risco=classificacao,
            comentarios=comentarios
        )

class AnaliseTecnicaIndicadores:
    def __init__(self):
        self.periodos_sma = {
            'curto': 21,
            'medio': 50,
            'longo': 200
        }
        self.periodos_wma = {
            'curto': 21,
            'longo': 89
        }

    def avaliar(self, historico_precos: pd.DataFrame) -> AnaliseTecnica:
        if historico_precos.empty or len(historico_precos) < self.periodos_sma['longo']:
            return AnaliseTecnica(
                sinal='NEUTRO',
                tendencia='Indefinida',
                score=50.0,
                detalhes={},
                comentario='Histórico insuficiente para análise técnica confiável.',
                eventos=[]
            )

        df = historico_precos.copy()
        close = df['Close']

        sma_curto = ta.sma(close, length=self.periodos_sma['curto'])
        sma_medio = ta.sma(close, length=self.periodos_sma['medio'])
        sma_longo = ta.sma(close, length=self.periodos_sma['longo'])
        ema_9 = ta.ema(close, length=9)
        rsi_14 = ta.rsi(close, length=14)
        macd_df = ta.macd(close)
        adx_df = ta.adx(df['High'], df['Low'], df['Close'])
        wma_curto = ta.wma(close, length=self.periodos_wma['curto'])
        wma_longo = ta.wma(close, length=self.periodos_wma['longo'])

        valor_atual = float(close.iloc[-1])
        sma_curto_val = float(sma_curto.iloc[-1])
        sma_medio_val = float(sma_medio.iloc[-1])
        sma_longo_val = float(sma_longo.iloc[-1])
        ema_9_val = float(ema_9.iloc[-1])
        rsi_val = float(rsi_14.iloc[-1]) if not np.isnan(rsi_14.iloc[-1]) else 50.0
        macd_val = float(macd_df['MACD_12_26_9'].iloc[-1])
        macd_signal_val = float(macd_df['MACDs_12_26_9'].iloc[-1])
        macd_hist_val = float(macd_df['MACDh_12_26_9'].iloc[-1])
        adx_val = float(adx_df['ADX_14'].iloc[-1]) if not np.isnan(adx_df['ADX_14'].iloc[-1]) else 15.0

        wma_curto_valid = wma_curto.dropna()
        wma_longo_valid = wma_longo.dropna()
        wma_curto_val = float(wma_curto_valid.iloc[-1]) if not wma_curto_valid.empty else float('nan')
        wma_longo_val = float(wma_longo_valid.iloc[-1]) if not wma_longo_valid.empty else float('nan')
        wma_diff_series = (wma_curto - wma_longo).dropna()
        wma_diff = float(wma_diff_series.iloc[-1]) if not wma_diff_series.empty else wma_curto_val - wma_longo_val
        wma_prev_diff = float(wma_diff_series.iloc[-2]) if len(wma_diff_series) > 1 else wma_diff

        score = 50.0
        observacoes = []
        eventos = []

        if valor_atual > sma_curto_val:
            score += 8
            observacoes.append('Preço acima da média curta, momentum positivo.')
        else:
            score -= 8
            observacoes.append('Preço abaixo da média curta, pressão vendedora no curto prazo.')

        if sma_medio_val > sma_longo_val:
            score += 12
            observacoes.append('SMA50 acima da SMA200 sinaliza tendência principal de alta.')
        else:
            score -= 12
            observacoes.append('SMA50 abaixo da SMA200 sugere tendência principal de baixa.')

        if ema_9_val > sma_curto_val:
            score += 6
            observacoes.append('EMA9 sustentada acima da média curta reforça compra.')
        else:
            score -= 6
            observacoes.append('EMA9 abaixo da média curta indica perda de fôlego de alta.')

        if macd_hist_val > 0 and macd_val > macd_signal_val:
            score += 10
            observacoes.append('MACD positivo reforça força compradora.')
        elif macd_hist_val < 0 and macd_val < macd_signal_val:
            score -= 10
            observacoes.append('MACD negativo evidencia força vendedora.')
        else:
            observacoes.append('MACD neutro, sem direção clara.')

        if rsi_val < 30:
            score += 12
            observacoes.append('RSI em sobrevenda sugere oportunidade de reversão.')
        elif rsi_val < 40:
            score += 8
            observacoes.append('RSI baixo com chance de recuperação.')
        elif rsi_val > 70:
            score -= 12
            observacoes.append('RSI em sobrecompra sinaliza possível correção.')
        elif rsi_val > 60:
            score -= 6
            observacoes.append('RSI elevado, atenção para realização de lucros.')
        else:
            score += 4
            observacoes.append('RSI equilibrado, mercado saudável.')

        if adx_val > 25:
            score += 5
            observacoes.append('ADX alto confirma força da tendência.')
        elif adx_val < 15:
            score -= 5
            observacoes.append('ADX baixo indica mercado lateral.')
        else:
            observacoes.append('ADX neutro, tendência moderada.')

        if not np.isnan(wma_curto_val) and not np.isnan(wma_longo_val):
            if wma_diff > 0:
                score += 8
                observacoes.append('WMA curta acima da longa indica viés altista no médio prazo.')
            else:
                score -= 8
                observacoes.append('WMA curta abaixo da longa reforça viés baixista no médio prazo.')

            if wma_prev_diff <= 0 < wma_diff:
                eventos.append('Cruzamento altista recente das WMAs (curta cruzou acima da longa).')
                score += 6
            elif wma_prev_diff >= 0 > wma_diff:
                eventos.append('Cruzamento baixista recente das WMAs (curta cruzou abaixo da longa).')
                score -= 6
            elif abs(wma_diff) < (0.002 * valor_atual):
                observacoes.append('WMAs muito próximas: atenção para possível mudança de tendência.')

        score = max(0, min(100, score))

        if score >= 65:
            sinal = 'COMPRA'
        elif score <= 40:
            sinal = 'VENDA'
        else:
            sinal = 'MANTER'

        if valor_atual > sma_longo_val and sma_medio_val > sma_longo_val:
            tendencia = 'Alta'
        elif valor_atual < sma_longo_val and sma_medio_val < sma_longo_val:
            tendencia = 'Baixa'
        else:
            tendencia = 'Lateral'

        lista_eventos = eventos + observacoes
        comentario_resumo = lista_eventos[0] if lista_eventos else 'Análise técnica inconclusiva.'

        detalhes = {
            'preco_atual': round(valor_atual, 2),
            'sma_21': round(sma_curto_val, 2),
            'sma_50': round(sma_medio_val, 2),
            'sma_200': round(sma_longo_val, 2),
            'ema_9': round(ema_9_val, 2),
            'rsi_14': round(rsi_val, 2),
            'macd': round(macd_val, 4),
            'macd_signal': round(macd_signal_val, 4),
            'macd_hist': round(macd_hist_val, 4),
            'adx_14': round(adx_val, 2),
            'wma_curta': round(wma_curto_val, 2) if not np.isnan(wma_curto_val) else None,
            'wma_longa': round(wma_longo_val, 2) if not np.isnan(wma_longo_val) else None,
            'wma_diff': round(wma_diff, 4) if not np.isnan(wma_diff) else None
        }

        return AnaliseTecnica(
            sinal=sinal,
            tendencia=tendencia,
            score=score,
            detalhes=detalhes,
            comentario=comentario_resumo,
            eventos=lista_eventos
        )

class AnalisadorAcoes:
    def __init__(self, fonte_dados: FonteDados):
        self.fonte_dados = fonte_dados
        self.estrategia = EstrategiaFundamentalista()
        self.avaliador_risco = AvaliadorRisco()
        self.analise_tecnica = AnaliseTecnicaIndicadores()

    def analisar_acao(self, ticker: str) -> Dict:
        resultado = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'sucesso': False
        }

        try:
            dados_basicos = self.fonte_dados.obter_dados_basicos(ticker)
            if not dados_basicos:
                resultado['erro'] = 'Não foi possível obter dados básicos'
                return resultado

            indicadores = self.fonte_dados.obter_indicadores_fundamentais(ticker)
            if not indicadores:
                resultado['erro'] = 'Não foi possível obter indicadores fundamentalistas'
                return resultado

            historico = self.fonte_dados.obter_historico_precos(ticker, '1y')
            if historico.empty:
                resultado['erro'] = 'Não foi possível obter histórico de preços'
                return resultado

            analise_fund = self.estrategia.analisar(indicadores)
            beta = indicadores.get('beta')
            avaliacao_risco = self.avaliador_risco.avaliar_risco(ticker, historico, beta)
            analise_tecnica = self.analise_tecnica.avaliar(historico)

            resultado.update({
                'sucesso': True,
                'dados_basicos': dados_basicos,
                'indicadores_fundamentais': indicadores,
                'analise_fundamentalista': analise_fund,
                'avaliacao_risco': avaliacao_risco,
                'analise_tecnica': analise_tecnica,
                'historico': historico,
                'recomendacao_final': self._recomendacao_final(analise_fund, avaliacao_risco, analise_tecnica)
            })

        except Exception as e:
            resultado['erro'] = f'Erro na análise: {str(e)}'

        return resultado

    def _recomendacao_final(self, analise_fund: Dict, avaliacao_risco: AvaliacaoRisco, analise_tecnica: AnaliseTecnica) -> Dict:
        recom_fund = analise_fund['recomendacao']
        sinal_tecnico = analise_tecnica.sinal
        risco_classificacao = avaliacao_risco.classificacao_risco

        mapa_fundamental = {
            'COMPRAR': 1.0,
            'MANTER': 0.2,
            'NEUTRO': 0.0,
            'VENDER': -1.0
        }
        mapa_tecnico = {
            'COMPRA': 1.0,
            'MANTER': 0.0,
            'NEUTRO': 0.0,
            'VENDA': -1.0
        }
        pesos = {
            'fundamental': 0.45,
            'tecnico': 0.35,
            'risco': 0.20
        }

        componente_fund = mapa_fundamental.get(recom_fund, 0.0)
        componente_tecnico = mapa_tecnico.get(sinal_tecnico, 0.0)
        risco_norm = np.clip(avaliacao_risco.score_risco / 100, 0, 1)
        componente_risco = (risco_norm * 2) - 1  # converte para intervalo [-1, 1]

        score_composto = (
            componente_fund * pesos['fundamental'] +
            componente_tecnico * pesos['tecnico'] +
            componente_risco * pesos['risco']
        )
        score_composto = float(np.clip(score_composto, -1, 1))

        if score_composto >= 0.6:
            recomendacao_final = 'COMPRA FORTE'
        elif score_composto >= 0.3:
            recomendacao_final = 'COMPRAR'
        elif score_composto >= -0.1:
            recomendacao_final = 'MANTER'
        elif score_composto >= -0.4:
            recomendacao_final = 'REDUZIR POSIÇÃO'
        else:
            recomendacao_final = 'VENDA FORTE'

        intensidade = abs(score_composto)
        if intensidade >= 0.7:
            confianca = 'ALTA'
        elif intensidade >= 0.4:
            confianca = 'MÉDIA'
        else:
            confianca = 'BAIXA'

        justificativa = " | ".join([
            f"Fundamental: {recom_fund} ({analise_fund['justificativa']})",
            f"Técnico: {sinal_tecnico} ({analise_tecnica.comentario})",
            f"Risco: {risco_classificacao} (score {avaliacao_risco.score_risco:.1f})"
        ])

        return {
            'recomendacao': recomendacao_final,
            'justificativa': justificativa,
            'confianca': confianca,
            'score_composto': score_composto,
            'componentes': {
                'fundamental': componente_fund,
                'tecnico': componente_tecnico,
                'risco': componente_risco
            },
            'pesos': pesos
        }

# ===== INTERFACE STREAMLIT =====

def main():
    st.title("📈 Sistema de Análise Fundamentalista de Ações")
    st.markdown("### Análise baseada em POO e consumo de APIs")

    # Sidebar
    st.sidebar.title("🔧 Configurações")
    st.sidebar.markdown("---")

    # Seleção de ações
    acoes_populares = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3', 'WEGE3', 'MGLU3', 'RENT3', 'GGBR4', 'USIM5']

    ticker_input = st.sidebar.text_input("Digite o ticker da ação:", value="PETR4").upper()

    st.sidebar.markdown("**Ações populares:**")
    for acao in acoes_populares:
        if st.sidebar.button(acao, key=f"btn_{acao}"):
            ticker_input = acao

    periodo_analise = st.sidebar.selectbox(
        "Período para análise:",
        ["1y", "2y", "5y"],
        index=0
    )

    # Botão de análise
    if st.sidebar.button("🚀 Analisar Ação", type="primary"):
        if ticker_input:
            st.session_state.ticker_atual = ticker_input
            st.session_state.periodo_atual = periodo_analise

    # Análise principal
    if hasattr(st.session_state, 'ticker_atual'):
        ticker = st.session_state.ticker_atual
        periodo = st.session_state.get('periodo_atual', '1y')

        with st.spinner(f"Analisando {ticker}..."):
            analisador = AnalisadorAcoes(YFinanceProvider())
            resultado = analisador.analisar_acao(ticker)

        if resultado['sucesso']:
            exibir_resultado_completo(resultado, periodo)
        else:
            st.error(f"❌ Erro na análise: {resultado.get('erro', 'Erro desconhecido')}")
    else:
        exibir_tela_inicial()

def exibir_tela_inicial():
    st.markdown("""
    ## 🎯 Bem-vindo ao Analisador de Ações

    Este sistema utiliza **Programação Orientada a Objetos (POO)** para analisar ações brasileiras através de:

    ### 🔍 **Análise Fundamentalista**
    - **P/L (Preço/Lucro)**: Avalia se a ação está cara ou barata
    - **P/VP (Preço/Valor Patrimonial)**: Compara preço com valor contábil
    - **ROE (Return on Equity)**: Mede a eficiência da empresa
    - **Dividend Yield**: Retorno em dividendos

    ### ⚡ **Avaliação de Risco**
    - **Volatilidade**: Medida de instabilidade dos preços
    - **Índice Sharpe**: Relação risco-retorno
    - **VaR (Value at Risk)**: Perda máxima esperada
    - **Beta**: Correlação com o mercado

    ### 🤖 **Tecnologias Utilizadas**
    - **Python com POO**: Arquitetura modular e extensível
    - **YFinance API**: Dados em tempo real do Yahoo Finance
    - **Streamlit**: Interface web interativa
    - **Plotly**: Gráficos interativos

    **👈 Selecione uma ação na barra lateral para começar!**
    """)

    # Exemplo de análise com múltiplas ações
    st.markdown("### 📊 Comparação Rápida - Principais Ações")

    if st.button("🔄 Atualizar Comparação"):
        acoes_exemplo = ['PETR4', 'VALE3', 'ITUB4']
        comparacao_data = []

        progress_bar = st.progress(0)
        for i, ticker in enumerate(acoes_exemplo):
            analisador = AnalisadorAcoes(YFinanceProvider())
            resultado = analisador.analisar_acao(ticker)

            if resultado['sucesso']:
                dados = resultado['dados_basicos']
                recom = resultado['recomendacao_final']
                risco = resultado['avaliacao_risco']
                analise_tecnica = resultado.get('analise_tecnica')

                comparacao_data.append({
                    'Ação': ticker,
                    'Preço': f"R$ {dados.preco_atual:.2f}",
                    'Variação Dia': f"{dados.variacao_dia:.2f}%",
                    'Recomendação': recom['recomendacao'],
                    'Risco': risco.classificacao_risco,
                    'Confiança': recom['confianca'],
                    'Sinal Técnico': analise_tecnica.sinal if analise_tecnica else '-'
                })

            progress_bar.progress((i + 1) / len(acoes_exemplo))

        if comparacao_data:
            df_comparacao = pd.DataFrame(comparacao_data)
            st.dataframe(df_comparacao, width='stretch', hide_index=True)

def exibir_resultado_completo(resultado, periodo):
    ticker = resultado['ticker']
    dados = resultado['dados_basicos']
    analise_fund = resultado['analise_fundamentalista']
    risco = resultado['avaliacao_risco']
    analise_tecnica = resultado['analise_tecnica']
    recom_final = resultado['recomendacao_final']
    historico = resultado['historico']

    # Header com informações principais
    st.markdown(f"# 📊 Análise Completa: {ticker}")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Preço Atual",
            f"R$ {dados.preco_atual:.2f}",
            f"{dados.variacao_dia:.2f}%"
        )

    with col2:
        cor_recom = "🟢" if "COMPRA" in recom_final['recomendacao'] else "🔴" if "VEND" in recom_final['recomendacao'] else "🟡"
        st.metric(
            "Recomendação",
            f"{cor_recom} {recom_final['recomendacao']}"
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

    with col5:
        icone_tecnico = "🟢" if analise_tecnica.sinal == 'COMPRA' else "🔴" if analise_tecnica.sinal == 'VENDA' else "🟡"
        st.metric(
            "Sinal Técnico",
            f"{icone_tecnico} {analise_tecnica.sinal}",
            f"Score {analise_tecnica.score:.0f}/100"
        )

    # Gráfico de preços
    st.markdown("### 📈 Evolução do Preço")

    fig_preco = go.Figure()
    fig_preco.add_trace(go.Scatter(
        x=historico.index,
        y=historico['Close'],
        mode='lines',
        name='Preço de Fechamento',
        line=dict(color='blue', width=2)
    ))

    fig_preco.update_layout(
        title=f'Histórico de Preços - {ticker} ({periodo})',
        xaxis_title='Data',
        yaxis_title='Preço (R$)',
        height=400
    )

    st.plotly_chart(fig_preco, width='stretch')

    # Análise detalhada em colunas
    col_esquerda, col_direita = st.columns(2)

    with col_esquerda:
        st.markdown("### 🔍 Indicadores Fundamentalistas")

        detalhes_fund = analise_fund.get('detalhes', {})
        if detalhes_fund:
            linhas = []
            for nome, texto in detalhes_fund.items():
                if '•' in texto:
                    valor_str, avaliacao = [parte.strip() for parte in texto.split('•', 1)]
                else:
                    valor_str, avaliacao = texto, ''
                linhas.append({'Indicador': nome, 'Valor': valor_str, 'Avaliação': avaliacao})
            df_indicadores = pd.DataFrame(linhas)
            st.dataframe(df_indicadores, width='stretch', hide_index=True)

        st.markdown("### 📊 Score Fundamentalista")
        score_normalizado = min(analise_fund['score'], 10) / 10 * 100

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score_normalizado,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Score (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, width='stretch')

    with col_direita:
        st.markdown("### ⚠️ Análise de Risco")

        risco_data = [
            ['Volatilidade Anual', f"{risco.volatilidade:.2f}"],
            ['Desvio Padrão (Quedas)', f"{risco.downside_deviation:.2f}"],
            ['Índice Sharpe', f"{risco.sharpe_ratio:.2f}"],
            ['Índice Sortino', f"{risco.sortino_ratio:.2f}"],
            ['VaR 95%', f"{risco.var_95:.2f}%"],
            ['CVaR 95%', f"{risco.cvar_95:.2f}%"],
            ['Max Drawdown', f"{risco.max_drawdown:.2f}%"],
            ['Beta', f"{risco.beta:.2f}"],
            ['Score de Risco', f"{risco.score_risco:.1f}"]
        ]

        df_risco = pd.DataFrame(risco_data, columns=['Métrica', 'Valor'])
        st.dataframe(df_risco, width='stretch', hide_index=True)

        st.markdown("#### Comentários de Risco")
        for comentario in risco.comentarios[:4]:
            st.markdown(f"- {comentario}")

        st.markdown("### 📊 Distribuição dos Retornos")
        retornos = historico['Close'].pct_change().dropna() * 100

        fig_hist = px.histogram(
            retornos, 
            nbins=30, 
            title='Distribuição dos Retornos Diários (%)',
            labels={'value': 'Retorno Diário (%)', 'count': 'Frequência'}
        )
        fig_hist.update_layout(height=300)
        st.plotly_chart(fig_hist, width='stretch')

        st.markdown("### 📈 Análise Técnica de Tendência")
        dados_tecnicos = pd.DataFrame(
            [(chave, valor) for chave, valor in analise_tecnica.detalhes.items() if valor is not None],
            columns=['Indicador', 'Valor']
        )
        st.dataframe(dados_tecnicos, width='stretch', hide_index=True)
        st.markdown(f"**Resumo Técnico:** {analise_tecnica.comentario}")
        if analise_tecnica.eventos:
            st.markdown("**Eventos Recentes:**")
            for evento in analise_tecnica.eventos[:5]:
                st.markdown(f"- {evento}")

    # Justificativa e recomendação
    st.markdown("### 💡 Justificativa da Recomendação")

    justificativas_lista = recom_final['justificativa'].split(' | ')
    justificativas_html = ''.join(f"<li>{item}</li>" for item in justificativas_lista)

    if "COMPRA" in recom_final['recomendacao']:
        cor_fundo = "#d4edda"
        cor_texto = "#155724"
    elif "VEND" in recom_final['recomendacao']:
        cor_fundo = "#f8d7da"
        cor_texto = "#721c24"
    else:
        cor_fundo = "#fff3cd"
        cor_texto = "#856404"

    st.markdown(f"""
    <div style="
        background-color: {cor_fundo};
        color: {cor_texto};
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {cor_texto};
        margin: 10px 0;
    ">
        <h4>🎯 {recom_final['recomendacao']}</h4>
        <p><strong>Nível de Confiança:</strong> {recom_final['confianca']}</p>
        <p><strong>Score Fundamentalista:</strong> {analise_fund['score']:.2f}/10</p>
        <p><strong>Score Técnico:</strong> {analise_tecnica.score:.0f}/100</p>
        <p><strong>Score Composto:</strong> {recom_final['score_composto']:.2f}</p>
        <p><strong>Justificativas:</strong></p>
        <ul>{justificativas_html}</ul>
    </div>
    """, unsafe_allow_html=True)

    # Detalhes técnicos (expandível)
    with st.expander("🔧 Detalhes Técnicos da Análise"):
        st.markdown("#### Análise Fundamentalista Detalhada")
        for key, value in analise_fund.get('detalhes', {}).items():
            st.write(f"**{key}**: {value}")

        st.markdown("#### Indicadores Técnicos")
        st.write(f"**Sinal Técnico:** {analise_tecnica.sinal} ({analise_tecnica.score:.0f}/100)")
        st.write(f"**Resumo Técnico:** {analise_tecnica.comentario}")
        if analise_tecnica.eventos:
            st.markdown("**Eventos e Observações:**")
            for evento in analise_tecnica.eventos:
                st.write(f"- {evento}")
        for chave, valor in analise_tecnica.detalhes.items():
            st.write(f"- {chave}: {valor}")

        st.markdown("#### Comentários de Risco")
        for comentario in risco.comentarios:
            st.write(f"- {comentario}")

        st.markdown("#### Informações Técnicas")
        st.write(f"**Timestamp da Análise**: {resultado['timestamp']}")
        st.write(f"**Volume Médio**: {dados.volume:,}")
        if dados.market_cap > 0:
            st.write(f"**Market Cap**: R$ {dados.market_cap/1000000:.0f} milhões")

if __name__ == "__main__":
    main()
