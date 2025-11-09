#!/usr/bin/env python3
"""
Sistema de Deep Learning para Análise de Ações
==============================================

Este módulo implementa técnicas avançadas de deep learning para análise
de ações, incluindo LSTM, CNN, ensemble methods e backtest histórico
para treinamento das redes neurais.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import MeanSquaredError, Huber
    from tensorflow.keras.metrics import MeanAbsoluteError
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# PyTorch não é necessário para este sistema
PYTORCH_AVAILABLE = False

# Technical Analysis - usando implementações básicas
TALIB_AVAILABLE = False  # Sempre usar implementações básicas

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class BacktestHistorico:
    """Sistema de backtest histórico para treinamento de redes neurais"""
    
    def __init__(self, ticker: str, periodo_inicio: str = "2015-01-01", periodo_fim: str = None):
        self.ticker = ticker
        self.periodo_inicio = periodo_inicio
        self.periodo_fim = periodo_fim or datetime.now().strftime("%Y-%m-%d")
        self.dados_historicos = None
        self.features = None
        self.targets = None
        
    def obter_dados_historicos(self) -> pd.DataFrame:
        """Obtém dados históricos completos para backtest"""
        try:
            # Obter dados históricos
            acao = yf.Ticker(f"{self.ticker}.SA")
            dados = acao.history(
                start=self.periodo_inicio,
                end=self.periodo_fim,
                interval="1d"
            )
            
            if dados.empty:
                raise ValueError(f"Nenhum dado encontrado para {self.ticker}")
            
            # Adicionar features técnicas
            dados = self._adicionar_features_tecnicas(dados)
            
            # Adicionar features fundamentais (simuladas para backtest)
            dados = self._adicionar_features_fundamentais(dados)
            
            # Adicionar targets (retornos futuros)
            dados = self._adicionar_targets(dados)
            
            self.dados_historicos = dados
            return dados
            
        except Exception as e:
            raise ValueError(f"Erro ao obter dados históricos: {str(e)}")
    
    def _adicionar_features_tecnicas(self, dados: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features técnicas usando TA-Lib"""
        if not TALIB_AVAILABLE:
            # Implementação básica sem TA-Lib
            dados['SMA_20'] = dados['Close'].rolling(20).mean()
            dados['SMA_50'] = dados['Close'].rolling(50).mean()
            dados['EMA_12'] = dados['Close'].ewm(span=12).mean()
            dados['EMA_26'] = dados['Close'].ewm(span=26).mean()
            dados['RSI'] = self._calcular_rsi_basico(dados['Close'])
            dados['MACD'] = dados['EMA_12'] - dados['EMA_26']
            dados['BB_upper'], dados['BB_middle'], dados['BB_lower'] = self._calcular_bollinger_basico(dados['Close'])
            dados['ATR'] = self._calcular_atr_basico(dados)
            dados['Volume_SMA'] = dados['Volume'].rolling(20).mean()
            dados['Price_Change'] = dados['Close'].pct_change()
            dados['Volatility'] = dados['Price_Change'].rolling(20).std()
        else:
            # Implementação com TA-Lib
            dados['SMA_20'] = talib.SMA(dados['Close'], timeperiod=20)
            dados['SMA_50'] = talib.SMA(dados['Close'], timeperiod=50)
            dados['EMA_12'] = talib.EMA(dados['Close'], timeperiod=12)
            dados['EMA_26'] = talib.EMA(dados['Close'], timeperiod=26)
            dados['RSI'] = talib.RSI(dados['Close'], timeperiod=14)
            dados['MACD'], dados['MACD_signal'], dados['MACD_hist'] = talib.MACD(dados['Close'])
            dados['BB_upper'], dados['BB_middle'], dados['BB_lower'] = talib.BBANDS(dados['Close'])
            dados['ATR'] = talib.ATR(dados['High'], dados['Low'], dados['Close'], timeperiod=14)
            dados['Volume_SMA'] = talib.SMA(dados['Volume'], timeperiod=20)
            dados['Price_Change'] = dados['Close'].pct_change()
            dados['Volatility'] = dados['Price_Change'].rolling(20).std()
        
        return dados
    
    def _calcular_rsi_basico(self, preços: pd.Series, periodo: int = 14) -> pd.Series:
        """Calcula RSI sem TA-Lib"""
        delta = preços.diff()
        ganho = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        perda = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = ganho / perda
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calcular_bollinger_basico(self, preços: pd.Series, periodo: int = 20, desvio: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula Bandas de Bollinger sem TA-Lib"""
        sma = preços.rolling(periodo).mean()
        std = preços.rolling(periodo).std()
        upper = sma + (std * desvio)
        lower = sma - (std * desvio)
        return upper, sma, lower
    
    def _calcular_atr_basico(self, dados: pd.DataFrame, periodo: int = 14) -> pd.Series:
        """Calcula ATR sem TA-Lib"""
        high_low = dados['High'] - dados['Low']
        high_close = np.abs(dados['High'] - dados['Close'].shift())
        low_close = np.abs(dados['Low'] - dados['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(periodo).mean()
        return atr
    
    def _somar_ultimos_12_meses(self, df, data_limite, data_inicio_janela, campo):
        """Soma valores dos últimos 12 meses (4 trimestres)"""
        if df is None or df.empty:
            return None
        
        # Filtrar colunas dentro da janela de 12 meses
        colunas_validas = [col for col in df.columns 
                         if isinstance(col, (pd.Timestamp, datetime)) 
                         and col <= data_limite 
                         and col >= data_inicio_janela]
        
        if not colunas_validas:
            return None
        
        # Ordenar por data (mais recente primeiro) e pegar últimos 4 trimestres
        colunas_ordenadas = sorted(colunas_validas, reverse=True)[:4]
        
        total = 0
        for col in colunas_ordenadas:
            try:
                if campo in df.index:
                    valor = df.loc[campo, col]
                    if pd.notna(valor) and valor != 0:
                        total += valor
            except:
                pass
        
        return total if total != 0 else None
    
    def _obter_valor_mais_recente(self, df, data_limite, campo):
        """Obtém o valor mais recente antes da data limite"""
        if df is None or df.empty:
            return None
        
        colunas_validas = [col for col in df.columns 
                         if isinstance(col, (pd.Timestamp, datetime)) 
                         and col <= data_limite]
        
        if not colunas_validas:
            return None
        
        colunas_ordenadas = sorted(colunas_validas, reverse=True)
        
        for col in colunas_ordenadas:
            try:
                if campo in df.index:
                    valor = df.loc[campo, col]
                    if pd.notna(valor) and valor != 0:
                        return valor
            except:
                pass
        
        return None
    
    def _adicionar_features_fundamentais(self, dados: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features fundamentais calculadas usando os últimos 12 meses de dados reais"""
        print("Calculando indicadores fundamentais usando últimos 12 meses de dados reais...")
        
        # Inicializar colunas - INDICADORES DE VALUATION
        dados['P_L_ratio'] = np.nan
        dados['P_VP_ratio'] = np.nan
        dados['Dividend_Yield'] = np.nan
        dados['PEG_Ratio'] = np.nan
        dados['EV_EBITDA'] = np.nan
        dados['EV_EBIT'] = np.nan
        dados['P_EBITDA'] = np.nan
        dados['P_EBIT'] = np.nan
        dados['VPA'] = np.nan
        dados['P_Ativo'] = np.nan
        dados['LPA'] = np.nan
        dados['P_SR'] = np.nan
        dados['P_Cap_Giro'] = np.nan
        dados['P_Ativo_Circ_Liq'] = np.nan
        
        # INDICADORES DE ENDIVIDAMENTO
        dados['Debt_to_Equity'] = np.nan
        dados['Div_Liquida_EBITDA'] = np.nan
        dados['Div_Liquida_EBIT'] = np.nan
        dados['PL_Ativos'] = np.nan
        dados['Passivos_Ativos'] = np.nan
        dados['Current_Ratio'] = np.nan
        
        # INDICADORES DE EFICIÊNCIA
        dados['Margem_Bruta'] = np.nan
        dados['Margem_EBITDA'] = np.nan
        dados['Margem_EBIT'] = np.nan
        dados['Margem_Liquida'] = np.nan
        
        # INDICADORES DE RENTABILIDADE
        dados['ROE'] = np.nan
        dados['ROA'] = np.nan
        dados['ROIC'] = np.nan
        dados['Giro_Ativos'] = np.nan
        
        # INDICADORES DE CRESCIMENTO
        dados['CAGR_Receitas_5Y'] = np.nan
        dados['CAGR_Lucros_5Y'] = np.nan
        
        try:
            # Obter objeto ticker
            acao = yf.Ticker(f"{self.ticker}.SA")
            info = acao.info
            
            # Obter dados financeiros trimestrais e anuais
            financials = acao.financials  # Dados anuais
            financials_quarterly = acao.quarterly_financials  # Dados trimestrais
            balance_sheet = acao.balance_sheet
            balance_sheet_quarterly = acao.quarterly_balance_sheet
            
            # Obter histórico de dividendos
            dividend_history = acao.dividends
            
            # Extrair informações básicas
            shares_outstanding = info.get('sharesOutstanding')
            if shares_outstanding is None or shares_outstanding <= 0:
                shares_outstanding = info.get('impliedSharesOutstanding')
            
            # Para cada data no DataFrame, calcular indicadores usando últimos 12 meses
            meses_processados = {}
            
            for data in dados.index:
                # Obter primeiro dia do mês
                primeiro_dia_mes = data.replace(day=1)
                
                # Se já processamos este mês, reutilizar valores
                if primeiro_dia_mes in meses_processados:
                    valores = meses_processados[primeiro_dia_mes]
                    for key, value in valores.items():
                        if key in dados.columns:
                            dados.loc[data, key] = value
                    continue
                
                # Calcular data de início (12 meses antes do primeiro dia do mês)
                data_inicio_janela = primeiro_dia_mes - timedelta(days=365)
                
                # Obter preço no primeiro dia do mês (ou próximo dia útil)
                try:
                    preco_data = dados.loc[primeiro_dia_mes:primeiro_dia_mes + timedelta(days=5), 'Close']
                    if len(preco_data) > 0:
                        preco = preco_data.iloc[0]
                    else:
                        preco = dados.loc[data, 'Close']
                except:
                    preco = dados.loc[data, 'Close']
                
                # 1. CALCULAR P/L (Preço/Lucro) usando lucro líquido dos últimos 12 meses
                pl_ratio = None
                net_income_12m = self._somar_ultimos_12_meses(financials_quarterly, primeiro_dia_mes, data_inicio_janela, 'Net Income')
                if net_income_12m is None:
                    # Tentar outros nomes de campo
                    net_income_12m = self._somar_ultimos_12_meses(financials_quarterly, primeiro_dia_mes, data_inicio_janela, 'NetIncome')
                if net_income_12m is None:
                    # Tentar dados anuais
                    net_income_12m = self._obter_valor_mais_recente(financials, primeiro_dia_mes, 'Net Income')
                    if net_income_12m is None:
                        net_income_12m = self._obter_valor_mais_recente(financials, primeiro_dia_mes, 'NetIncome')
                
                if net_income_12m is not None and net_income_12m > 0 and shares_outstanding and shares_outstanding > 0:
                    eps = net_income_12m / shares_outstanding
                    if eps > 0:
                        pl_ratio = preco / eps
                
                # Fallback para info do yfinance
                if pl_ratio is None or not (0 < pl_ratio < 1000):
                    trailing_pe = info.get('trailingPE')
                    if trailing_pe and 0 < trailing_pe < 1000:
                        pl_ratio = trailing_pe
                
                # 2. CALCULAR P/VP (Preço/Valor Patrimonial) usando patrimônio líquido mais recente
                pvp_ratio = None
                total_equity = self._obter_valor_mais_recente(balance_sheet_quarterly, primeiro_dia_mes, 'Stockholders Equity')
                if total_equity is None:
                    total_equity = self._obter_valor_mais_recente(balance_sheet_quarterly, primeiro_dia_mes, 'Total Stockholder Equity')
                if total_equity is None:
                    total_equity = self._obter_valor_mais_recente(balance_sheet, primeiro_dia_mes, 'Stockholders Equity')
                    if total_equity is None:
                        total_equity = self._obter_valor_mais_recente(balance_sheet, primeiro_dia_mes, 'Total Stockholder Equity')
                
                if total_equity is not None and total_equity > 0 and shares_outstanding and shares_outstanding > 0:
                    vpa = total_equity / shares_outstanding
                    if vpa > 0:
                        pvp_ratio = preco / vpa
                
                # Fallback para info do yfinance
                if pvp_ratio is None or not (0 < pvp_ratio < 50):
                    price_to_book = info.get('priceToBook')
                    if price_to_book and 0 < price_to_book < 50:
                        pvp_ratio = price_to_book
                
                # 3. CALCULAR ROE (Return on Equity) usando lucro líquido dos últimos 12 meses
                roe = None
                # Usar o mesmo net_income_12m calculado para P/L
                if net_income_12m is not None and net_income_12m != 0 and total_equity is not None and total_equity > 0:
                    roe = net_income_12m / total_equity
                
                # Fallback para info do yfinance
                if roe is None:
                    roe_info = info.get('returnOnEquity')
                    if roe_info and -1 < roe_info < 10:
                        roe = roe_info
                
                # 4. CALCULAR DIVIDEND YIELD usando dividendos dos últimos 12 meses
                dividend_yield = None
                if dividend_history is not None and not dividend_history.empty:
                    # Filtrar dividendos dos últimos 12 meses
                    dividendos_12m = dividend_history[
                        (dividend_history.index >= data_inicio_janela) & 
                        (dividend_history.index <= primeiro_dia_mes)
                    ]
                    if not dividendos_12m.empty:
                        total_dividendos_12m = dividendos_12m.sum()
                        if total_dividendos_12m > 0 and preco > 0:
                            dividend_yield = total_dividendos_12m / preco
                
                # Fallback para info do yfinance
                if dividend_yield is None:
                    dy_info = info.get('dividendYield')
                    if dy_info and 0 <= dy_info <= 0.5:
                        dividend_yield = dy_info
                
                # 5. CALCULAR MARGEM LÍQUIDA usando dados dos últimos 12 meses
                margem_liquida = None
                # Receita total dos últimos 12 meses
                total_revenue_12m = self._somar_ultimos_12_meses(financials_quarterly, primeiro_dia_mes, data_inicio_janela, 'Total Revenue')
                if total_revenue_12m is None:
                    total_revenue_12m = self._somar_ultimos_12_meses(financials_quarterly, primeiro_dia_mes, data_inicio_janela, 'Revenue')
                if total_revenue_12m is None:
                    total_revenue_12m = self._obter_valor_mais_recente(financials, primeiro_dia_mes, 'Total Revenue')
                    if total_revenue_12m is None:
                        total_revenue_12m = self._obter_valor_mais_recente(financials, primeiro_dia_mes, 'Revenue')
                
                if net_income_12m is not None and total_revenue_12m is not None and total_revenue_12m > 0:
                    margem_liquida = net_income_12m / total_revenue_12m
                
                # Fallback para info do yfinance
                if margem_liquida is None:
                    profit_margin = info.get('profitMargins')
                    if profit_margin is not None and -1 < profit_margin < 1:
                        margem_liquida = profit_margin
                
                # 6. CALCULAR DÍVIDA/PATRIMÔNIO (Debt to Equity) usando dados mais recentes
                debt_to_equity = None
                total_debt = self._obter_valor_mais_recente(balance_sheet_quarterly, primeiro_dia_mes, 'Total Debt')
                if total_debt is None:
                    total_debt = self._obter_valor_mais_recente(balance_sheet_quarterly, primeiro_dia_mes, 'Total Liabilities')
                if total_debt is None:
                    total_debt = self._obter_valor_mais_recente(balance_sheet, primeiro_dia_mes, 'Total Debt')
                    if total_debt is None:
                        total_debt = self._obter_valor_mais_recente(balance_sheet, primeiro_dia_mes, 'Total Liabilities')
                
                if total_debt is not None and total_equity is not None and total_equity > 0:
                    debt_to_equity = total_debt / total_equity
                
                # Fallback para info do yfinance
                if debt_to_equity is None:
                    dte_info = info.get('debtToEquity')
                    if dte_info and 0 <= dte_info <= 10:
                        debt_to_equity = dte_info
                
                # ========== CALCULAR INDICADORES ADICIONAIS ==========
                
                # Obter dados adicionais necessários
                market_cap = info.get('marketCap')
                if market_cap is None or market_cap <= 0:
                    market_cap = preco * shares_outstanding if shares_outstanding else None
                
                # Calcular EBITDA e EBIT dos últimos 12 meses
                ebitda_12m = self._somar_ultimos_12_meses(financials_quarterly, primeiro_dia_mes, data_inicio_janela, 'EBITDA')
                if ebitda_12m is None:
                    ebitda_12m = self._obter_valor_mais_recente(financials, primeiro_dia_mes, 'EBITDA')
                
                ebit_12m = self._somar_ultimos_12_meses(financials_quarterly, primeiro_dia_mes, data_inicio_janela, 'EBIT')
                if ebit_12m is None:
                    ebit_12m = self._obter_valor_mais_recente(financials, primeiro_dia_mes, 'EBIT')
                if ebit_12m is None:
                    # EBIT pode ser calculado como: Receita - Custo dos Produtos Vendidos - Despesas Operacionais
                    # Ou: EBITDA - Depreciação e Amortização
                    if ebitda_12m is not None:
                        # Tentar obter D&A
                        da_12m = self._somar_ultimos_12_meses(financials_quarterly, primeiro_dia_mes, data_inicio_janela, 'Depreciation And Amortization')
                        if da_12m is None:
                            da_12m = self._obter_valor_mais_recente(financials, primeiro_dia_mes, 'Depreciation And Amortization')
                        if da_12m is not None:
                            ebit_12m = ebitda_12m - da_12m
                
                # Calcular dívida líquida
                cash = self._obter_valor_mais_recente(balance_sheet_quarterly, primeiro_dia_mes, 'Cash And Cash Equivalents')
                if cash is None:
                    cash = self._obter_valor_mais_recente(balance_sheet, primeiro_dia_mes, 'Cash And Cash Equivalents')
                if cash is None:
                    cash = self._obter_valor_mais_recente(balance_sheet_quarterly, primeiro_dia_mes, 'Cash')
                    if cash is None:
                        cash = self._obter_valor_mais_recente(balance_sheet, primeiro_dia_mes, 'Cash')
                
                divida_liquida = None
                if total_debt is not None:
                    divida_liquida = total_debt - (cash if cash is not None else 0)
                
                # Calcular ativos totais
                total_assets = self._obter_valor_mais_recente(balance_sheet_quarterly, primeiro_dia_mes, 'Total Assets')
                if total_assets is None:
                    total_assets = self._obter_valor_mais_recente(balance_sheet, primeiro_dia_mes, 'Total Assets')
                
                # Calcular passivos totais
                total_liabilities = self._obter_valor_mais_recente(balance_sheet_quarterly, primeiro_dia_mes, 'Total Liabilities')
                if total_liabilities is None:
                    total_liabilities = self._obter_valor_mais_recente(balance_sheet, primeiro_dia_mes, 'Total Liabilities')
                
                # Calcular receita bruta e custo dos produtos vendidos
                gross_profit_12m = self._somar_ultimos_12_meses(financials_quarterly, primeiro_dia_mes, data_inicio_janela, 'Gross Profit')
                if gross_profit_12m is None:
                    gross_profit_12m = self._obter_valor_mais_recente(financials, primeiro_dia_mes, 'Gross Profit')
                
                # Calcular capital de giro (Ativo Circulante - Passivo Circulante)
                current_assets = self._obter_valor_mais_recente(balance_sheet_quarterly, primeiro_dia_mes, 'Current Assets')
                if current_assets is None:
                    current_assets = self._obter_valor_mais_recente(balance_sheet_quarterly, primeiro_dia_mes, 'Total Current Assets')
                if current_assets is None:
                    current_assets = self._obter_valor_mais_recente(balance_sheet, primeiro_dia_mes, 'Current Assets')
                    if current_assets is None:
                        current_assets = self._obter_valor_mais_recente(balance_sheet, primeiro_dia_mes, 'Total Current Assets')
                
                current_liabilities = self._obter_valor_mais_recente(balance_sheet_quarterly, primeiro_dia_mes, 'Current Liabilities')
                if current_liabilities is None:
                    current_liabilities = self._obter_valor_mais_recente(balance_sheet_quarterly, primeiro_dia_mes, 'Total Current Liabilities')
                if current_liabilities is None:
                    current_liabilities = self._obter_valor_mais_recente(balance_sheet, primeiro_dia_mes, 'Current Liabilities')
                    if current_liabilities is None:
                        current_liabilities = self._obter_valor_mais_recente(balance_sheet, primeiro_dia_mes, 'Total Current Liabilities')
                
                # Calcular Current Ratio
                current_ratio = None
                if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
                    current_ratio = current_assets / current_liabilities
                
                # Fallback para info do yfinance
                if current_ratio is None:
                    current_ratio = info.get('currentRatio')
                    if current_ratio and not (0 < current_ratio <= 20):
                        current_ratio = None
                
                working_capital = None
                if current_assets is not None and current_liabilities is not None:
                    working_capital = current_assets - current_liabilities
                
                # Calcular ativo circulante líquido (Ativo Circulante - Passivo Circulante)
                ativo_circ_liq = working_capital
                
                # ========== INDICADORES DE VALUATION ADICIONAIS ==========
                
                # PEG Ratio = P/L / Taxa de Crescimento de Lucros
                peg_ratio = None
                if pl_ratio is not None and pl_ratio > 0:
                    # Tentar obter PEG do yfinance
                    peg_info = info.get('pegRatio')
                    if peg_info and -10 < peg_info < 10:
                        peg_ratio = peg_info
                
                # Enterprise Value
                enterprise_value = None
                if market_cap is not None:
                    if divida_liquida is not None:
                        enterprise_value = market_cap + divida_liquida
                    else:
                        enterprise_value = market_cap
                
                # EV/EBITDA
                ev_ebitda = None
                if enterprise_value is not None and ebitda_12m is not None and ebitda_12m > 0:
                    ev_ebitda = enterprise_value / ebitda_12m
                
                # EV/EBIT
                ev_ebit = None
                if enterprise_value is not None and ebit_12m is not None and ebit_12m > 0:
                    ev_ebit = enterprise_value / ebit_12m
                
                # P/EBITDA
                p_ebitda = None
                if ebitda_12m is not None and ebitda_12m > 0 and shares_outstanding and shares_outstanding > 0:
                    ebitda_per_share = ebitda_12m / shares_outstanding
                    if ebitda_per_share > 0:
                        p_ebitda = preco / ebitda_per_share
                
                # P/EBIT
                p_ebit = None
                if ebit_12m is not None and ebit_12m > 0 and shares_outstanding and shares_outstanding > 0:
                    ebit_per_share = ebit_12m / shares_outstanding
                    if ebit_per_share > 0:
                        p_ebit = preco / ebit_per_share
                
                # VPA (Valor Patrimonial por Ação) - já calculado como vpa
                vpa = None
                if total_equity is not None and shares_outstanding and shares_outstanding > 0:
                    vpa = total_equity / shares_outstanding
                
                # P/Ativo
                p_ativo = None
                if total_assets is not None and total_assets > 0 and shares_outstanding and shares_outstanding > 0:
                    ativo_per_share = total_assets / shares_outstanding
                    if ativo_per_share > 0:
                        p_ativo = preco / ativo_per_share
                
                # LPA (Lucro por Ação) - já calculado como eps
                lpa = None
                if net_income_12m is not None and net_income_12m > 0 and shares_outstanding and shares_outstanding > 0:
                    lpa = net_income_12m / shares_outstanding
                
                # P/SR (Preço/Receita)
                p_sr = None
                if total_revenue_12m is not None and total_revenue_12m > 0 and shares_outstanding and shares_outstanding > 0:
                    revenue_per_share = total_revenue_12m / shares_outstanding
                    if revenue_per_share > 0:
                        p_sr = preco / revenue_per_share
                
                # P/Cap. Giro (Preço/Capital de Giro)
                p_cap_giro = None
                if working_capital is not None and working_capital != 0 and shares_outstanding and shares_outstanding > 0:
                    cap_giro_per_share = working_capital / shares_outstanding
                    if cap_giro_per_share != 0:
                        p_cap_giro = preco / cap_giro_per_share
                
                # P/Ativo Circ. Liq.
                p_ativo_circ_liq = None
                if ativo_circ_liq is not None and ativo_circ_liq != 0 and shares_outstanding and shares_outstanding > 0:
                    ativo_circ_liq_per_share = ativo_circ_liq / shares_outstanding
                    if ativo_circ_liq_per_share != 0:
                        p_ativo_circ_liq = preco / ativo_circ_liq_per_share
                
                # ========== INDICADORES DE ENDIVIDAMENTO ADICIONAIS ==========
                
                # Dívida Líquida/EBITDA
                div_liquida_ebitda = None
                if divida_liquida is not None and ebitda_12m is not None and ebitda_12m > 0:
                    div_liquida_ebitda = divida_liquida / ebitda_12m
                
                # Dívida Líquida/EBIT
                div_liquida_ebit = None
                if divida_liquida is not None and ebit_12m is not None and ebit_12m > 0:
                    div_liquida_ebit = divida_liquida / ebit_12m
                
                # PL/Ativos
                pl_ativos = None
                if total_equity is not None and total_assets is not None and total_assets > 0:
                    pl_ativos = total_equity / total_assets
                
                # Passivos/Ativos
                passivos_ativos = None
                if total_liabilities is not None and total_assets is not None and total_assets > 0:
                    passivos_ativos = total_liabilities / total_assets
                
                # ========== INDICADORES DE EFICIÊNCIA ==========
                
                # Margem Bruta
                margem_bruta = None
                if gross_profit_12m is not None and total_revenue_12m is not None and total_revenue_12m > 0:
                    margem_bruta = gross_profit_12m / total_revenue_12m
                
                # Margem EBITDA
                margem_ebitda = None
                if ebitda_12m is not None and total_revenue_12m is not None and total_revenue_12m > 0:
                    margem_ebitda = ebitda_12m / total_revenue_12m
                
                # Margem EBIT
                margem_ebit = None
                if ebit_12m is not None and total_revenue_12m is not None and total_revenue_12m > 0:
                    margem_ebit = ebit_12m / total_revenue_12m
                
                # ========== INDICADORES DE RENTABILIDADE ADICIONAIS ==========
                
                # ROA (Return on Assets)
                roa = None
                if net_income_12m is not None and total_assets is not None and total_assets > 0:
                    roa = net_income_12m / total_assets
                
                # ROIC (Return on Invested Capital)
                roic = None
                # ROIC = (EBIT * (1 - Tax Rate)) / (Dívida + Patrimônio Líquido - Caixa)
                if ebit_12m is not None and ebit_12m > 0:
                    # Tentar obter taxa de imposto
                    tax_rate = info.get('taxRate')
                    if tax_rate is None:
                        tax_rate = 0.34  # Taxa padrão Brasil
                    
                    invested_capital = None
                    if total_debt is not None and total_equity is not None:
                        invested_capital = total_debt + total_equity - (cash if cash is not None else 0)
                    
                    if invested_capital is not None and invested_capital > 0:
                        nopat = ebit_12m * (1 - tax_rate)  # Net Operating Profit After Tax
                        roic = nopat / invested_capital
                
                # Giro de Ativos (Asset Turnover)
                giro_ativos = None
                if total_revenue_12m is not None and total_assets is not None and total_assets > 0:
                    giro_ativos = total_revenue_12m / total_assets
                
                # ========== INDICADORES DE CRESCIMENTO ==========
                
                # CAGR Receitas 5 anos
                cagr_receitas_5y = None
                if financials is not None and not financials.empty:
                    # Obter receitas dos últimos 5 anos
                    receitas_5y = []
                    for col in sorted(financials.columns, reverse=True)[:5]:
                        try:
                            if 'Total Revenue' in financials.index:
                                rev = financials.loc['Total Revenue', col]
                            elif 'Revenue' in financials.index:
                                rev = financials.loc['Revenue', col]
                            else:
                                rev = None
                            
                            if rev is not None and pd.notna(rev) and rev > 0:
                                receitas_5y.append(rev)
                        except:
                            pass
                    
                    if len(receitas_5y) >= 2:
                        receita_inicial = receitas_5y[-1]
                        receita_final = receitas_5y[0]
                        if receita_inicial > 0:
                            cagr_receitas_5y = ((receita_final / receita_inicial) ** (1 / len(receitas_5y)) - 1) if len(receitas_5y) > 1 else None
                
                # CAGR Lucros 5 anos
                cagr_lucros_5y = None
                if financials is not None and not financials.empty:
                    # Obter lucros dos últimos 5 anos
                    lucros_5y = []
                    for col in sorted(financials.columns, reverse=True)[:5]:
                        try:
                            if 'Net Income' in financials.index:
                                lucro = financials.loc['Net Income', col]
                            elif 'NetIncome' in financials.index:
                                lucro = financials.loc['NetIncome', col]
                            else:
                                lucro = None
                            
                            if lucro is not None and pd.notna(lucro):
                                lucros_5y.append(lucro)
                        except:
                            pass
                    
                    if len(lucros_5y) >= 2:
                        # Filtrar valores negativos ou zero
                        lucros_validos = [l for l in lucros_5y if l > 0]
                        if len(lucros_validos) >= 2:
                            lucro_inicial = lucros_validos[-1]
                            lucro_final = lucros_validos[0]
                            if lucro_inicial > 0:
                                cagr_lucros_5y = ((lucro_final / lucro_inicial) ** (1 / len(lucros_validos)) - 1) if len(lucros_validos) > 1 else None
                
                # Armazenar valores calculados
                valores = {
                    # VALUATION
                    'P_L_ratio': pl_ratio if pl_ratio and 0 < pl_ratio < 1000 else np.nan,
                    'P_VP_ratio': pvp_ratio if pvp_ratio and 0 < pvp_ratio < 50 else np.nan,
                    'Dividend_Yield': dividend_yield if dividend_yield and 0 <= dividend_yield <= 0.5 else np.nan,
                    'PEG_Ratio': peg_ratio if peg_ratio and -10 < peg_ratio < 10 else np.nan,
                    'EV_EBITDA': ev_ebitda if ev_ebitda and 0 < ev_ebitda < 100 else np.nan,
                    'EV_EBIT': ev_ebit if ev_ebit and 0 < ev_ebit < 100 else np.nan,
                    'P_EBITDA': p_ebitda if p_ebitda and 0 < p_ebitda < 100 else np.nan,
                    'P_EBIT': p_ebit if p_ebit and 0 < p_ebit < 100 else np.nan,
                    'VPA': vpa if vpa and vpa > 0 else np.nan,
                    'P_Ativo': p_ativo if p_ativo and 0 < p_ativo < 10 else np.nan,
                    'LPA': lpa if lpa and lpa != 0 else np.nan,
                    'P_SR': p_sr if p_sr and 0 < p_sr < 50 else np.nan,
                    'P_Cap_Giro': p_cap_giro if p_cap_giro and -100 < p_cap_giro < 100 else np.nan,
                    'P_Ativo_Circ_Liq': p_ativo_circ_liq if p_ativo_circ_liq and -100 < p_ativo_circ_liq < 100 else np.nan,
                    
                    # ENDIVIDAMENTO
                    'Debt_to_Equity': debt_to_equity if debt_to_equity and 0 <= debt_to_equity <= 10 else np.nan,
                    'Div_Liquida_EBITDA': div_liquida_ebitda if div_liquida_ebitda and -10 < div_liquida_ebitda < 50 else np.nan,
                    'Div_Liquida_EBIT': div_liquida_ebit if div_liquida_ebit and -10 < div_liquida_ebit < 50 else np.nan,
                    'PL_Ativos': pl_ativos if pl_ativos and 0 <= pl_ativos <= 1 else np.nan,
                    'Passivos_Ativos': passivos_ativos if passivos_ativos and 0 <= passivos_ativos <= 1 else np.nan,
                    'Current_Ratio': current_ratio if current_ratio and 0 < current_ratio <= 20 else np.nan,
                    
                    # EFICIÊNCIA
                    'Margem_Bruta': margem_bruta if margem_bruta and -1 < margem_bruta < 1 else np.nan,
                    'Margem_EBITDA': margem_ebitda if margem_ebitda and -1 < margem_ebitda < 1 else np.nan,
                    'Margem_EBIT': margem_ebit if margem_ebit and -1 < margem_ebit < 1 else np.nan,
                    'Margem_Liquida': margem_liquida if margem_liquida and -1 < margem_liquida < 1 else np.nan,
                    
                    # RENTABILIDADE
                    'ROE': roe if roe and -1 < roe < 10 else np.nan,
                    'ROA': roa if roa and -1 < roa < 1 else np.nan,
                    'ROIC': roic if roic and -1 < roic < 1 else np.nan,
                    'Giro_Ativos': giro_ativos if giro_ativos and 0 < giro_ativos < 10 else np.nan,
                    
                    # CRESCIMENTO
                    'CAGR_Receitas_5Y': cagr_receitas_5y if cagr_receitas_5y and -1 < cagr_receitas_5y < 10 else np.nan,
                    'CAGR_Lucros_5Y': cagr_lucros_5y if cagr_lucros_5y and -1 < cagr_lucros_5y < 10 else np.nan
                }
                
                meses_processados[primeiro_dia_mes] = valores
                
                # Atribuir valores ao DataFrame
                for key, value in valores.items():
                    if key in dados.columns:
                        dados.loc[data, key] = value
            
            # Current_Ratio já foi calculado no loop principal acima
            
            # Preencher NaN apenas com forward/backward fill (dados reais anteriores/posteriores)
            # NÃO usar valores padrão simulados
            todas_colunas = ['P_L_ratio', 'P_VP_ratio', 'Dividend_Yield', 'PEG_Ratio', 'EV_EBITDA', 'EV_EBIT',
                           'P_EBITDA', 'P_EBIT', 'VPA', 'P_Ativo', 'LPA', 'P_SR', 'P_Cap_Giro', 'P_Ativo_Circ_Liq',
                           'Debt_to_Equity', 'Div_Liquida_EBITDA', 'Div_Liquida_EBIT', 'PL_Ativos', 'Passivos_Ativos', 'Current_Ratio',
                           'Margem_Bruta', 'Margem_EBITDA', 'Margem_EBIT', 'Margem_Liquida',
                           'ROE', 'ROA', 'ROIC', 'Giro_Ativos',
                           'CAGR_Receitas_5Y', 'CAGR_Lucros_5Y']
            
            for col in todas_colunas:
                if col in dados.columns:
                    dados[col] = dados[col].ffill().bfill()
            
            # Estatísticas dos indicadores calculados
            print("Indicadores calculados usando últimos 12 meses de dados reais:")
            categorias = {
                'VALUATION': ['P_L_ratio', 'P_VP_ratio', 'Dividend_Yield', 'PEG_Ratio', 'EV_EBITDA', 'EV_EBIT', 'P_EBITDA', 'P_EBIT', 'VPA', 'P_Ativo', 'LPA', 'P_SR', 'P_Cap_Giro', 'P_Ativo_Circ_Liq'],
                'ENDIVIDAMENTO': ['Debt_to_Equity', 'Div_Liquida_EBITDA', 'Div_Liquida_EBIT', 'PL_Ativos', 'Passivos_Ativos', 'Current_Ratio'],
                'EFICIÊNCIA': ['Margem_Bruta', 'Margem_EBITDA', 'Margem_EBIT', 'Margem_Liquida'],
                'RENTABILIDADE': ['ROE', 'ROA', 'ROIC', 'Giro_Ativos'],
                'CRESCIMENTO': ['CAGR_Receitas_5Y', 'CAGR_Lucros_5Y']
            }
            
            for categoria, indicadores in categorias.items():
                print(f"\n  {categoria}:")
                for ind in indicadores:
                    if ind in dados.columns:
                        mean_val = dados[ind].mean()
                        count_val = dados[ind].notna().sum()
                        if not np.isnan(mean_val):
                            print(f"    {ind}: média={mean_val:.4f} ({count_val} valores)")
                        else:
                            print(f"    {ind}: N/A ({count_val} valores)")
            
        except Exception as e:
            print(f"Erro ao calcular indicadores fundamentais: {str(e)}")
            print("AVISO: Alguns indicadores podem estar como NaN - apenas dados reais são usados")
            # NÃO preencher com valores padrão - deixar NaN se não houver dados reais
        
        return dados
    
    def _adicionar_targets(self, dados: pd.DataFrame) -> pd.DataFrame:
        """Adiciona targets para treinamento (retornos futuros)"""
        # Retornos para diferentes horizontes
        dados['Return_1d'] = dados['Close'].shift(-1) / dados['Close'] - 1
        dados['Return_5d'] = dados['Close'].shift(-5) / dados['Close'] - 1
        dados['Return_10d'] = dados['Close'].shift(-10) / dados['Close'] - 1
        dados['Return_20d'] = dados['Close'].shift(-20) / dados['Close'] - 1
        
        # Classificação binária (alta/baixa)
        dados['Direction_1d'] = (dados['Return_1d'] > 0).astype(int)
        dados['Direction_5d'] = (dados['Return_5d'] > 0).astype(int)
        
        # Volatilidade futura
        dados['Volatility_5d'] = dados['Return_1d'].rolling(5).std().shift(-5)
        
        return dados
    
    def preparar_dados_treinamento(self, features: List[str], target: str, 
                                 janela_temporal: int = 60, 
                                 split_temporal: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepara dados para treinamento de redes neurais"""
        if self.dados_historicos is None:
            self.obter_dados_historicos()
        
        # Remover NaN
        dados_limpos = self.dados_historicos.dropna()
        
        # Selecionar features
        X = dados_limpos[features].values
        y = dados_limpos[target].values
        
        # Criar sequências temporais
        X_seq, y_seq = [], []
        for i in range(janela_temporal, len(X)):
            X_seq.append(X[i-janela_temporal:i])
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split temporal (não aleatório para séries temporais)
        split_idx = int(len(X_seq) * split_temporal)
        
        X_train = X_seq[:split_idx]
        X_test = X_seq[split_idx:]
        y_train = y_seq[:split_idx]
        y_test = y_seq[split_idx:]
        
        return X_train, X_test, y_train, y_test

class LSTMPredictor:
    """Preditor LSTM para séries temporais de preços"""
    
    def __init__(self, input_shape: Tuple[int, int], 
                 lstm_units: List[int] = [50, 50],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def construir_modelo(self) -> keras.Model:
        """Constrói modelo LSTM"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow não está disponível")
        
        model = models.Sequential()
        
        # Primeira camada LSTM
        model.add(layers.LSTM(
            self.lstm_units[0],
            return_sequences=True,
            input_shape=self.input_shape
        ))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Camadas LSTM adicionais
        for units in self.lstm_units[1:]:
            model.add(layers.LSTM(units, return_sequences=True))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Camada LSTM final
        model.add(layers.LSTM(self.lstm_units[-1], return_sequences=False))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Camadas densas
        model.add(layers.Dense(25, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(1, activation='linear'))
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=Huber(),
            metrics=[MeanAbsoluteError()]
        )
        
        self.model = model
        return model
    
    def treinar(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 100, batch_size: int = 32,
                early_stopping: bool = True) -> Dict:
        """Treina o modelo LSTM"""
        if self.model is None:
            self.construir_modelo()
        
        # Callbacks
        callbacks_list = []
        if early_stopping:
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
            callbacks_list.append(early_stop)
        
        # Redução de learning rate
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        callbacks_list.append(reduce_lr)
        
        # Treinamento
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=0
        )
        
        return {
            'history': self.history.history,
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1]
        }
    
    def prever(self, X: np.ndarray) -> np.ndarray:
        """Faz predições"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado")
        return self.model.predict(X, verbose=0)
    
    def avaliar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Avalia o modelo"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado")
        
        predictions = self.prever(X_test)
        
        # Métricas
        mse = np.mean((predictions.flatten() - y_test) ** 2)
        mae = np.mean(np.abs(predictions.flatten() - y_test))
        rmse = np.sqrt(mse)
        
        # R² Score
        ss_res = np.sum((y_test - predictions.flatten()) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'predictions': predictions.flatten(),
            'actual': y_test
        }

class CNNPredictor:
    """Preditor CNN para análise de padrões de preços"""
    
    def __init__(self, input_shape: Tuple[int, int],
                 filters: List[int] = [32, 64, 128],
                 kernel_size: int = 3,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def construir_modelo(self) -> keras.Model:
        """Constrói modelo CNN"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow não está disponível")
        
        model = models.Sequential()
        
        # Camadas convolucionais
        for i, filters in enumerate(self.filters):
            if i == 0:
                model.add(layers.Conv1D(
                    filters=filters,
                    kernel_size=self.kernel_size,
                    activation='relu',
                    input_shape=self.input_shape
                ))
            else:
                model.add(layers.Conv1D(
                    filters=filters,
                    kernel_size=self.kernel_size,
                    activation='relu'
                ))
            
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Camadas densas
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(25, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(1, activation='linear'))
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=Huber(),
            metrics=[MeanAbsoluteError()]
        )
        
        self.model = model
        return model
    
    def treinar(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 100, batch_size: int = 32) -> Dict:
        """Treina o modelo CNN"""
        if self.model is None:
            self.construir_modelo()
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Treinamento
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        return {
            'history': self.history.history,
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1]
        }
    
    def prever(self, X: np.ndarray) -> np.ndarray:
        """Faz predições"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado")
        return self.model.predict(X, verbose=0)
    
    def avaliar(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Avalia o modelo"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado")
        
        predictions = self.prever(X_test)
        
        # Métricas
        mse = np.mean((predictions.flatten() - y_test) ** 2)
        mae = np.mean(np.abs(predictions.flatten() - y_test))
        rmse = np.sqrt(mse)
        
        # R² Score
        ss_res = np.sum((y_test - predictions.flatten()) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'predictions': predictions.flatten(),
            'actual': y_test
        }

class EnsembleDeepLearning:
    """Sistema de ensemble com múltiplas redes neurais"""
    
    def __init__(self, input_shape: Tuple[int, int]):
        self.input_shape = input_shape
        self.models = {}
        self.weights = {}
        self.trained = False
    
    def adicionar_modelo(self, nome: str, modelo, peso: float = 1.0):
        """Adiciona um modelo ao ensemble"""
        self.models[nome] = modelo
        self.weights[nome] = peso
    
    def treinar_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        epochs: int = 100, batch_size: int = 32) -> Dict:
        """Treina todos os modelos do ensemble"""
        resultados = {}
        
        for nome, modelo in self.models.items():
            print(f"Treinando {nome}...")
            
            if hasattr(modelo, 'treinar'):
                resultado = modelo.treinar(X_train, y_train, X_val, y_val, epochs, batch_size)
                resultados[nome] = resultado
            else:
                raise ValueError(f"Modelo {nome} não tem método treinar")
        
        self.trained = True
        return resultados
    
    def prever_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Faz predições usando ensemble"""
        if not self.trained:
            raise ValueError("Ensemble não foi treinado")
        
        predicoes = {}
        for nome, modelo in self.models.items():
            if hasattr(modelo, 'prever'):
                predicoes[nome] = modelo.prever(X)
            else:
                raise ValueError(f"Modelo {nome} não tem método prever")
        
        # Combinação ponderada
        predicao_final = np.zeros_like(predicoes[list(predicoes.keys())[0]])
        peso_total = sum(self.weights.values())
        
        for nome, predicao in predicoes.items():
            peso = self.weights[nome] / peso_total
            predicao_final += peso * predicao
        
        return predicao_final
    
    def avaliar_ensemble(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Avalia o ensemble"""
        if not self.trained:
            raise ValueError("Ensemble não foi treinado")
        
        predictions = self.prever_ensemble(X_test)
        
        # Métricas
        mse = np.mean((predictions.flatten() - y_test) ** 2)
        mae = np.mean(np.abs(predictions.flatten() - y_test))
        rmse = np.sqrt(mse)
        
        # R² Score
        ss_res = np.sum((y_test - predictions.flatten()) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'predictions': predictions.flatten(),
            'actual': y_test
        }

class SistemaDeepLearning:
    """Sistema principal de deep learning para análise de ações"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.backtest = BacktestHistorico(ticker)
        self.ensemble = None
        self.features_selecionadas = None
        self.target_selecionado = None
        self.resultados_treinamento = None
        
    def configurar_sistema(self, features: List[str] = None, target: str = "Return_5d",
                          janela_temporal: int = 60, split_temporal: float = 0.8):
        """Configura o sistema de deep learning"""
        
        # Features padrão se não especificadas
        if features is None:
            features = [
                # Técnicas
                'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR',
                'Volume_SMA', 'Price_Change', 'Volatility',
                # Valuation
                'P_L_ratio', 'P_VP_ratio', 'Dividend_Yield', 'PEG_Ratio',
                'EV_EBITDA', 'EV_EBIT', 'P_EBITDA', 'P_EBIT', 'VPA', 'P_Ativo',
                'LPA', 'P_SR', 'P_Cap_Giro', 'P_Ativo_Circ_Liq',
                # Endividamento
                'Debt_to_Equity', 'Div_Liquida_EBITDA', 'Div_Liquida_EBIT',
                'PL_Ativos', 'Passivos_Ativos', 'Current_Ratio',
                # Eficiência
                'Margem_Bruta', 'Margem_EBITDA', 'Margem_EBIT', 'Margem_Liquida',
                # Rentabilidade
                'ROE', 'ROA', 'ROIC', 'Giro_Ativos',
                # Crescimento
                'CAGR_Receitas_5Y', 'CAGR_Lucros_5Y'
            ]
        
        self.features_selecionadas = features
        self.target_selecionado = target
        
        # Obter dados históricos
        print(f"Obtendo dados históricos para {self.ticker}...")
        self.backtest.obter_dados_historicos()
        
        # Preparar dados de treinamento
        print("Preparando dados de treinamento...")
        X_train, X_test, y_train, y_test = self.backtest.preparar_dados_treinamento(
            features, target, janela_temporal, split_temporal
        )
        
        print(f"Dados de treinamento: {X_train.shape}")
        print(f"Dados de teste: {X_test.shape}")
        
        # Criar ensemble
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.ensemble = EnsembleDeepLearning(input_shape)
        
        # Adicionar modelos ao ensemble
        lstm_model = LSTMPredictor(input_shape, lstm_units=[50, 50, 25])
        cnn_model = CNNPredictor(input_shape, filters=[32, 64, 128])
        
        self.ensemble.adicionar_modelo("LSTM", lstm_model, peso=0.6)
        self.ensemble.adicionar_modelo("CNN", cnn_model, peso=0.4)
        
        return X_train, X_test, y_train, y_test
    
    def treinar_sistema(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       epochs: int = 100, batch_size: int = 32) -> Dict:
        """Treina o sistema de deep learning"""
        
        if self.ensemble is None:
            raise ValueError("Sistema não foi configurado")
        
        print("Iniciando treinamento do ensemble...")
        
        # Split para validação
        split_idx = int(len(X_train) * 0.8)
        X_train_split = X_train[:split_idx]
        X_val_split = X_train[split_idx:]
        y_train_split = y_train[:split_idx]
        y_val_split = y_train[split_idx:]
        
        # Treinar ensemble
        resultados_treinamento = self.ensemble.treinar_ensemble(
            X_train_split, y_train_split, X_val_split, y_val_split,
            epochs, batch_size
        )
        
        # Avaliar ensemble
        print("Avaliando ensemble...")
        resultados_avaliacao = self.ensemble.avaliar_ensemble(X_test, y_test)
        
        self.resultados_treinamento = {
            'treinamento': resultados_treinamento,
            'avaliacao': resultados_avaliacao
        }
        
        return self.resultados_treinamento
    
    def fazer_predicoes(self, X: np.ndarray) -> np.ndarray:
        """Faz predições usando o sistema treinado"""
        if self.ensemble is None or not self.ensemble.trained:
            raise ValueError("Sistema não foi treinado")
        
        return self.ensemble.prever_ensemble(X)
    
    def obter_metricas(self) -> Dict:
        """Obtém métricas do sistema"""
        if self.resultados_treinamento is None:
            raise ValueError("Sistema não foi treinado")
        
        return self.resultados_treinamento['avaliacao']
    
    def gerar_relatorio(self) -> Dict:
        """Gera relatório completo do sistema"""
        if self.resultados_treinamento is None:
            raise ValueError("Sistema não foi treinado")
        
        metricas = self.obter_metricas()
        
        relatorio = {
            'ticker': self.ticker,
            'features_utilizadas': self.features_selecionadas,
            'target': self.target_selecionado,
            'modelos_ensemble': list(self.ensemble.models.keys()),
            'pesos_ensemble': self.ensemble.weights,
            'metricas': {
                'r2_score': metricas['r2_score'],
                'rmse': metricas['rmse'],
                'mae': metricas['mae'],
                'mse': metricas['mse']
            },
            'qualidade_modelo': self._classificar_qualidade(metricas['r2_score']),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return relatorio
    
    def _classificar_qualidade(self, r2_score: float) -> str:
        """Classifica a qualidade do modelo baseado no R²"""
        if r2_score >= 0.8:
            return "EXCELENTE"
        elif r2_score >= 0.6:
            return "BOM"
        elif r2_score >= 0.4:
            return "REGULAR"
        elif r2_score >= 0.2:
            return "RUIM"
        else:
            return "MUITO_RUIM"
    
    def analisar_sinal_compra_venda(self, horizonte_dias: int = 5) -> Dict:
        """Analisa se é hora de comprar ou vender baseado em predições do modelo"""
        if self.ensemble is None or not self.ensemble.trained:
            raise ValueError("Sistema não foi treinado")
        
        if self.backtest.dados_historicos is None:
            raise ValueError("Dados históricos não disponíveis")
        
        # Obter últimos dados para predição
        dados = self.backtest.dados_historicos.dropna()
        if len(dados) < 60:
            raise ValueError("Dados insuficientes para predição")
        
        # Preparar última janela temporal
        features = dados[self.features_selecionadas].values
        janela_temporal = 60
        
        if len(features) < janela_temporal:
            raise ValueError(f"Necessário pelo menos {janela_temporal} dias de dados")
        
        # Última janela para predição
        X_pred = features[-janela_temporal:].reshape(1, janela_temporal, len(self.features_selecionadas))
        
        # Fazer predição de retorno futuro
        predicao_retorno = self.fazer_predicoes(X_pred)[0][0]
        
        # Obter dados atuais
        preco_atual = dados['Close'].iloc[-1]
        
        # Calcular preço previsto
        preco_previsto = preco_atual * (1 + predicao_retorno)
        
        # Calcular indicadores técnicos atuais
        rsi_atual = dados['RSI'].iloc[-1] if 'RSI' in dados.columns else 50
        macd_atual = dados['MACD'].iloc[-1] if 'MACD' in dados.columns else 0
        
        # Análise de tendência
        sma_20 = dados['SMA_20'].iloc[-1] if 'SMA_20' in dados.columns else preco_atual
        sma_50 = dados['SMA_50'].iloc[-1] if 'SMA_50' in dados.columns else preco_atual
        
        # Calcular força do sinal
        forca_sinal = 0
        razoes = []
        
        # Sinal baseado na predição
        if predicao_retorno > 0.02:  # Retorno previsto > 2%
            forca_sinal += 3
            razoes.append(f"Retorno previsto positivo: {predicao_retorno*100:.2f}%")
        elif predicao_retorno < -0.02:  # Retorno previsto < -2%
            forca_sinal -= 3
            razoes.append(f"Retorno previsto negativo: {predicao_retorno*100:.2f}%")
        
        # Sinal baseado em RSI
        if rsi_atual < 30:
            forca_sinal += 2
            razoes.append("RSI indica sobrevenda (oportunidade de compra)")
        elif rsi_atual > 70:
            forca_sinal -= 2
            razoes.append("RSI indica sobrecompra (oportunidade de venda)")
        
        # Sinal baseado em MACD
        if macd_atual > 0:
            forca_sinal += 1
            razoes.append("MACD positivo (tendência de alta)")
        else:
            forca_sinal -= 1
            razoes.append("MACD negativo (tendência de baixa)")
        
        # Sinal baseado em médias móveis
        if preco_atual > sma_20 > sma_50:
            forca_sinal += 2
            razoes.append("Preço acima das médias móveis (tendência de alta)")
        elif preco_atual < sma_20 < sma_50:
            forca_sinal -= 2
            razoes.append("Preço abaixo das médias móveis (tendência de baixa)")
        
        # Determinar recomendação
        if forca_sinal >= 5:
            recomendacao = "COMPRAR FORTE"
            confianca = min(90, 60 + abs(forca_sinal) * 5)
        elif forca_sinal >= 3:
            recomendacao = "COMPRAR"
            confianca = min(80, 50 + abs(forca_sinal) * 5)
        elif forca_sinal <= -5:
            recomendacao = "VENDER FORTE"
            confianca = min(90, 60 + abs(forca_sinal) * 5)
        elif forca_sinal <= -3:
            recomendacao = "VENDER"
            confianca = min(80, 50 + abs(forca_sinal) * 5)
        else:
            recomendacao = "MANTER"
            confianca = 50
        
        # Ajustar confiança baseado na qualidade do modelo
        metricas = self.obter_metricas()
        r2_score = metricas['r2_score']
        if r2_score < 0.3:
            confianca = max(30, confianca * 0.7)  # Reduzir confiança se modelo ruim
        
        return {
            'recomendacao': recomendacao,
            'forca_sinal': forca_sinal,
            'confianca': confianca,
            'preco_atual': float(preco_atual),
            'preco_previsto': float(preco_previsto),
            'retorno_previsto': float(predicao_retorno * 100),
            'razoes': razoes,
            'indicadores_atuais': {
                'rsi': float(rsi_atual),
                'macd': float(macd_atual),
                'sma_20': float(sma_20),
                'sma_50': float(sma_50)
            },
            'qualidade_modelo': self._classificar_qualidade(r2_score),
            'r2_score': float(r2_score)
        }
    
    def calcular_risco_manter_acao(self, periodo_dias: int = 30) -> Dict:
        """Calcula o risco de manter a ação baseado em dados reais históricos"""
        if self.backtest.dados_historicos is None:
            raise ValueError("Dados históricos não disponíveis")
        
        dados = self.backtest.dados_historicos.dropna()
        if len(dados) < periodo_dias:
            raise ValueError(f"Dados insuficientes (necessário {periodo_dias} dias)")
        
        # Calcular métricas de risco usando dados reais
        precos = dados['Close']
        retornos = precos.pct_change().dropna()
        
        # Volatilidade (risco de variação)
        volatilidade = retornos.std() * np.sqrt(252)  # Anualizada
        volatilidade_periodo = retornos.tail(periodo_dias).std() * np.sqrt(252)
        
        # Maximum Drawdown (risco de perda máxima)
        peak = precos.expanding(min_periods=1).max()
        drawdown = (precos / peak - 1.0)
        max_drawdown = drawdown.min() * 100
        
        # Value at Risk (VaR) - 95% de confiança
        var_95 = np.percentile(retornos.tail(periodo_dias), 5) * 100
        
        # Sharpe Ratio (ajustado para risco)
        retorno_medio = retornos.mean() * 252
        sharpe_ratio = (retorno_medio - 0.1375) / volatilidade if volatilidade > 0 else 0
        
        # Beta (se disponível nos dados)
        beta = None
        if 'Beta' in dados.columns:
            beta = dados['Beta'].iloc[-1]
        
        # Calcular probabilidade de perda
        retornos_negativos = retornos[retornos < 0]
        prob_perda = len(retornos_negativos) / len(retornos) * 100 if len(retornos) > 0 else 0
        
        # Classificar nível de risco
        score_risco = 0
        
        # Volatilidade
        if volatilidade_periodo > 0.40:
            score_risco += 3
        elif volatilidade_periodo > 0.30:
            score_risco += 2
        elif volatilidade_periodo > 0.20:
            score_risco += 1
        
        # Drawdown
        if max_drawdown < -30:
            score_risco += 3
        elif max_drawdown < -20:
            score_risco += 2
        elif max_drawdown < -10:
            score_risco += 1
        
        # Probabilidade de perda
        if prob_perda > 60:
            score_risco += 2
        elif prob_perda > 50:
            score_risco += 1
        
        # Classificação final
        if score_risco >= 7:
            nivel_risco = "MUITO ALTO"
            cor_risco = "🔴"
        elif score_risco >= 5:
            nivel_risco = "ALTO"
            cor_risco = "🟠"
        elif score_risco >= 3:
            nivel_risco = "MÉDIO"
            cor_risco = "🟡"
        elif score_risco >= 1:
            nivel_risco = "BAIXO"
            cor_risco = "🟢"
        else:
            nivel_risco = "MUITO BAIXO"
            cor_risco = "🟢"
        
        # Recomendação baseada no risco
        if score_risco >= 7:
            recomendacao_risco = "REDUZIR POSIÇÃO IMEDIATAMENTE"
        elif score_risco >= 5:
            recomendacao_risco = "REDUZIR POSIÇÃO"
        elif score_risco >= 3:
            recomendacao_risco = "MANTER COM CUIDADO"
        else:
            recomendacao_risco = "MANTER POSIÇÃO"
        
        return {
            'nivel_risco': nivel_risco,
            'score_risco': score_risco,
            'cor_risco': cor_risco,
            'recomendacao_risco': recomendacao_risco,
            'metricas': {
                'volatilidade_anual': float(volatilidade * 100),
                'volatilidade_periodo': float(volatilidade_periodo * 100),
                'max_drawdown': float(max_drawdown),
                'var_95': float(var_95),
                'sharpe_ratio': float(sharpe_ratio),
                'probabilidade_perda': float(prob_perda),
                'beta': float(beta) if beta is not None else None
            },
            'analise': {
                'volatilidade_alta': volatilidade_periodo > 0.30,
                'drawdown_severo': max_drawdown < -20,
                'alta_prob_perda': prob_perda > 55,
                'sharpe_negativo': sharpe_ratio < 0
            }
        }
    
    def gerar_analise_completa(self, horizonte_dias: int = 5) -> Dict:
        """Gera análise completa com predições, sinais e risco baseado em dados reais"""
        if self.ensemble is None or not self.ensemble.trained:
            raise ValueError("Sistema não foi treinado")
        
        # Análise de sinal compra/venda
        sinal = self.analisar_sinal_compra_venda(horizonte_dias)
        
        # Análise de risco
        risco = self.calcular_risco_manter_acao()
        
        # Combinar análises
        recomendacao_final = self._combinar_recomendacoes(sinal, risco)
        
        # Métricas do modelo
        metricas = self.obter_metricas()
        
        return {
            'ticker': self.ticker,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sinal_compra_venda': sinal,
            'risco_manter': risco,
            'recomendacao_final': recomendacao_final,
            'qualidade_modelo': {
                'r2_score': float(metricas['r2_score']),
                'rmse': float(metricas['rmse']),
                'classificacao': self._classificar_qualidade(metricas['r2_score'])
            },
            'resumo': {
                'acao': recomendacao_final['acao'],
                'confianca': recomendacao_final['confianca'],
                'nivel_risco': risco['nivel_risco'],
                'retorno_previsto': sinal['retorno_previsto']
            }
        }
    
    def _combinar_recomendacoes(self, sinal: Dict, risco: Dict) -> Dict:
        """Combina sinal de compra/venda com análise de risco"""
        recomendacao_sinal = sinal['recomendacao']
        score_risco = risco['score_risco']
        
        # Lógica de combinação
        if recomendacao_sinal == "COMPRAR FORTE":
            if score_risco <= 2:
                acao_final = "COMPRAR FORTE"
                confianca = min(90, sinal['confianca'] + 10)
            elif score_risco <= 4:
                acao_final = "COMPRAR"
                confianca = sinal['confianca']
            else:
                acao_final = "AGUARDAR (risco alto)"
                confianca = max(30, sinal['confianca'] - 20)
        
        elif recomendacao_sinal == "COMPRAR":
            if score_risco <= 3:
                acao_final = "COMPRAR"
                confianca = sinal['confianca']
            elif score_risco <= 5:
                acao_final = "COMPRAR COM CUIDADO"
                confianca = max(40, sinal['confianca'] - 15)
            else:
                acao_final = "AGUARDAR"
                confianca = max(30, sinal['confianca'] - 25)
        
        elif recomendacao_sinal == "VENDER FORTE":
            acao_final = "VENDER FORTE"
            confianca = min(95, sinal['confianca'] + 10)
        
        elif recomendacao_sinal == "VENDER":
            acao_final = "VENDER"
            confianca = sinal['confianca']
        
        else:  # MANTER
            if score_risco >= 6:
                acao_final = "CONSIDERAR VENDA (risco alto)"
                confianca = 60
            elif score_risco >= 4:
                acao_final = "MANTER COM CUIDADO"
                confianca = 50
            else:
                acao_final = "MANTER"
                confianca = 55
        
        return {
            'acao': acao_final,
            'confianca': confianca,
            'razoes': {
                'sinal': sinal['razoes'],
                'risco': risco['recomendacao_risco']
            }
        }

def exemplo_uso():
    """Exemplo de uso do sistema de deep learning"""
    print("🚀 EXEMPLO DO SISTEMA DE DEEP LEARNING")
    print("=" * 50)
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow não está disponível")
        print("💡 Instale: pip install tensorflow")
        return
    
    # Configurar sistema
    ticker = "PETR4"
    sistema = SistemaDeepLearning(ticker)
    
    try:
        # Configurar sistema
        X_train, X_test, y_train, y_test = sistema.configurar_sistema()
        
        # Treinar sistema
        resultados = sistema.treinar_sistema(X_train, y_train, X_test, y_test, epochs=50)
        
        # Gerar relatório
        relatorio = sistema.gerar_relatorio()
        
        print("\n📊 RELATÓRIO DO SISTEMA")
        print("=" * 30)
        print(f"Ticker: {relatorio['ticker']}")
        print(f"Target: {relatorio['target']}")
        print(f"Modelos: {', '.join(relatorio['modelos_ensemble'])}")
        print(f"R² Score: {relatorio['metricas']['r2_score']:.4f}")
        print(f"RMSE: {relatorio['metricas']['rmse']:.4f}")
        print(f"MAE: {relatorio['metricas']['mae']:.4f}")
        print(f"Qualidade: {relatorio['qualidade_modelo']}")
        
        # Fazer predições
        predicoes = sistema.fazer_predicoes(X_test[:10])
        print(f"\n🔮 Predições (primeiras 10): {predicoes.flatten()[:5]}")
        
        # Análise completa com sinais de compra/venda e risco
        print("\n" + "=" * 50)
        print("📊 ANÁLISE COMPLETA - COMPRA/VENDA E RISCO")
        print("=" * 50)
        
        analise_completa = sistema.gerar_analise_completa()
        
        print(f"\n🎯 RECOMENDAÇÃO FINAL: {analise_completa['resumo']['acao']}")
        print(f"📈 Confiança: {analise_completa['resumo']['confianca']:.1f}%")
        print(f"⚠️  Nível de Risco: {analise_completa['resumo']['nivel_risco']}")
        print(f"💰 Retorno Previsto: {analise_completa['resumo']['retorno_previsto']:.2f}%")
        
        print("\n📊 SINAL DE COMPRA/VENDA:")
        sinal = analise_completa['sinal_compra_venda']
        print(f"  Recomendação: {sinal['recomendacao']}")
        print(f"  Força do Sinal: {sinal['forca_sinal']}")
        print(f"  Preço Atual: R$ {sinal['preco_atual']:.2f}")
        print(f"  Preço Previsto: R$ {sinal['preco_previsto']:.2f}")
        print(f"  Razões:")
        for razao in sinal['razoes']:
            print(f"    - {razao}")
        
        print("\n⚠️  ANÁLISE DE RISCO:")
        risco = analise_completa['risco_manter']
        print(f"  Nível: {risco['cor_risco']} {risco['nivel_risco']}")
        print(f"  Score de Risco: {risco['score_risco']}/10")
        print(f"  Recomendação: {risco['recomendacao_risco']}")
        print(f"  Volatilidade: {risco['metricas']['volatilidade_periodo']:.2f}%")
        print(f"  Max Drawdown: {risco['metricas']['max_drawdown']:.2f}%")
        print(f"  VaR 95%: {risco['metricas']['var_95']:.2f}%")
        print(f"  Sharpe Ratio: {risco['metricas']['sharpe_ratio']:.2f}")
        print(f"  Probabilidade de Perda: {risco['metricas']['probabilidade_perda']:.1f}%")
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    exemplo_uso()
