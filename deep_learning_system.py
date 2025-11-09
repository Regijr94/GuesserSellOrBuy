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
    
    def _adicionar_features_fundamentais(self, dados: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features fundamentais calculadas com janela de 1 ano a partir do primeiro dia do mês"""
        print("Calculando indicadores fundamentais (PL e PVP) com janela de 1 ano...")
        
        # Inicializar colunas
        dados['P_L_ratio'] = np.nan
        dados['P_VP_ratio'] = np.nan
        dados['ROE'] = np.nan
        dados['Dividend_Yield'] = np.nan
        dados['Debt_to_Equity'] = np.nan
        dados['Current_Ratio'] = np.nan
        
        try:
            # Obter objeto ticker
            acao = yf.Ticker(f"{self.ticker}.SA")
            info = acao.info
            
            # Obter dados financeiros anuais
            financials = acao.financials
            balance_sheet = acao.balance_sheet
            
            # Extrair informações básicas
            shares_outstanding = info.get('sharesOutstanding')
            if shares_outstanding is None or shares_outstanding <= 0:
                shares_outstanding = info.get('impliedSharesOutstanding')
            
            # Para cada data no DataFrame, calcular indicadores baseados no primeiro dia do mês
            meses_processados = {}
            
            for idx, data in enumerate(dados.index):
                # Obter primeiro dia do mês
                primeiro_dia_mes = data.replace(day=1)
                
                # Se já processamos este mês, reutilizar valores
                if primeiro_dia_mes in meses_processados:
                    dados.loc[data, 'P_L_ratio'] = meses_processados[primeiro_dia_mes]['P_L_ratio']
                    dados.loc[data, 'P_VP_ratio'] = meses_processados[primeiro_dia_mes]['P_VP_ratio']
                    dados.loc[data, 'ROE'] = meses_processados[primeiro_dia_mes]['ROE']
                    dados.loc[data, 'Dividend_Yield'] = meses_processados[primeiro_dia_mes]['Dividend_Yield']
                    continue
                
                # Calcular data de início (1 ano antes do primeiro dia do mês)
                data_inicio_janela = primeiro_dia_mes - timedelta(days=365)
                
                # Obter preço no primeiro dia do mês (ou próximo dia útil)
                try:
                    # Tentar obter preço no primeiro dia do mês
                    preco_data = dados.loc[primeiro_dia_mes:primeiro_dia_mes + timedelta(days=5), 'Close']
                    if len(preco_data) > 0:
                        preco = preco_data.iloc[0]
                    else:
                        preco = dados.loc[data, 'Close']
                except:
                    preco = dados.loc[data, 'Close']
                
                # Calcular PL (Preço/Lucro) usando janela de 1 ano
                pl_ratio = None
                if financials is not None and not financials.empty:
                    # Procurar lucro líquido no período de 1 ano antes do primeiro dia do mês
                    net_income = None
                    # Ordenar colunas por data (mais recente primeiro)
                    colunas_ordenadas = sorted([col for col in financials.columns if col <= primeiro_dia_mes], reverse=True)
                    
                    for col in colunas_ordenadas:
                        # Verificar se a data está dentro da janela de 1 ano
                        if col >= data_inicio_janela:
                            if 'Net Income' in financials.index:
                                try:
                                    net_income = financials.loc['Net Income', col]
                                except:
                                    pass
                            elif 'NetIncome' in financials.index:
                                try:
                                    net_income = financials.loc['NetIncome', col]
                                except:
                                    pass
                            # Não usar estimativas - apenas dados reais
                            
                            if net_income is not None:
                                break
                    
                    if net_income is not None and net_income > 0 and shares_outstanding and shares_outstanding > 0:
                        eps = net_income / shares_outstanding
                        if eps > 0:
                            pl_ratio = preco / eps
                
                # Se não conseguiu calcular PL, tentar usar info do yfinance
                if pl_ratio is None or not (0 < pl_ratio < 1000):
                    trailing_pe = info.get('trailingPE')
                    if trailing_pe and 0 < trailing_pe < 1000:
                        pl_ratio = trailing_pe
                
                # Calcular PVP (Preço/Valor Patrimonial) usando janela de 1 ano
                pvp_ratio = None
                if balance_sheet is not None and not balance_sheet.empty:
                    # Procurar patrimônio líquido no período de 1 ano antes do primeiro dia do mês
                    total_equity = None
                    # Ordenar colunas por data (mais recente primeiro)
                    colunas_ordenadas = sorted([col for col in balance_sheet.columns if col <= primeiro_dia_mes], reverse=True)
                    
                    for col in colunas_ordenadas:
                        # Verificar se a data está dentro da janela de 1 ano
                        if col >= data_inicio_janela:
                            if 'Stockholders Equity' in balance_sheet.index:
                                try:
                                    total_equity = balance_sheet.loc['Stockholders Equity', col]
                                except:
                                    pass
                            elif 'Total Stockholder Equity' in balance_sheet.index:
                                try:
                                    total_equity = balance_sheet.loc['Total Stockholder Equity', col]
                                except:
                                    pass
                            
                            if total_equity is not None:
                                break
                    
                    if total_equity is not None and total_equity > 0 and shares_outstanding and shares_outstanding > 0:
                        vpa = total_equity / shares_outstanding
                        if vpa > 0:
                            pvp_ratio = preco / vpa
                
                # Se não conseguiu calcular PVP, tentar usar info do yfinance
                if pvp_ratio is None or not (0 < pvp_ratio < 50):
                    price_to_book = info.get('priceToBook')
                    if price_to_book and 0 < price_to_book < 50:
                        pvp_ratio = price_to_book
                
                # Calcular ROE usando janela de 1 ano
                roe = None
                if financials is not None and balance_sheet is not None:
                    net_income = None
                    total_equity = None
                    
                    # Procurar lucro líquido na janela de 1 ano
                    colunas_fin = sorted([col for col in financials.columns if col <= primeiro_dia_mes], reverse=True)
                    for col in colunas_fin:
                        if col >= data_inicio_janela:
                            try:
                                if 'Net Income' in financials.index:
                                    net_income = financials.loc['Net Income', col]
                                elif 'NetIncome' in financials.index:
                                    net_income = financials.loc['NetIncome', col]
                            except:
                                pass
                            if net_income is not None:
                                break
                    
                    # Procurar patrimônio líquido na janela de 1 ano
                    colunas_bs = sorted([col for col in balance_sheet.columns if col <= primeiro_dia_mes], reverse=True)
                    for col in colunas_bs:
                        if col >= data_inicio_janela:
                            try:
                                if 'Stockholders Equity' in balance_sheet.index:
                                    total_equity = balance_sheet.loc['Stockholders Equity', col]
                                elif 'Total Stockholder Equity' in balance_sheet.index:
                                    total_equity = balance_sheet.loc['Total Stockholder Equity', col]
                            except:
                                pass
                            if total_equity is not None:
                                break
                    
                    if net_income is not None and total_equity is not None and total_equity > 0:
                        roe = net_income / total_equity
                
                if roe is None:
                    roe_info = info.get('returnOnEquity')
                    if roe_info and -1 < roe_info < 10:
                        roe = roe_info
                
                # Dividend Yield
                dividend_yield = info.get('dividendYield')
                if dividend_yield and not (0 <= dividend_yield <= 0.5):
                    dividend_yield = None
                
                # Armazenar valores calculados
                valores = {
                    'P_L_ratio': pl_ratio if pl_ratio and 0 < pl_ratio < 1000 else np.nan,
                    'P_VP_ratio': pvp_ratio if pvp_ratio and 0 < pvp_ratio < 50 else np.nan,
                    'ROE': roe if roe and -1 < roe < 10 else np.nan,
                    'Dividend_Yield': dividend_yield if dividend_yield else np.nan
                }
                
                meses_processados[primeiro_dia_mes] = valores
                
                # Atribuir valores ao DataFrame
                dados.loc[data, 'P_L_ratio'] = valores['P_L_ratio']
                dados.loc[data, 'P_VP_ratio'] = valores['P_VP_ratio']
                dados.loc[data, 'ROE'] = valores['ROE']
                dados.loc[data, 'Dividend_Yield'] = valores['Dividend_Yield']
            
            # Preencher valores NaN restantes com forward fill (usar último valor conhecido)
            dados['P_L_ratio'] = dados['P_L_ratio'].ffill().bfill()
            dados['P_VP_ratio'] = dados['P_VP_ratio'].ffill().bfill()
            dados['ROE'] = dados['ROE'].ffill().bfill()
            dados['Dividend_Yield'] = dados['Dividend_Yield'].ffill().bfill()
            
            # Calcular Debt_to_Equity e Current_Ratio usando dados reais históricos
            for data in dados.index:
                primeiro_dia_mes = data.replace(day=1)
                data_inicio_janela = primeiro_dia_mes - timedelta(days=365)
                
                # Se já processamos este mês e já temos Debt_to_Equity, reutilizar valores
                if primeiro_dia_mes in meses_processados and 'Debt_to_Equity' in meses_processados[primeiro_dia_mes]:
                    dados.loc[data, 'Debt_to_Equity'] = meses_processados[primeiro_dia_mes]['Debt_to_Equity']
                    dados.loc[data, 'Current_Ratio'] = meses_processados[primeiro_dia_mes]['Current_Ratio']
                    continue
                
                # Garantir que o dicionário existe para este mês
                if primeiro_dia_mes not in meses_processados:
                    meses_processados[primeiro_dia_mes] = {}
                
                # Calcular Debt_to_Equity usando dados reais
                debt_to_equity = None
                if balance_sheet is not None and not balance_sheet.empty:
                    total_debt = None
                    total_equity = None
                    
                    colunas_bs = sorted([col for col in balance_sheet.columns if col <= primeiro_dia_mes], reverse=True)
                    for col in colunas_bs:
                        if col >= data_inicio_janela:
                            try:
                                # Tentar obter dívida total
                                if 'Total Debt' in balance_sheet.index:
                                    total_debt = balance_sheet.loc['Total Debt', col]
                                elif 'Total Liabilities' in balance_sheet.index:
                                    total_debt = balance_sheet.loc['Total Liabilities', col]
                                
                                # Obter patrimônio líquido
                                if 'Stockholders Equity' in balance_sheet.index:
                                    total_equity = balance_sheet.loc['Stockholders Equity', col]
                                elif 'Total Stockholder Equity' in balance_sheet.index:
                                    total_equity = balance_sheet.loc['Total Stockholder Equity', col]
                                
                                if total_debt is not None and total_equity is not None and total_equity > 0:
                                    debt_to_equity = total_debt / total_equity
                                    break
                            except:
                                pass
                
                # Se não conseguiu calcular, tentar usar info do yfinance
                if debt_to_equity is None:
                    debt_to_equity = info.get('debtToEquity')
                    if debt_to_equity and not (0 <= debt_to_equity <= 10):
                        debt_to_equity = None
                
                # Calcular Current_Ratio usando dados reais
                current_ratio = None
                if balance_sheet is not None and not balance_sheet.empty:
                    current_assets = None
                    current_liabilities = None
                    
                    colunas_bs = sorted([col for col in balance_sheet.columns if col <= primeiro_dia_mes], reverse=True)
                    for col in colunas_bs:
                        if col >= data_inicio_janela:
                            try:
                                if 'Current Assets' in balance_sheet.index:
                                    current_assets = balance_sheet.loc['Current Assets', col]
                                elif 'Total Current Assets' in balance_sheet.index:
                                    current_assets = balance_sheet.loc['Total Current Assets', col]
                                
                                if 'Current Liabilities' in balance_sheet.index:
                                    current_liabilities = balance_sheet.loc['Current Liabilities', col]
                                elif 'Total Current Liabilities' in balance_sheet.index:
                                    current_liabilities = balance_sheet.loc['Total Current Liabilities', col]
                                
                                if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
                                    current_ratio = current_assets / current_liabilities
                                    break
                            except:
                                pass
                
                # Se não conseguiu calcular, tentar usar info do yfinance
                if current_ratio is None:
                    current_ratio = info.get('currentRatio')
                    if current_ratio and not (0 < current_ratio <= 20):
                        current_ratio = None
                
                # Armazenar valores no dicionário de meses processados
                meses_processados[primeiro_dia_mes]['Debt_to_Equity'] = debt_to_equity if debt_to_equity else np.nan
                meses_processados[primeiro_dia_mes]['Current_Ratio'] = current_ratio if current_ratio else np.nan
                
                dados.loc[data, 'Debt_to_Equity'] = meses_processados[primeiro_dia_mes]['Debt_to_Equity']
                dados.loc[data, 'Current_Ratio'] = meses_processados[primeiro_dia_mes]['Current_Ratio']
            
            # Preencher NaN apenas com forward/backward fill (dados reais anteriores/posteriores)
            # NÃO usar valores padrão simulados
            dados['Debt_to_Equity'] = dados['Debt_to_Equity'].ffill().bfill()
            dados['Current_Ratio'] = dados['Current_Ratio'].ffill().bfill()
            
            # Estatísticas dos indicadores calculados
            pl_mean = dados['P_L_ratio'].mean()
            pvp_mean = dados['P_VP_ratio'].mean()
            print(f"Indicadores calculados com dados reais:")
            print(f"  PL médio: {pl_mean:.2f}" if not np.isnan(pl_mean) else "  PL médio: N/A")
            print(f"  PVP médio: {pvp_mean:.2f}" if not np.isnan(pvp_mean) else "  PVP médio: N/A")
            print(f"  Dados reais: {dados[['P_L_ratio', 'P_VP_ratio', 'ROE', 'Dividend_Yield']].notna().sum().sum()} valores calculados")
            
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
                'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR',
                'Volume_SMA', 'Price_Change', 'Volatility', 'P_L_ratio',
                'P_VP_ratio', 'ROE', 'Dividend_Yield', 'Debt_to_Equity'
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
        nivel_risco = risco['nivel_risco']
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
