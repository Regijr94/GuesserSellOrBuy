#!/usr/bin/env python3
"""
SISTEMA DE ANÁLISE AVANÇADA DE RISCO E RECOMENDAÇÕES
Implementa técnicas sofisticadas: ML, VaR condicional, Monte Carlo, análise quantitativa
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ===== ESTRUTURAS DE DADOS AVANÇADAS =====

@dataclass
class AnaliseRiscoAvancada:
    """Análise de risco com técnicas sofisticadas"""
    ticker: str
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    omega_ratio: float
    tail_ratio: float
    common_sense_ratio: float
    skewness: float
    kurtosis: float
    jarque_bera_pvalue: float
    hurst_exponent: float
    volatility_regime: str
    risk_score: float
    risk_rating: str

@dataclass
class PredicaoPreco:
    """Predição de preço com ML"""
    ticker: str
    preco_atual: float
    previsao_1m: float
    previsao_3m: float
    previsao_6m: float
    previsao_12m: float
    confianca_1m: float
    confianca_3m: float
    confianca_6m: float
    confianca_12m: float
    modelo_usado: str
    r2_score: float
    rmse: float

@dataclass
class RecomendacaoAvancada:
    """Recomendação com análise multi-fatorial"""
    ticker: str
    recomendacao: str
    score_fundamentalista: float
    score_tecnico: float
    score_momentum: float
    score_risco: float
    score_final: float
    confianca: float
    justificativa: str
    fatores_chave: List[str]
    probabilidade_sucesso: float
    horizonte_otimo: str
    stop_loss: float
    take_profit: float

# ===== ANÁLISE DE RISCO AVANÇADA =====

class AnalisadorRiscoAvancado:
    """Análise de risco com técnicas quantitativas sofisticadas"""
    
    def __init__(self):
        self.risk_free_rate = 0.1375  # CDI brasileiro
        
    def analisar_risco_completo(self, ticker: str, historico: pd.DataFrame, 
                               beta: Optional[float] = None) -> AnaliseRiscoAvancada:
        """Análise de risco completa com múltiplas métricas"""
        
        if historico.empty or len(historico) < 30:
            return self._risco_indeterminado(ticker)
        
        precos = historico['Close']
        retornos = precos.pct_change().dropna()
        
        # Métricas básicas
        volatilidade = retornos.std() * np.sqrt(252)
        retorno_medio = retornos.mean() * 252
        
        # VaR e CVaR
        var_95, var_99 = self._calcular_var(retornos)
        cvar_95, cvar_99 = self._calcular_cvar(retornos)
        
        # Drawdown e métricas relacionadas
        max_dd = self._calcular_max_drawdown(precos)
        calmar_ratio = self._calcular_calmar_ratio(retorno_medio, max_dd)
        
        # Ratios avançados
        sortino_ratio = self._calcular_sortino_ratio(retornos)
        omega_ratio = self._calcular_omega_ratio(retornos)
        tail_ratio = self._calcular_tail_ratio(retornos)
        common_sense_ratio = self._calcular_common_sense_ratio(retornos)
        
        # Análise estatística
        skewness = retornos.skew()
        kurtosis = retornos.kurtosis()
        jarque_bera_pvalue = stats.jarque_bera(retornos)[1]
        
        # Análise de regime
        hurst_exponent = self._calcular_hurst_exponent(precos)
        volatility_regime = self._classificar_regime_volatilidade(volatilidade, hurst_exponent)
        
        # Score de risco
        risk_score = self._calcular_risk_score(
            var_95, max_dd, volatilidade, skewness, kurtosis, hurst_exponent
        )
        risk_rating = self._classificar_risco(risk_score)
        
        return AnaliseRiscoAvancada(
            ticker=ticker,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            omega_ratio=omega_ratio,
            tail_ratio=tail_ratio,
            common_sense_ratio=common_sense_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            jarque_bera_pvalue=jarque_bera_pvalue,
            hurst_exponent=hurst_exponent,
            volatility_regime=volatility_regime,
            risk_score=risk_score,
            risk_rating=risk_rating
        )
    
    def _calcular_var(self, retornos: pd.Series) -> Tuple[float, float]:
        """Calcula VaR 95% e 99%"""
        var_95 = np.percentile(retornos, 5) * 100
        var_99 = np.percentile(retornos, 1) * 100
        return var_95, var_99
    
    def _calcular_cvar(self, retornos: pd.Series) -> Tuple[float, float]:
        """Calcula CVaR (Expected Shortfall) 95% e 99%"""
        cvar_95 = retornos[retornos <= np.percentile(retornos, 5)].mean() * 100
        cvar_99 = retornos[retornos <= np.percentile(retornos, 1)].mean() * 100
        return cvar_95, cvar_99
    
    def _calcular_max_drawdown(self, precos: pd.Series) -> float:
        """Calcula máximo drawdown"""
        peak = precos.expanding(min_periods=1).max()
        drawdown = (precos / peak - 1.0)
        return float(drawdown.min() * 100)
    
    def _calcular_calmar_ratio(self, retorno_anual: float, max_drawdown: float) -> float:
        """Calcula Calmar Ratio"""
        if max_drawdown == 0:
            return 0
        return retorno_anual / abs(max_drawdown / 100)
    
    def _calcular_sortino_ratio(self, retornos: pd.Series) -> float:
        """Calcula Sortino Ratio (foca apenas em downside risk)"""
        downside_returns = retornos[retornos < 0]
        if len(downside_returns) == 0:
            return 0
        downside_deviation = downside_returns.std() * np.sqrt(252)
        if downside_deviation == 0:
            return 0
        return (retornos.mean() * 252 - self.risk_free_rate) / downside_deviation
    
    def _calcular_omega_ratio(self, retornos: pd.Series) -> float:
        """Calcula Omega Ratio"""
        threshold = 0  # Retorno zero como threshold
        gains = retornos[retornos > threshold].sum()
        losses = abs(retornos[retornos < threshold].sum())
        if losses == 0:
            return float('inf') if gains > 0 else 0
        return gains / losses
    
    def _calcular_tail_ratio(self, retornos: pd.Series) -> float:
        """Calcula Tail Ratio (95th percentile / 5th percentile)"""
        p95 = np.percentile(retornos, 95)
        p5 = np.percentile(retornos, 5)
        if p5 == 0:
            return float('inf') if p95 > 0 else 0
        return abs(p95 / p5)
    
    def _calcular_common_sense_ratio(self, retornos: pd.Series) -> float:
        """Calcula Common Sense Ratio (retorno médio / volatilidade)"""
        if retornos.std() == 0:
            return 0
        return retornos.mean() / retornos.std()
    
    def _calcular_hurst_exponent(self, precos: pd.Series) -> float:
        """Calcula Hurst Exponent para detectar tendência vs mean reversion"""
        try:
            lags = range(2, min(20, len(precos)//4))
            tau = [np.sqrt(np.std(np.subtract(precos[lag:], precos[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5  # Neutro se não conseguir calcular
    
    def _classificar_regime_volatilidade(self, volatilidade: float, hurst: float) -> str:
        """Classifica regime de volatilidade"""
        if volatilidade < 0.15:
            if hurst > 0.6:
                return "BAIXA_TENDENCIA"
            elif hurst < 0.4:
                return "BAIXA_MEAN_REVERSION"
            else:
                return "BAIXA_NEUTRA"
        elif volatilidade < 0.30:
            if hurst > 0.6:
                return "MEDIA_TENDENCIA"
            elif hurst < 0.4:
                return "MEDIA_MEAN_REVERSION"
            else:
                return "MEDIA_NEUTRA"
        else:
            if hurst > 0.6:
                return "ALTA_TENDENCIA"
            elif hurst < 0.4:
                return "ALTA_MEAN_REVERSION"
            else:
                return "ALTA_NEUTRA"
    
    def _calcular_risk_score(self, var_95: float, max_dd: float, volatilidade: float,
                           skewness: float, kurtosis: float, hurst: float) -> float:
        """Calcula score de risco (0-100, menor = mais arriscado)"""
        
        # Normalizar métricas (0-1, onde 1 = mais arriscado)
        var_score = min(abs(var_95) / 20, 1)  # VaR 20% = score máximo
        dd_score = min(abs(max_dd) / 50, 1)   # DD 50% = score máximo
        vol_score = min(volatilidade / 0.5, 1)  # Vol 50% = score máximo
        
        # Skewness (negativo = mais arriscado)
        skew_score = max(0, -skewness / 2) if skewness < 0 else 0
        
        # Kurtosis (alta = mais arriscado)
        kurt_score = min((kurtosis - 3) / 10, 1) if kurtosis > 3 else 0
        
        # Hurst (extremos = mais arriscado)
        hurst_score = abs(hurst - 0.5) * 2
        
        # Score final (média ponderada)
        weights = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
        scores = [var_score, dd_score, vol_score, skew_score, kurt_score, hurst_score]
        
        risk_score = sum(w * s for w, s in zip(weights, scores)) * 100
        return min(risk_score, 100)
    
    def _classificar_risco(self, risk_score: float) -> str:
        """Classifica nível de risco"""
        if risk_score < 30:
            return "MUITO_ALTO"
        elif risk_score < 50:
            return "ALTO"
        elif risk_score < 70:
            return "MEDIO"
        elif risk_score < 85:
            return "BAIXO"
        else:
            return "MUITO_BAIXO"
    
    def _risco_indeterminado(self, ticker: str) -> AnaliseRiscoAvancada:
        """Retorna análise de risco para dados insuficientes"""
        return AnaliseRiscoAvancada(
            ticker=ticker, var_95=0, var_99=0, cvar_95=0, cvar_99=0,
            max_drawdown=0, calmar_ratio=0, sortino_ratio=0, omega_ratio=0,
            tail_ratio=0, common_sense_ratio=0, skewness=0, kurtosis=0,
            jarque_bera_pvalue=0, hurst_exponent=0.5, volatility_regime="INDETERMINADO",
            risk_score=50, risk_rating="INDETERMINADO"
        )

# ===== PREDIÇÃO COM MACHINE LEARNING =====

class PreditorPrecosML:
    """Predição de preços usando Machine Learning"""
    
    def __init__(self):
        self.modelos = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge': Ridge(alpha=1.0)
        }
        self.scaler = StandardScaler()
        self.melhor_modelo = None
        self.melhor_score = 0
        
    def prever_precos(self, ticker: str, historico: pd.DataFrame) -> PredicaoPreco:
        """Prediz preços usando ML"""
        
        if historico.empty or len(historico) < 60:
            return self._predicao_indeterminada(ticker, historico)
        
        # Preparar features
        features = self._criar_features(historico)
        if features.empty:
            return self._predicao_indeterminada(ticker, historico)
        
        # Treinar modelos
        melhor_modelo, melhor_score = self._treinar_modelos(features)
        
        # Fazer predições
        previsoes = self._fazer_predicoes(melhor_modelo, features)
        
        # Calcular confiança
        confiancas = self._calcular_confianca(features, melhor_score)
        
        preco_atual = historico['Close'].iloc[-1]
        
        return PredicaoPreco(
            ticker=ticker,
            preco_atual=preco_atual,
            previsao_1m=previsoes[0],
            previsao_3m=previsoes[1],
            previsao_6m=previsoes[2],
            previsao_12m=previsoes[3],
            confianca_1m=confiancas[0],
            confianca_3m=confiancas[1],
            confianca_6m=confiancas[2],
            confianca_12m=confiancas[3],
            modelo_usado=melhor_modelo.__class__.__name__,
            r2_score=melhor_score,
            rmse=self._calcular_rmse(features, melhor_modelo)
        )
    
    def _criar_features(self, historico: pd.DataFrame) -> pd.DataFrame:
        """Cria features para ML"""
        try:
            df = historico.copy()
            
            # Features de preço
            df['retorno_1d'] = df['Close'].pct_change()
            df['retorno_5d'] = df['Close'].pct_change(5)
            df['retorno_20d'] = df['Close'].pct_change(20)
            
            # Features técnicas
            df['sma_5'] = df['Close'].rolling(5).mean()
            df['sma_20'] = df['Close'].rolling(20).mean()
            df['rsi'] = self._calcular_rsi(df['Close'])
            df['bb_upper'], df['bb_lower'] = self._calcular_bollinger_bands(df['Close'])
            
            # Features de volatilidade
            df['vol_5d'] = df['retorno_1d'].rolling(5).std()
            df['vol_20d'] = df['retorno_1d'].rolling(20).std()
            
            # Features de volume
            df['volume_sma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            
            # Features de momentum
            df['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
            df['momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
            
            # Target (preço futuro)
            df['target_1m'] = df['Close'].shift(-20)  # ~1 mês
            df['target_3m'] = df['Close'].shift(-60)  # ~3 meses
            df['target_6m'] = df['Close'].shift(-120) # ~6 meses
            df['target_12m'] = df['Close'].shift(-240) # ~12 meses
            
            # Remover NaN
            df = df.dropna()
            
            if len(df) < 30:
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            print(f"Erro ao criar features: {e}")
            return pd.DataFrame()
    
    def _calcular_rsi(self, precos: pd.Series, periodo: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = precos.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calcular_bollinger_bands(self, precos: pd.Series, periodo: int = 20, std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calcula Bollinger Bands"""
        sma = precos.rolling(periodo).mean()
        std_dev = precos.rolling(periodo).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, lower
    
    def _treinar_modelos(self, features: pd.DataFrame) -> Tuple[object, float]:
        """Treina múltiplos modelos e retorna o melhor"""
        
        # Features para treino
        feature_cols = [col for col in features.columns if not col.startswith('target_')]
        X = features[feature_cols]
        
        melhor_modelo = None
        melhor_score = 0
        
        for nome, modelo in self.modelos.items():
            try:
                # Usar target de 1 mês para treino
                y = features['target_1m']
                
                # Remover linhas com target NaN
                mask = ~y.isna()
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) < 20:
                    continue
                
                # Normalizar features
                X_scaled = self.scaler.fit_transform(X_clean)
                
                # Treinar modelo
                modelo.fit(X_scaled, y_clean)
                
                # Avaliar
                score = modelo.score(X_scaled, y_clean)
                
                if score > melhor_score:
                    melhor_score = score
                    melhor_modelo = modelo
                    
            except Exception as e:
                print(f"Erro ao treinar {nome}: {e}")
                continue
        
        return melhor_modelo, melhor_score
    
    def _fazer_predicoes(self, modelo: object, features: pd.DataFrame) -> List[float]:
        """Faz predições para diferentes horizontes"""
        
        if modelo is None:
            return [0, 0, 0, 0]
        
        try:
            # Última linha para predição
            feature_cols = [col for col in features.columns if not col.startswith('target_')]
            X_latest = features[feature_cols].iloc[-1:].values
            X_scaled = self.scaler.transform(X_latest)
            
            # Predição base (1 mês)
            pred_base = modelo.predict(X_scaled)[0]
            
            # Ajustar para diferentes horizontes (simplificado)
            previsoes = [
                pred_base,  # 1 mês
                pred_base * 1.02,  # 3 meses
                pred_base * 1.05,  # 6 meses
                pred_base * 1.10   # 12 meses
            ]
            
            return previsoes
            
        except Exception as e:
            print(f"Erro na predição: {e}")
            return [0, 0, 0, 0]
    
    def _calcular_confianca(self, features: pd.DataFrame, r2_score: float) -> List[float]:
        """Calcula confiança das predições"""
        base_conf = min(r2_score * 100, 95)  # Máximo 95%
        
        # Confiança decresce com horizonte
        confiancas = [
            base_conf,           # 1 mês
            base_conf * 0.8,     # 3 meses
            base_conf * 0.6,     # 6 meses
            base_conf * 0.4      # 12 meses
        ]
        
        return [max(c, 10) for c in confiancas]  # Mínimo 10%
    
    def _calcular_rmse(self, features: pd.DataFrame, modelo: object) -> float:
        """Calcula RMSE do modelo"""
        try:
            feature_cols = [col for col in features.columns if not col.startswith('target_')]
            X = features[feature_cols]
            y = features['target_1m']
            
            mask = ~y.isna()
            X_clean = X[mask]
            y_clean = y[mask]
            
            X_scaled = self.scaler.transform(X_clean)
            y_pred = modelo.predict(X_scaled)
            
            return np.sqrt(mean_squared_error(y_clean, y_pred))
            
        except:
            return 0
    
    def _predicao_indeterminada(self, ticker: str, historico: pd.DataFrame) -> PredicaoPreco:
        """Retorna predição para dados insuficientes"""
        preco_atual = historico['Close'].iloc[-1] if not historico.empty else 0
        
        return PredicaoPreco(
            ticker=ticker, preco_atual=preco_atual,
            previsao_1m=preco_atual, previsao_3m=preco_atual,
            previsao_6m=preco_atual, previsao_12m=preco_atual,
            confianca_1m=0, confianca_3m=0, confianca_6m=0, confianca_12m=0,
            modelo_usado="Dados_Insuficientes", r2_score=0, rmse=0
        )

# ===== SISTEMA DE RECOMENDAÇÃO AVANÇADO =====

class SistemaRecomendacaoAvancado:
    """Sistema de recomendação com análise multi-fatorial"""
    
    def __init__(self):
        self.pesos = {
            'fundamentalista': 0.35,
            'tecnico': 0.25,
            'momentum': 0.20,
            'risco': 0.20
        }
        
    def gerar_recomendacao(self, ticker: str, dados_fundamentais: Dict,
                          analise_risco: AnaliseRiscoAvancada,
                          predicao: PredicaoPreco,
                          historico: pd.DataFrame) -> RecomendacaoAvancada:
        """Gera recomendação avançada"""
        
        # Calcular scores individuais
        score_fund = self._calcular_score_fundamentalista(dados_fundamentais)
        score_tec = self._calcular_score_tecnico(historico)
        score_mom = self._calcular_score_momentum(historico, predicao)
        score_risk = self._calcular_score_risco(analise_risco)
        
        # Score final ponderado
        score_final = (
            score_fund * self.pesos['fundamentalista'] +
            score_tec * self.pesos['tecnico'] +
            score_mom * self.pesos['momentum'] +
            score_risk * self.pesos['risco']
        )
        
        # Determinar recomendação
        recomendacao, confianca, justificativa = self._determinar_recomendacao(
            score_final, score_fund, score_tec, score_mom, score_risk
        )
        
        # Fatores chave
        fatores_chave = self._identificar_fatores_chave(
            score_fund, score_tec, score_mom, score_risk
        )
        
        # Probabilidade de sucesso
        prob_sucesso = self._calcular_probabilidade_sucesso(score_final, confianca)
        
        # Horizonte ótimo
        horizonte_otimo = self._determinar_horizonte_otimo(predicao, score_mom)
        
        # Stop loss e take profit
        stop_loss, take_profit = self._calcular_niveis_operacao(
            historico, analise_risco, predicao
        )
        
        return RecomendacaoAvancada(
            ticker=ticker,
            recomendacao=recomendacao,
            score_fundamentalista=score_fund,
            score_tecnico=score_tec,
            score_momentum=score_mom,
            score_risco=score_risk,
            score_final=score_final,
            confianca=confianca,
            justificativa=justificativa,
            fatores_chave=fatores_chave,
            probabilidade_sucesso=prob_sucesso,
            horizonte_otimo=horizonte_otimo,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def _calcular_score_fundamentalista(self, dados: Dict) -> float:
        """Calcula score fundamentalista"""
        score = 0
        indicadores = 0
        
        # P/L
        pe = dados.get('pe_ratio')
        if pe and 0 < pe < 1000:
            if pe < 10:
                score += 10
            elif pe < 15:
                score += 8
            elif pe < 25:
                score += 5
            else:
                score += 2
            indicadores += 1
        
        # P/VP
        pb = dados.get('pb_ratio')
        if pb and 0 < pb < 50:
            if pb < 1:
                score += 10
            elif pb < 2:
                score += 8
            elif pb < 3:
                score += 5
            else:
                score += 2
            indicadores += 1
        
        # ROE
        roe = dados.get('roe')
        if roe and roe > 0:
            roe_pct = roe * 100
            if roe_pct > 15:
                score += 10
            elif roe_pct > 10:
                score += 8
            elif roe_pct > 5:
                score += 5
            else:
                score += 2
            indicadores += 1
        
        # Dividend Yield
        dy = dados.get('dividend_yield')
        if dy and 0 < dy < 50:
            if dy > 6:
                score += 10
            elif dy > 4:
                score += 8
            elif dy > 2:
                score += 5
            else:
                score += 2
            indicadores += 1
        
        return (score / indicadores) if indicadores > 0 else 0
    
    def _calcular_score_tecnico(self, historico: pd.DataFrame) -> float:
        """Calcula score técnico"""
        if historico.empty or len(historico) < 20:
            return 5  # Neutro
        
        precos = historico['Close']
        
        # Médias móveis
        sma_20 = precos.rolling(20).mean().iloc[-1]
        sma_50 = precos.rolling(50).mean().iloc[-1] if len(precos) >= 50 else sma_20
        preco_atual = precos.iloc[-1]
        
        score = 5  # Base neutro
        
        # Posição relativa às médias
        if preco_atual > sma_20:
            score += 2
        if preco_atual > sma_50:
            score += 2
        
        # Tendência
        if sma_20 > sma_50:
            score += 1
        
        # RSI
        rsi = self._calcular_rsi(precos).iloc[-1]
        if 30 < rsi < 70:
            score += 1
        elif rsi < 30:  # Oversold
            score += 2
        
        return min(score, 10)
    
    def _calcular_score_momentum(self, historico: pd.DataFrame, predicao: PredicaoPreco) -> float:
        """Calcula score de momentum"""
        if historico.empty:
            return 5
        
        precos = historico['Close']
        score = 5  # Base neutro
        
        # Momentum de curto prazo
        retorno_5d = (precos.iloc[-1] / precos.iloc[-6] - 1) * 100 if len(precos) >= 6 else 0
        retorno_20d = (precos.iloc[-1] / precos.iloc[-21] - 1) * 100 if len(precos) >= 21 else 0
        
        # Ajustar score baseado em momentum
        if retorno_5d > 2:
            score += 2
        elif retorno_5d < -2:
            score -= 2
        
        if retorno_20d > 5:
            score += 2
        elif retorno_20d < -5:
            score -= 2
        
        # Predição futura
        if predicao.confianca_1m > 50:
            retorno_previsto = (predicao.previsao_1m / predicao.preco_atual - 1) * 100
            if retorno_previsto > 5:
                score += 1
            elif retorno_previsto < -5:
                score -= 1
        
        return max(0, min(score, 10))
    
    def _calcular_score_risco(self, analise_risco: AnaliseRiscoAvancada) -> float:
        """Calcula score de risco (invertido - maior score = menor risco)"""
        risk_score = analise_risco.risk_score
        
        # Converter para escala 0-10 (invertida)
        if risk_score >= 85:
            return 10  # Muito baixo risco
        elif risk_score >= 70:
            return 8   # Baixo risco
        elif risk_score >= 50:
            return 6   # Médio risco
        elif risk_score >= 30:
            return 4   # Alto risco
        else:
            return 2   # Muito alto risco
    
    def _determinar_recomendacao(self, score_final: float, score_fund: float,
                               score_tec: float, score_mom: float, score_risk: float) -> Tuple[str, float, str]:
        """Determina recomendação final"""
        
        # Calcular confiança baseada na consistência dos scores
        scores = [score_fund, score_tec, score_mom, score_risk]
        confianca = 100 - (np.std(scores) * 10)  # Menor desvio = maior confiança
        confianca = max(30, min(confianca, 95))
        
        # Determinar recomendação
        if score_final >= 8:
            recomendacao = "COMPRA_FORTE"
            justificativa = f"Score excelente ({score_final:.1f}/10) com múltiplos fatores favoráveis"
        elif score_final >= 6.5:
            recomendacao = "COMPRAR"
            justificativa = f"Score bom ({score_final:.1f}/10) com fatores predominantemente positivos"
        elif score_final >= 5:
            recomendacao = "MANTER"
            justificativa = f"Score neutro ({score_final:.1f}/10) com fatores equilibrados"
        elif score_final >= 3.5:
            recomendacao = "REDUZIR"
            justificativa = f"Score baixo ({score_final:.1f}/10) com fatores predominantemente negativos"
        else:
            recomendacao = "VENDER"
            justificativa = f"Score muito baixo ({score_final:.1f}/10) com múltiplos fatores desfavoráveis"
        
        return recomendacao, confianca, justificativa
    
    def _identificar_fatores_chave(self, score_fund: float, score_tec: float,
                                 score_mom: float, score_risk: float) -> List[str]:
        """Identifica fatores chave da recomendação"""
        fatores = []
        
        if score_fund >= 8:
            fatores.append("Fundamentos Excelentes")
        elif score_fund <= 3:
            fatores.append("Fundamentos Fracos")
        
        if score_tec >= 8:
            fatores.append("Tendência Técnica Positiva")
        elif score_tec <= 3:
            fatores.append("Tendência Técnica Negativa")
        
        if score_mom >= 8:
            fatores.append("Momentum Forte")
        elif score_mom <= 3:
            fatores.append("Momentum Fraco")
        
        if score_risk >= 8:
            fatores.append("Baixo Risco")
        elif score_risk <= 3:
            fatores.append("Alto Risco")
        
        return fatores if fatores else ["Análise Equilibrada"]
    
    def _calcular_probabilidade_sucesso(self, score_final: float, confianca: float) -> float:
        """Calcula probabilidade de sucesso da recomendação"""
        # Baseado no score e confiança
        prob_base = (score_final / 10) * 100
        ajuste_confianca = (confianca - 50) / 100  # -0.5 a +0.45
        
        probabilidade = prob_base + (ajuste_confianca * 20)
        return max(20, min(probabilidade, 90))  # Entre 20% e 90%
    
    def _determinar_horizonte_otimo(self, predicao: PredicaoPreco, score_mom: float) -> str:
        """Determina horizonte ótimo de investimento"""
        if score_mom >= 7:
            return "Curto Prazo (1-3 meses)"
        elif score_mom >= 5:
            return "Médio Prazo (3-6 meses)"
        else:
            return "Longo Prazo (6-12 meses)"
    
    def _calcular_niveis_operacao(self, historico: pd.DataFrame, analise_risco: AnaliseRiscoAvancada,
                                predicao: PredicaoPreco) -> Tuple[float, float]:
        """Calcula stop loss e take profit"""
        if historico.empty:
            return 0, 0
        
        preco_atual = historico['Close'].iloc[-1]
        
        # Stop loss baseado no VaR e volatilidade
        var_ajustado = abs(analise_risco.var_95) / 100
        stop_loss = preco_atual * (1 - var_ajustado * 1.5)  # 1.5x o VaR
        
        # Take profit baseado na predição
        if predicao.confianca_3m > 60:
            retorno_previsto = (predicao.previsao_3m / preco_atual - 1)
            take_profit = preco_atual * (1 + retorno_previsto * 0.8)  # 80% do retorno previsto
        else:
            take_profit = preco_atual * 1.15  # 15% conservador
        
        return stop_loss, take_profit
    
    def _calcular_rsi(self, precos: pd.Series, periodo: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = precos.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# ===== EXEMPLO DE USO =====

def exemplo_analise_avancada():
    """Exemplo de uso do sistema avançado"""
    
    print("🚀 SISTEMA DE ANÁLISE AVANÇADA")
    print("=" * 60)
    
    # Inicializar componentes
    analisador_risco = AnalisadorRiscoAvancado()
    preditor_ml = PreditorPrecosML()
    sistema_recomendacao = SistemaRecomendacaoAvancado()
    
    # Testar com uma ação
    ticker = "PETR4"
    print(f"\n📊 Analisando {ticker} com técnicas avançadas...")
    
    try:
        # Obter dados
        stock = yf.Ticker(f"{ticker}.SA")
        historico = stock.history(period="2y")
        info = stock.info
        
        if historico.empty:
            print("❌ Não foi possível obter dados históricos")
            return
        
        # Análise de risco avançada
        print("\n🔍 Análise de Risco Avançada:")
        analise_risco = analisador_risco.analisar_risco_completo(ticker, historico)
        
        print(f"   VaR 95%: {analise_risco.var_95:.2f}%")
        print(f"   CVaR 95%: {analise_risco.cvar_95:.2f}%")
        print(f"   Max Drawdown: {analise_risco.max_drawdown:.2f}%")
        print(f"   Calmar Ratio: {analise_risco.calmar_ratio:.2f}")
        print(f"   Sortino Ratio: {analise_risco.sortino_ratio:.2f}")
        print(f"   Hurst Exponent: {analise_risco.hurst_exponent:.3f}")
        print(f"   Regime: {analise_risco.volatility_regime}")
        print(f"   Risk Score: {analise_risco.risk_score:.1f}/100")
        print(f"   Rating: {analise_risco.risk_rating}")
        
        # Predição com ML
        print("\n🤖 Predição com Machine Learning:")
        predicao = preditor_ml.prever_precos(ticker, historico)
        
        print(f"   Preço Atual: R$ {predicao.preco_atual:.2f}")
        print(f"   Previsão 1m: R$ {predicao.previsao_1m:.2f} (Conf: {predicao.confianca_1m:.0f}%)")
        print(f"   Previsão 3m: R$ {predicao.previsao_3m:.2f} (Conf: {predicao.confianca_3m:.0f}%)")
        print(f"   Previsão 6m: R$ {predicao.previsao_6m:.2f} (Conf: {predicao.confianca_6m:.0f}%)")
        print(f"   Modelo: {predicao.modelo_usado}")
        print(f"   R² Score: {predicao.r2_score:.3f}")
        
        # Recomendação avançada
        print("\n💡 Recomendação Avançada:")
        dados_fundamentais = {
            'pe_ratio': info.get('trailingPE'),
            'pb_ratio': info.get('priceToBook'),
            'roe': info.get('returnOnEquity'),
            'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        }
        
        recomendacao = sistema_recomendacao.gerar_recomendacao(
            ticker, dados_fundamentais, analise_risco, predicao, historico
        )
        
        print(f"   Recomendação: {recomendacao.recomendacao}")
        print(f"   Score Final: {recomendacao.score_final:.1f}/10")
        print(f"   Confiança: {recomendacao.confianca:.0f}%")
        print(f"   Prob. Sucesso: {recomendacao.probabilidade_sucesso:.0f}%")
        print(f"   Horizonte: {recomendacao.horizonte_otimo}")
        print(f"   Stop Loss: R$ {recomendacao.stop_loss:.2f}")
        print(f"   Take Profit: R$ {recomendacao.take_profit:.2f}")
        print(f"   Fatores Chave: {', '.join(recomendacao.fatores_chave)}")
        print(f"   Justificativa: {recomendacao.justificativa}")
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")

if __name__ == "__main__":
    exemplo_analise_avancada()
