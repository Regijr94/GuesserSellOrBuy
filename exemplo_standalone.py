#!/usr/bin/env python3
"""
Exemplo Standalone - Sistema de Análise de Ações
Use este arquivo para testar as classes POO sem interface web
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass

# ===== ESTRUTURAS DE DADOS =====

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
    classificacao_risco: str

# ===== CLASSES PRINCIPAIS =====

class YFinanceProvider:
    """Provedor de dados usando Yahoo Finance - Versão Simplificada"""

    def _formatar_ticker_brasileiro(self, ticker: str) -> str:
        if not ticker.endswith('.SA'):
            return f"{ticker}.SA"
        return ticker

    def obter_dados_basicos(self, ticker: str) -> Optional[DadosFinanceiros]:
        try:
            ticker_formatado = self._formatar_ticker_brasileiro(ticker)
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
            print(f"Erro ao obter dados básicos: {e}")
            return None

    def obter_historico_precos(self, ticker: str, periodo: str = '1y') -> pd.DataFrame:
        try:
            ticker_formatado = self._formatar_ticker_brasileiro(ticker)
            stock = yf.Ticker(ticker_formatado)
            return stock.history(period=periodo)
        except Exception as e:
            print(f"Erro ao obter histórico: {e}")
            return pd.DataFrame()

    def obter_indicadores_fundamentais(self, ticker: str) -> Dict:
        try:
            ticker_formatado = self._formatar_ticker_brasileiro(ticker)
            stock = yf.Ticker(ticker_formatado)
            info = stock.info

            return {
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'roe': info.get('returnOnEquity'),
                'beta': info.get('beta'),
                'eps': info.get('trailingEps'),
                'profit_margin': info.get('profitMargins')
            }
        except Exception as e:
            print(f"Erro ao obter indicadores: {e}")
            return {}

class CalculadoraIndicadores:
    """Calculadora de indicadores técnicos e de risco"""

    @staticmethod
    def calcular_volatilidade(precos: pd.Series, janela: int = 252) -> float:
        retornos = precos.pct_change().dropna()
        if len(retornos) < 2:
            return 0.0
        return float(retornos.std() * np.sqrt(janela))

    @staticmethod
    def calcular_sharpe_ratio(precos: pd.Series, taxa_livre_risco: float = 0.1375) -> float:
        retornos = precos.pct_change().dropna()
        if len(retornos) < 2:
            return 0.0

        retorno_medio = retornos.mean() * 252
        volatilidade = retornos.std() * np.sqrt(252)

        if volatilidade == 0:
            return 0.0

        return float((retorno_medio - taxa_livre_risco) / volatilidade)

    @staticmethod
    def calcular_var_95(precos: pd.Series) -> float:
        retornos = precos.pct_change().dropna()
        if len(retornos) < 2:
            return 0.0
        return float(np.percentile(retornos, 5) * 100)

    @staticmethod
    def calcular_max_drawdown(precos: pd.Series) -> float:
        if len(precos) < 2:
            return 0.0
        peak = precos.expanding(min_periods=1).max()
        drawdown = (precos / peak - 1.0)
        return float(drawdown.min() * 100)

class EstrategiaFundamentalista:
    """Estratégia de análise fundamentalista"""

    def __init__(self):
        self.pesos = {
            'pe_ratio': 0.25,
            'pb_ratio': 0.20,
            'dividend_yield': 0.15,
            'roe': 0.20,
            'profit_margin': 0.20
        }

    def analisar(self, dados: Dict) -> Dict:
        score = 0
        detalhes = {}

        # P/L
        pe = dados.get('pe_ratio')
        if pe and pe > 0:
            if pe < 10:
                score_pe = 10
                analise = 'Excelente'
            elif pe < 15:
                score_pe = 8  
                analise = 'Bom'
            elif pe < 25:
                score_pe = 5
                analise = 'Razoável'
            else:
                score_pe = 2
                analise = 'Alto'
            score += score_pe * self.pesos['pe_ratio']
            detalhes['pe_analise'] = analise

        # P/VP
        pb = dados.get('pb_ratio')
        if pb and pb > 0:
            if pb < 1:
                score_pb = 10
                analise = 'Subvalorizada'
            elif pb < 2:
                score_pb = 8
                analise = 'Boa'
            elif pb < 3:
                score_pb = 5
                analise = 'Razoável'
            else:
                score_pb = 2
                analise = 'Cara'
            score += score_pb * self.pesos['pb_ratio']
            detalhes['pb_analise'] = analise

        # ROE
        roe = dados.get('roe')
        if roe and roe > 0:
            roe_percent = roe * 100
            if roe_percent > 15:
                score_roe = 10
                analise = 'Excelente'
            elif roe_percent > 10:
                score_roe = 8
                analise = 'Bom'
            elif roe_percent > 5:
                score_roe = 5
                analise = 'Razoável'
            else:
                score_roe = 2
                analise = 'Baixo'
            score += score_roe * self.pesos['roe']
            detalhes['roe_analise'] = analise

        # Recomendação
        if score >= 8:
            recomendacao = 'COMPRAR'
        elif score >= 6:
            recomendacao = 'MANTER'
        elif score >= 4:
            recomendacao = 'NEUTRO'
        else:
            recomendacao = 'VENDER'

        return {
            'recomendacao': recomendacao,
            'score': score,
            'detalhes': detalhes
        }

class AvaliadorRisco:
    """Avaliador de risco de investimentos"""

    def __init__(self):
        self.calculadora = CalculadoraIndicadores()

    def avaliar_risco(self, ticker: str, historico: pd.DataFrame, beta: Optional[float] = None) -> AvaliacaoRisco:
        if historico.empty:
            return AvaliacaoRisco(
                ticker=ticker, volatilidade=0, sharpe_ratio=0, beta=1.0,
                var_95=0, max_drawdown=0, classificacao_risco="INDETERMINADO"
            )

        precos = historico['Close']

        volatilidade = self.calculadora.calcular_volatilidade(precos)
        sharpe_ratio = self.calculadora.calcular_sharpe_ratio(precos)
        var_95 = self.calculadora.calcular_var_95(precos)
        max_drawdown = self.calculadora.calcular_max_drawdown(precos)
        beta_final = beta if beta else 1.0

        # Classificar risco
        if volatilidade < 0.15 and abs(beta_final - 1) < 0.3:
            classificacao = "BAIXO"
        elif volatilidade < 0.30 and abs(beta_final - 1) < 0.6:
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
    """Sistema principal de análise"""

    def __init__(self):
        self.provider = YFinanceProvider()
        self.estrategia = EstrategiaFundamentalista()
        self.avaliador_risco = AvaliadorRisco()

    def analisar_acao(self, ticker: str) -> Dict:
        resultado = {'ticker': ticker, 'sucesso': False}

        try:
            # Dados básicos
            dados_basicos = self.provider.obter_dados_basicos(ticker)
            if not dados_basicos:
                resultado['erro'] = 'Dados básicos não disponíveis'
                return resultado

            # Indicadores fundamentalistas
            indicadores = self.provider.obter_indicadores_fundamentais(ticker)

            # Histórico
            historico = self.provider.obter_historico_precos(ticker, '1y')

            # Análises
            analise_fund = self.estrategia.analisar(indicadores)
            avaliacao_risco = self.avaliador_risco.avaliar_risco(
                ticker, historico, indicadores.get('beta')
            )

            # Recomendação final
            recom_final = self._combinar_recomendacoes(analise_fund, avaliacao_risco)

            resultado.update({
                'sucesso': True,
                'dados_basicos': dados_basicos,
                'indicadores': indicadores,
                'analise_fundamentalista': analise_fund,
                'avaliacao_risco': avaliacao_risco,
                'recomendacao_final': recom_final
            })

        except Exception as e:
            resultado['erro'] = str(e)

        return resultado

    def _combinar_recomendacoes(self, analise_fund: Dict, risco: AvaliacaoRisco) -> Dict:
        recom_fund = analise_fund['recomendacao']
        nivel_risco = risco.classificacao_risco

        if recom_fund == 'COMPRAR':
            if nivel_risco == 'BAIXO':
                final = 'COMPRA FORTE'
            elif nivel_risco == 'MÉDIO':
                final = 'COMPRAR'
            else:
                final = 'CAUTELA'
        elif recom_fund == 'MANTER':
            if nivel_risco == 'ALTO':
                final = 'REDUZIR POSIÇÃO'
            else:
                final = 'MANTER'
        else:
            final = 'VENDER' if nivel_risco != 'ALTO' else 'VENDA FORTE'

        return {
            'recomendacao': final,
            'score_fund': analise_fund['score'],
            'risco': nivel_risco
        }

# ===== EXEMPLO DE USO =====

def exemplo_analise_simples():
    """Exemplo básico de como usar o sistema"""

    print("🚀 EXEMPLO DE ANÁLISE - SISTEMA POO")
    print("=" * 50)

    # Criar o analisador
    analisador = AnalisadorAcoes()

    # Lista de ações para testar
    acoes = ['PETR4', 'VALE3', 'ITUB4']

    for ticker in acoes:
        print(f"\n📊 Analisando {ticker}...")
        resultado = analisador.analisar_acao(ticker)

        if resultado['sucesso']:
            dados = resultado['dados_basicos']
            indicadores = resultado['indicadores']
            analise = resultado['analise_fundamentalista']
            risco = resultado['avaliacao_risco']
            final = resultado['recomendacao_final']

            print(f"💰 Preço: R$ {dados.preco_atual:.2f} ({dados.variacao_dia:+.2f}%)")
            print(f"📈 P/L: {indicadores.get('pe_ratio', 'N/A')}")
            print(f"📊 P/VP: {indicadores.get('pb_ratio', 'N/A')}")
            print(f"⚡ Beta: {indicadores.get('beta', 'N/A')}")
            print(f"🎯 Score Fundamentalista: {analise['score']:.2f}/10")
            print(f"⚠️  Risco: {risco.classificacao_risco}")
            print(f"📝 Recomendação: {final['recomendacao']}")

        else:
            print(f"❌ Erro: {resultado.get('erro')}")

        print("-" * 30)

    print("\n✅ Análise concluída!")
    print("💡 Para usar a interface web, execute:")
    print("   streamlit run analisador_acoes_completo.py")

def exemplo_acao_especifica():
    """Exemplo detalhado de uma ação específica"""

    print("\n🔍 ANÁLISE DETALHADA - PETR4")
    print("=" * 40)

    analisador = AnalisadorAcoes()
    resultado = analisador.analisar_acao('PETR4')

    if resultado['sucesso']:
        dados = resultado['dados_basicos']
        indicadores = resultado['indicadores']
        risco = resultado['avaliacao_risco']

        print(f"📅 Data/Hora: {dados.timestamp.strftime('%d/%m/%Y %H:%M')}")
        print(f"💰 Preço Atual: R$ {dados.preco_atual:.2f}")
        print(f"📊 Volume: {dados.volume:,}")

        print("\n📋 INDICADORES FUNDAMENTALISTAS:")
        for key, value in indicadores.items():
            if value is not None:
                if key == 'roe' and isinstance(value, float):
                    print(f"   {key.upper()}: {value*100:.2f}%")
                elif isinstance(value, float):
                    print(f"   {key.upper()}: {value:.2f}")
                else:
                    print(f"   {key.upper()}: {value}")

        print("\n⚠️  MÉTRICAS DE RISCO:")
        print(f"   Volatilidade: {risco.volatilidade:.3f}")
        print(f"   Sharpe Ratio: {risco.sharpe_ratio:.3f}")
        print(f"   VaR 95%: {risco.var_95:.2f}%")
        print(f"   Max Drawdown: {risco.max_drawdown:.2f}%")
        print(f"   Classificação: {risco.classificacao_risco}")

    else:
        print(f"❌ Erro na análise: {resultado.get('erro')}")

if __name__ == "__main__":
    exemplo_analise_simples()
    exemplo_acao_especifica()
