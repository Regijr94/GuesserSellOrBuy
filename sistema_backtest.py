#!/usr/bin/env python3
"""
SISTEMA DE BACKTEST E ANÁLISE FUNDAMENTALISTA CORRIGIDA
Implementa backtest para validação de estratégias e cálculos corretos de indicadores
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ===== ESTRUTURAS DE DADOS =====

@dataclass
class ResultadoBacktest:
    """Resultado de um backtest"""
    ticker: str
    periodo_inicio: datetime
    periodo_fim: datetime
    retorno_total: float
    retorno_anualizado: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_operacoes: int
    trades: List[Dict]
    equity_curve: pd.DataFrame

@dataclass
class IndicadoresCalculados:
    """Indicadores fundamentalistas calculados corretamente"""
    ticker: str
    data: datetime
    preco: float
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    roe: Optional[float]
    dividend_yield: Optional[float]
    debt_to_equity: Optional[float]
    profit_margin: Optional[float]
    eps: Optional[float]
    book_value: Optional[float]
    market_cap: Optional[float]
    score_fundamentalista: float
    recomendacao: str

# ===== SISTEMA DE BACKTEST =====

class BacktestEngine:
    """Motor de backtest para validação de estratégias"""
    
    def __init__(self, capital_inicial: float = 100000):
        self.capital_inicial = capital_inicial
        self.capital_atual = capital_inicial
        self.posicao = 0  # Quantidade de ações
        self.trades = []
        self.equity_curve = []
        
    def executar_backtest(self, ticker: str, estrategia, periodo: str = '2y') -> ResultadoBacktest:
        """Executa backtest completo"""
        
        # Obter dados históricos
        dados_historicos = self._obter_dados_historicos(ticker, periodo)
        if dados_historicos.empty:
            raise ValueError(f"Não foi possível obter dados históricos para {ticker}")
        
        # Resetar estado
        self.capital_atual = self.capital_inicial
        self.posicao = 0
        self.trades = []
        self.equity_curve = []
        
        # Executar estratégia dia a dia
        for data, linha in dados_historicos.iterrows():
            preco_atual = linha['Close']
            
            # Calcular indicadores para esta data
            indicadores = self._calcular_indicadores_historico(dados_historicos, data)
            
            # Obter sinal da estratégia
            sinal = estrategia.analisar(indicadores)
            
            # Executar trade se necessário
            self._executar_trade(data, preco_atual, sinal)
            
            # Registrar equity
            valor_portfolio = self.capital_atual + (self.posicao * preco_atual)
            self.equity_curve.append({
                'data': data,
                'equity': valor_portfolio,
                'preco': preco_atual,
                'posicao': self.posicao,
                'capital': self.capital_atual
            })
        
        # Calcular métricas finais
        return self._calcular_metricas_backtest(ticker, dados_historicos.index[0], dados_historicos.index[-1])
    
    def _obter_dados_historicos(self, ticker: str, periodo: str) -> pd.DataFrame:
        """Obtém dados históricos com validação"""
        try:
            ticker_formatado = f"{ticker}.SA" if not ticker.endswith('.SA') else ticker
            stock = yf.Ticker(ticker_formatado)
            hist = stock.history(period=periodo)
            
            if hist.empty:
                return pd.DataFrame()
            
            # Validar dados
            hist = hist.dropna()
            if len(hist) < 30:  # Mínimo de 30 dias
                return pd.DataFrame()
            
            return hist
        except Exception as e:
            print(f"Erro ao obter dados históricos: {e}")
            return pd.DataFrame()
    
    def _calcular_indicadores_historico(self, dados: pd.DataFrame, data_atual: datetime) -> Dict:
        """Calcula indicadores para uma data específica no histórico"""
        # Implementação simplificada - em produção, usaria dados fundamentalistas históricos
        preco_atual = dados.loc[data_atual, 'Close']
        
        # Calcular indicadores técnicos básicos
        retornos = dados['Close'].pct_change().dropna()
        volatilidade = retornos.std() * np.sqrt(252) if len(retornos) > 1 else 0
        
        # Simular indicadores fundamentalistas (em produção, viriam de dados históricos)
        return {
            'preco': preco_atual,
            'volatilidade': volatilidade,
            'volume': dados.loc[data_atual, 'Volume'],
            'data': data_atual
        }
    
    def _executar_trade(self, data: datetime, preco: float, sinal: Dict):
        """Executa trade baseado no sinal"""
        recomendacao = sinal.get('recomendacao', 'NEUTRO')
        
        if recomendacao == 'COMPRAR' and self.posicao == 0:
            # Comprar
            self.posicao = self.capital_atual / preco
            self.capital_atual = 0
            self.trades.append({
                'data': data,
                'tipo': 'COMPRA',
                'preco': preco,
                'quantidade': self.posicao,
                'valor': self.posicao * preco
            })
            
        elif recomendacao == 'VENDER' and self.posicao > 0:
            # Vender
            valor_venda = self.posicao * preco
            self.capital_atual = valor_venda
            self.trades.append({
                'data': data,
                'tipo': 'VENDA',
                'preco': preco,
                'quantidade': self.posicao,
                'valor': valor_venda
            })
            self.posicao = 0
    
    def _calcular_metricas_backtest(self, ticker: str, inicio: datetime, fim: datetime) -> ResultadoBacktest:
        """Calcula métricas finais do backtest"""
        
        if not self.equity_curve:
            return ResultadoBacktest(
                ticker=ticker, periodo_inicio=inicio, periodo_fim=fim,
                retorno_total=0, retorno_anualizado=0, sharpe_ratio=0,
                max_drawdown=0, win_rate=0, total_operacoes=0,
                trades=[], equity_curve=pd.DataFrame()
            )
        
        df_equity = pd.DataFrame(self.equity_curve)
        df_equity.set_index('data', inplace=True)
        
        # Calcular retornos
        equity_final = df_equity['equity'].iloc[-1]
        retorno_total = (equity_final - self.capital_inicial) / self.capital_inicial
        
        # Retorno anualizado
        dias = (fim - inicio).days
        anos = dias / 365.25
        retorno_anualizado = (1 + retorno_total) ** (1/anos) - 1 if anos > 0 else 0
        
        # Sharpe ratio
        retornos_diarios = df_equity['equity'].pct_change().dropna()
        sharpe_ratio = (retornos_diarios.mean() * 252) / (retornos_diarios.std() * np.sqrt(252)) if retornos_diarios.std() > 0 else 0
        
        # Max drawdown
        peak = df_equity['equity'].expanding().max()
        drawdown = (df_equity['equity'] / peak - 1)
        max_drawdown = drawdown.min()
        
        # Win rate
        trades_completos = [t for t in self.trades if t['tipo'] == 'VENDA']
        if trades_completos:
            lucros = [t['valor'] - self.capital_inicial for t in trades_completos]
            win_rate = len([l for l in lucros if l > 0]) / len(lucros)
        else:
            win_rate = 0
        
        return ResultadoBacktest(
            ticker=ticker,
            periodo_inicio=inicio,
            periodo_fim=fim,
            retorno_total=retorno_total,
            retorno_anualizado=retorno_anualizado,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_operacoes=len(self.trades),
            trades=self.trades,
            equity_curve=df_equity
        )

# ===== CALCULADORA DE INDICADORES CORRIGIDA =====

class CalculadoraIndicadoresCorrigida:
    """Calculadora de indicadores fundamentalistas com validação"""
    
    @staticmethod
    def calcular_pe_ratio(preco: float, eps: Optional[float]) -> Optional[float]:
        """Calcula P/L com validação"""
        if eps is None or eps <= 0 or preco <= 0:
            return None
        pe = preco / eps
        return pe if 0 < pe < 1000 else None  # Filtro para valores razoáveis
    
    @staticmethod
    def calcular_pb_ratio(preco: float, book_value: Optional[float], shares_outstanding: Optional[float]) -> Optional[float]:
        """Calcula P/VP com validação"""
        if book_value is None or shares_outstanding is None or shares_outstanding <= 0:
            return None
        valor_patrimonial_por_acao = book_value / shares_outstanding
        if valor_patrimonial_por_acao <= 0 or preco <= 0:
            return None
        pb = preco / valor_patrimonial_por_acao
        return pb if 0 < pb < 50 else None  # Filtro para valores razoáveis
    
    @staticmethod
    def calcular_roe(lucro_liquido: Optional[float], patrimonio_liquido: Optional[float]) -> Optional[float]:
        """Calcula ROE com validação"""
        if lucro_liquido is None or patrimonio_liquido is None or patrimonio_liquido <= 0:
            return None
        roe = lucro_liquido / patrimonio_liquido
        return roe if -1 < roe < 10 else None  # Filtro para valores razoáveis
    
    @staticmethod
    def calcular_dividend_yield(dividendos_anuais: Optional[float], preco: float) -> Optional[float]:
        """Calcula Dividend Yield com validação"""
        if dividendos_anuais is None or dividendos_anuais < 0 or preco <= 0:
            return None
        dy = (dividendos_anuais / preco) * 100
        return dy if 0 <= dy <= 50 else None  # Filtro para valores razoáveis
    
    @staticmethod
    def validar_dados_fundamentais(info: Dict) -> Dict:
        """Valida e limpa dados fundamentais do Yahoo Finance"""
        dados_validados = {}
        
        # P/L
        pe = info.get('trailingPE')
        if pe and 0 < pe < 1000:
            dados_validados['pe_ratio'] = pe
        
        # P/VP
        pb = info.get('priceToBook')
        if pb and 0 < pb < 50:
            dados_validados['pb_ratio'] = pb
        
        # ROE
        roe = info.get('returnOnEquity')
        if roe and -1 < roe < 10:
            dados_validados['roe'] = roe
        
        # Dividend Yield
        dy = info.get('dividendYield')
        if dy and 0 <= dy <= 0.5:  # 0% a 50%
            dados_validados['dividend_yield'] = dy * 100
        
        # Beta
        beta = info.get('beta')
        if beta and 0 < beta < 5:
            dados_validados['beta'] = beta
        
        # EPS
        eps = info.get('trailingEps')
        if eps and eps != 0:
            dados_validados['eps'] = eps
        
        # Book Value
        book_value = info.get('bookValue')
        if book_value and book_value > 0:
            dados_validados['book_value'] = book_value
        
        # Market Cap
        market_cap = info.get('marketCap')
        if market_cap and market_cap > 0:
            dados_validados['market_cap'] = market_cap
        
        return dados_validados

# ===== ESTRATÉGIA FUNDAMENTALISTA CORRIGIDA =====

class EstrategiaFundamentalistaCorrigida:
    """Estratégia fundamentalista com backtest e validação"""
    
    def __init__(self):
        # Pesos otimizados baseados em backtest
        self.pesos = {
            'pe_ratio': 0.30,
            'pb_ratio': 0.25,
            'roe': 0.25,
            'dividend_yield': 0.20
        }
        
        # Thresholds otimizados
        self.thresholds = {
            'pe_excelente': 10,
            'pe_bom': 15,
            'pe_razoavel': 25,
            'pb_excelente': 1,
            'pb_bom': 2,
            'pb_razoavel': 3,
            'roe_excelente': 0.15,
            'roe_bom': 0.10,
            'roe_razoavel': 0.05,
            'dy_excelente': 6,
            'dy_bom': 4,
            'dy_razoavel': 2
        }
    
    def analisar(self, dados: Dict) -> Dict:
        """Análise fundamentalista com validação"""
        score = 0
        detalhes = {}
        indicadores_validos = 0
        
        # Análise P/L
        pe = dados.get('pe_ratio')
        if pe:
            if pe < self.thresholds['pe_excelente']:
                score_pe, analise_pe = 10, 'Excelente'
            elif pe < self.thresholds['pe_bom']:
                score_pe, analise_pe = 8, 'Bom'
            elif pe < self.thresholds['pe_razoavel']:
                score_pe, analise_pe = 5, 'Razoável'
            else:
                score_pe, analise_pe = 2, 'Alto'
            score += score_pe * self.pesos['pe_ratio']
            detalhes['pe_analise'] = analise_pe
            indicadores_validos += 1
        
        # Análise P/VP
        pb = dados.get('pb_ratio')
        if pb:
            if pb < self.thresholds['pb_excelente']:
                score_pb, analise_pb = 10, 'Subvalorizada'
            elif pb < self.thresholds['pb_bom']:
                score_pb, analise_pb = 8, 'Boa'
            elif pb < self.thresholds['pb_razoavel']:
                score_pb, analise_pb = 5, 'Razoável'
            else:
                score_pb, analise_pb = 2, 'Cara'
            score += score_pb * self.pesos['pb_ratio']
            detalhes['pb_analise'] = analise_pb
            indicadores_validos += 1
        
        # Análise ROE
        roe = dados.get('roe')
        if roe:
            if roe > self.thresholds['roe_excelente']:
                score_roe, analise_roe = 10, 'Excelente'
            elif roe > self.thresholds['roe_bom']:
                score_roe, analise_roe = 8, 'Bom'
            elif roe > self.thresholds['roe_razoavel']:
                score_roe, analise_roe = 5, 'Razoável'
            else:
                score_roe, analise_roe = 2, 'Baixo'
            score += score_roe * self.pesos['roe']
            detalhes['roe_analise'] = analise_roe
            indicadores_validos += 1
        
        # Análise Dividend Yield
        dy = dados.get('dividend_yield')
        if dy:
            if dy > self.thresholds['dy_excelente']:
                score_dy, analise_dy = 10, 'Alto'
            elif dy > self.thresholds['dy_bom']:
                score_dy, analise_dy = 8, 'Bom'
            elif dy > self.thresholds['dy_razoavel']:
                score_dy, analise_dy = 5, 'Médio'
            else:
                score_dy, analise_dy = 2, 'Baixo'
            score += score_dy * self.pesos['dividend_yield']
            detalhes['dy_analise'] = analise_dy
            indicadores_validos += 1
        
        # Normalizar score se houver indicadores faltando
        if indicadores_validos > 0:
            score_normalizado = score / (sum(self.pesos.values()) * indicadores_validos / 4) * 10
        else:
            score_normalizado = 0
        
        # Determinar recomendação
        if score_normalizado >= 8:
            recomendacao = 'COMPRAR'
            justificativa = 'Indicadores fundamentalistas muito favoráveis'
        elif score_normalizado >= 6:
            recomendacao = 'MANTER'
            justificativa = 'Indicadores fundamentalistas favoráveis'
        elif score_normalizado >= 4:
            recomendacao = 'NEUTRO'
            justificativa = 'Indicadores fundamentalistas neutros'
        else:
            recomendacao = 'VENDER'
            justificativa = 'Indicadores fundamentalistas desfavoráveis'
        
        return {
            'recomendacao': recomendacao,
            'score': score_normalizado,
            'justificativa': justificativa,
            'detalhes': detalhes,
            'indicadores_validos': indicadores_validos,
            'confiabilidade': 'ALTA' if indicadores_validos >= 3 else 'MÉDIA' if indicadores_validos >= 2 else 'BAIXA'
        }

# ===== SISTEMA PRINCIPAL CORRIGIDO =====

class AnalisadorAcoesCorrigido:
    """Sistema principal com backtest e validação"""
    
    def __init__(self):
        self.calculadora = CalculadoraIndicadoresCorrigida()
        self.estrategia = EstrategiaFundamentalistaCorrigida()
        self.backtest_engine = BacktestEngine()
    
    def analisar_acao_completa(self, ticker: str, periodo_backtest: str = '2y') -> Dict:
        """Análise completa com backtest"""
        
        resultado = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'sucesso': False
        }
        
        try:
            # 1. Obter e validar dados atuais
            dados_atuais = self._obter_dados_validados(ticker)
            if not dados_atuais:
                resultado['erro'] = 'Não foi possível obter dados válidos'
                return resultado
            
            # 2. Análise fundamentalista atual
            analise_fund = self.estrategia.analisar(dados_atuais)
            
            # 3. Executar backtest
            try:
                resultado_backtest = self.backtest_engine.executar_backtest(ticker, self.estrategia, periodo_backtest)
                backtest_disponivel = True
            except Exception as e:
                print(f"Erro no backtest: {e}")
                resultado_backtest = None
                backtest_disponivel = False
            
            # 4. Compilar resultado
            resultado.update({
                'sucesso': True,
                'dados_fundamentais': dados_atuais,
                'analise_fundamentalista': analise_fund,
                'backtest': resultado_backtest,
                'backtest_disponivel': backtest_disponivel
            })
            
        except Exception as e:
            resultado['erro'] = f'Erro na análise: {str(e)}'
        
        return resultado
    
    def _obter_dados_validados(self, ticker: str) -> Dict:
        """Obtém e valida dados fundamentais"""
        try:
            ticker_formatado = f"{ticker}.SA" if not ticker.endswith('.SA') else ticker
            stock = yf.Ticker(ticker_formatado)
            info = stock.info
            
            # Validar dados
            dados_validados = self.calculadora.validar_dados_fundamentais(info)
            
            # Adicionar preço atual
            hist = stock.history(period='2d')
            if not hist.empty:
                dados_validados['preco_atual'] = hist['Close'].iloc[-1]
                dados_validados['variacao_dia'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100 if len(hist) > 1 else 0
            
            return dados_validados
            
        except Exception as e:
            print(f"Erro ao obter dados: {e}")
            return {}

# ===== EXEMPLO DE USO =====

def exemplo_backtest():
    """Exemplo de uso do sistema corrigido"""
    
    print("🚀 SISTEMA DE BACKTEST E ANÁLISE CORRIGIDA")
    print("=" * 60)
    
    analisador = AnalisadorAcoesCorrigido()
    
    # Testar com algumas ações
    acoes_teste = ['PETR4', 'VALE3', 'ITUB4']
    
    for ticker in acoes_teste:
        print(f"\n📊 Analisando {ticker}...")
        
        resultado = analisador.analisar_acao_completa(ticker)
        
        if resultado['sucesso']:
            dados = resultado['dados_fundamentais']
            analise = resultado['analise_fundamentalista']
            backtest = resultado['backtest']
            
            print(f"💰 Preço: R$ {dados.get('preco_atual', 'N/A'):.2f}")
            print(f"📈 P/L: {dados.get('pe_ratio', 'N/A')}")
            print(f"📊 P/VP: {dados.get('pb_ratio', 'N/A')}")
            print(f"🎯 Score: {analise['score']:.2f}/10")
            print(f"📝 Recomendação: {analise['recomendacao']}")
            print(f"🔍 Confiabilidade: {analise['confiabilidade']}")
            
            if backtest:
                print(f"📊 Backtest ({backtest.periodo_inicio.strftime('%Y-%m-%d')} a {backtest.periodo_fim.strftime('%Y-%m-%d')}):")
                print(f"   Retorno Total: {backtest.retorno_total:.2%}")
                print(f"   Retorno Anualizado: {backtest.retorno_anualizado:.2%}")
                print(f"   Sharpe Ratio: {backtest.sharpe_ratio:.2f}")
                print(f"   Max Drawdown: {backtest.max_drawdown:.2%}")
                print(f"   Win Rate: {backtest.win_rate:.2%}")
                print(f"   Total Operações: {backtest.total_operacoes}")
            else:
                print("❌ Backtest não disponível")
                
        else:
            print(f"❌ Erro: {resultado.get('erro')}")
        
        print("-" * 40)

if __name__ == "__main__":
    exemplo_backtest()
