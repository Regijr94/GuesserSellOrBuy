#!/usr/bin/env python3
"""
EXEMPLO DE USO DO SISTEMA AVANÇADO
Demonstra como usar as técnicas quantitativas e ML implementadas
"""

import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def exemplo_completo():
    """Exemplo completo do sistema avançado"""
    
    print("🚀 EXEMPLO DO SISTEMA AVANÇADO")
    print("=" * 60)
    
    try:
        # Importar sistema avançado
        from analise_avancada import (
            AnalisadorRiscoAvancado, PreditorPrecosML, SistemaRecomendacaoAvancado
        )
        
        print("✅ Sistema avançado carregado com sucesso!")
        
        # Inicializar componentes
        analisador_risco = AnalisadorRiscoAvancado()
        preditor_ml = PreditorPrecosML()
        sistema_recomendacao = SistemaRecomendacaoAvancado()
        
        print("✅ Componentes inicializados!")
        
        # Testar com uma ação
        ticker = "PETR4"
        print(f"\n📊 Testando análise avançada para {ticker}...")
        
        # Simular dados (em produção, viriam do Yahoo Finance)
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Criar dados simulados para demonstração
        dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        # Simular preços com tendência e volatilidade
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% retorno médio, 2% volatilidade
        prices = 100 * np.exp(np.cumsum(returns))
        
        historico_simulado = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        print("✅ Dados históricos simulados criados!")
        
        # Análise de risco avançada
        print("\n🔍 Executando análise de risco avançada...")
        analise_risco = analisador_risco.analisar_risco_completo(ticker, historico_simulado)
        
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
        print("\n🤖 Executando predição com Machine Learning...")
        predicao = preditor_ml.prever_precos(ticker, historico_simulado)
        
        print(f"   Preço Atual: R$ {predicao.preco_atual:.2f}")
        print(f"   Previsão 1m: R$ {predicao.previsao_1m:.2f} (Conf: {predicao.confianca_1m:.0f}%)")
        print(f"   Previsão 3m: R$ {predicao.previsao_3m:.2f} (Conf: {predicao.confianca_3m:.0f}%)")
        print(f"   Previsão 6m: R$ {predicao.previsao_6m:.2f} (Conf: {predicao.confianca_6m:.0f}%)")
        print(f"   Modelo: {predicao.modelo_usado}")
        print(f"   R² Score: {predicao.r2_score:.3f}")
        
        # Recomendação avançada
        print("\n💡 Gerando recomendação avançada...")
        dados_fundamentais = {
            'pe_ratio': 12.5,
            'pb_ratio': 1.8,
            'roe': 0.15,
            'dividend_yield': 4.2
        }
        
        recomendacao = sistema_recomendacao.gerar_recomendacao(
            ticker, dados_fundamentais, analise_risco, predicao, historico_simulado
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
        
        print("\n🎉 Análise avançada concluída com sucesso!")
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("💡 Instale as dependências: pip install scikit-learn scipy")
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")

def exemplo_metricas_risco():
    """Exemplo focado em métricas de risco"""
    
    print("\n📊 EXEMPLO DE MÉTRICAS DE RISCO")
    print("=" * 40)
    
    try:
        from analise_avancada import AnalisadorRiscoAvancado
        import pandas as pd
        import numpy as np
        
        # Criar dados com diferentes características de risco
        np.random.seed(123)
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        # Cenário 1: Baixo risco (baixa volatilidade, retorno positivo)
        returns_baixo = np.random.normal(0.001, 0.01, len(dates))
        prices_baixo = 100 * np.exp(np.cumsum(returns_baixo))
        
        historico_baixo = pd.DataFrame({
            'Close': prices_baixo,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Cenário 2: Alto risco (alta volatilidade, retorno negativo)
        returns_alto = np.random.normal(-0.0005, 0.04, len(dates))
        prices_alto = 100 * np.exp(np.cumsum(returns_alto))
        
        historico_alto = pd.DataFrame({
            'Close': prices_alto,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        analisador = AnalisadorRiscoAvancado()
        
        print("🔍 Analisando cenário de BAIXO RISCO:")
        risco_baixo = analisador.analisar_risco_completo("BAIXO_RISCO", historico_baixo)
        print(f"   VaR 95%: {risco_baixo.var_95:.2f}%")
        print(f"   Max Drawdown: {risco_baixo.max_drawdown:.2f}%")
        print(f"   Calmar Ratio: {risco_baixo.calmar_ratio:.2f}")
        print(f"   Risk Score: {risco_baixo.risk_score:.1f}/100")
        print(f"   Rating: {risco_baixo.risk_rating}")
        
        print("\n🔍 Analisando cenário de ALTO RISCO:")
        risco_alto = analisador.analisar_risco_completo("ALTO_RISCO", historico_alto)
        print(f"   VaR 95%: {risco_alto.var_95:.2f}%")
        print(f"   Max Drawdown: {risco_alto.max_drawdown:.2f}%")
        print(f"   Calmar Ratio: {risco_alto.calmar_ratio:.2f}")
        print(f"   Risk Score: {risco_alto.risk_score:.1f}/100")
        print(f"   Rating: {risco_alto.risk_rating}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")

def exemplo_ml():
    """Exemplo focado em Machine Learning"""
    
    print("\n🤖 EXEMPLO DE MACHINE LEARNING")
    print("=" * 40)
    
    try:
        from analise_avancada import PreditorPrecosML
        import pandas as pd
        import numpy as np
        
        # Criar dados com padrões para ML
        np.random.seed(456)
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        
        # Simular preços com tendência e sazonalidade
        trend = np.linspace(100, 150, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)  # Sazonalidade anual
        noise = np.random.normal(0, 2, len(dates))
        prices = trend + seasonal + noise
        
        historico_ml = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        preditor = PreditorPrecosML()
        
        print("🔍 Treinando modelos de ML...")
        predicao = preditor.prever_precos("ML_TEST", historico_ml)
        
        print(f"   Preço Atual: R$ {predicao.preco_atual:.2f}")
        print(f"   Previsão 1m: R$ {predicao.previsao_1m:.2f} (Conf: {predicao.confianca_1m:.0f}%)")
        print(f"   Previsão 3m: R$ {predicao.previsao_3m:.2f} (Conf: {predicao.confianca_3m:.0f}%)")
        print(f"   Modelo Usado: {predicao.modelo_usado}")
        print(f"   R² Score: {predicao.r2_score:.3f}")
        print(f"   RMSE: R$ {predicao.rmse:.2f}")
        
        # Interpretação
        if predicao.r2_score > 0.7:
            print("   ✅ Modelo excelente (R² > 0.7)")
        elif predicao.r2_score > 0.5:
            print("   ⚠️ Modelo moderado (R² > 0.5)")
        else:
            print("   ❌ Modelo baixo (R² < 0.5)")
            
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    exemplo_completo()
    exemplo_metricas_risco()
    exemplo_ml()
    
    print("\n" + "="*60)
    print("🎯 RESUMO DOS EXEMPLOS")
    print("="*60)
    print("✅ Análise de risco com VaR, CVaR, Calmar Ratio")
    print("✅ Machine Learning com Random Forest, Gradient Boosting")
    print("✅ Recomendações multi-fatoriais")
    print("✅ Métricas quantitativas avançadas")
    print("✅ Sistema de confiança e probabilidade de sucesso")
    print("\n💡 Para usar no sistema principal:")
    print("   1. Instale: pip install scikit-learn scipy")
    print("   2. Execute: streamlit run analisador_acoes_completo.py")
    print("   3. Ative: 'Análise Avançada (ML + Quant)' na sidebar")
