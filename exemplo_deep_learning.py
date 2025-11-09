#!/usr/bin/env python3
"""
Exemplo de Uso do Sistema de Deep Learning
==========================================

Este script demonstra como usar o sistema de deep learning
para análise de ações com backtest histórico.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

def testar_imports():
    """Testa se todas as dependências estão disponíveis"""
    print("🔍 Testando dependências...")
    
    # Imports básicos
    try:
        import numpy as np
        import pandas as pd
        import yfinance as yf
        print("✅ Imports básicos: OK")
    except ImportError as e:
        print(f"❌ Erro nos imports básicos: {e}")
        return False
    
    # TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow não disponível")
        return False
    
    # Sistema de deep learning
    try:
        from deep_learning_system import SistemaDeepLearning, BacktestHistorico
        from transfer_learning_system import SistemaAvancadoDeepLearning, ValidacaoCruzadaTemporal
        print("✅ Sistema de deep learning: OK")
        return True
    except ImportError as e:
        print(f"❌ Erro no sistema de deep learning: {e}")
        return False

def exemplo_sistema_basico():
    """Exemplo do sistema básico de deep learning"""
    print("\n🚀 EXEMPLO 1: SISTEMA BÁSICO DE DEEP LEARNING")
    print("=" * 50)
    
    try:
        from deep_learning_system import SistemaDeepLearning
        
        # Configurar sistema
        ticker = "PETR4"
        print(f"📊 Configurando sistema para {ticker}...")
        
        sistema = SistemaDeepLearning(ticker)
        
        # Configurar sistema
        print("🔧 Preparando dados de treinamento...")
        X_train, X_test, y_train, y_test = sistema.configurar_sistema(
            janela_temporal=60,
            split_temporal=0.8
        )
        
        print(f"📈 Dados de treinamento: {X_train.shape}")
        print(f"📊 Dados de teste: {X_test.shape}")
        
        # Treinar sistema
        print("🧠 Treinando modelo de deep learning...")
        resultados = sistema.treinar_sistema(
            X_train, y_train, X_test, y_test,
            epochs=50, batch_size=32
        )
        
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
        print("\n🔮 Fazendo predições...")
        predicoes = sistema.fazer_predicoes(X_test[:5])
        print(f"Predições (primeiras 5): {predicoes.flatten()[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def exemplo_transfer_learning():
    """Exemplo de transfer learning"""
    print("\n🔄 EXEMPLO 2: TRANSFER LEARNING")
    print("=" * 40)
    
    try:
        from transfer_learning_system import SistemaAvancadoDeepLearning
        
        # Configurar sistema avançado
        print("🏗️ Configurando sistema de transfer learning...")
        sistema_avancado = SistemaAvancadoDeepLearning("PETR4")
        
        # Configurar sistema completo
        print("🚀 Treinando modelo base e transferindo conhecimento...")
        resultados = sistema_avancado.configurar_sistema_completo(
            acoes_secundarias=["VALE3", "ITUB4"]
        )
        
        print("\n📊 RESULTADOS DO TRANSFER LEARNING")
        print("=" * 40)
        
        # Modelo base
        base_r2 = resultados['modelo_base']['avaliacao']['r2_score']
        print(f"Modelo Base (PETR4): R² = {base_r2:.4f}")
        
        # Transferências
        print("\n🔄 Transferências:")
        for acao, resultado in resultados['transferencias'].items():
            if 'erro' not in resultado:
                r2 = resultado['avaliacao']['r2_score']
                print(f"  {acao}: R² = {r2:.4f}")
            else:
                print(f"  {acao}: ERRO - {resultado['erro']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def exemplo_validacao_cruzada():
    """Exemplo de validação cruzada temporal"""
    print("\n📊 EXEMPLO 3: VALIDAÇÃO CRUZADA TEMPORAL")
    print("=" * 45)
    
    try:
        from transfer_learning_system import ValidacaoCruzadaTemporal
        
        # Configurar validação cruzada
        print("🔄 Configurando validação cruzada temporal...")
        validacao = ValidacaoCruzadaTemporal("PETR4", n_folds=3)
        
        # Features para validação
        features = [
            'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR',
            'Volume_SMA', 'Price_Change', 'Volatility'
        ]
        
        # Executar validação
        print("📈 Executando validação cruzada...")
        resultados = validacao.executar_validacao_cruzada(features, "Return_5d")
        
        # Gerar relatório
        relatorio = validacao.gerar_relatorio_validacao()
        
        if 'erro' not in relatorio:
            print(f"\n📊 RELATÓRIO DE VALIDAÇÃO CRUZADA")
            print(f"Folds válidos: {relatorio['n_folds']}")
            print(f"R² médio: {relatorio['metricas_medias']['r2_score_medio']:.4f}")
            print(f"R² std: {relatorio['metricas_medias']['r2_score_std']:.4f}")
            print(f"RMSE médio: {relatorio['metricas_medias']['rmse_medio']:.4f}")
            print(f"Estabilidade: {relatorio['estabilidade']}")
            print(f"Qualidade: {relatorio['qualidade_geral']}")
        else:
            print(f"❌ Erro na validação: {relatorio['erro']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def exemplo_backtest_historico():
    """Exemplo de backtest histórico"""
    print("\n📈 EXEMPLO 4: BACKTEST HISTÓRICO")
    print("=" * 35)
    
    try:
        from deep_learning_system import BacktestHistorico
        
        # Configurar backtest
        print("📊 Configurando backtest histórico...")
        backtest = BacktestHistorico("PETR4", "2020-01-01", "2024-01-01")
        
        # Obter dados históricos
        print("🔍 Obtendo dados históricos...")
        dados = backtest.obter_dados_historicos()
        
        print(f"📈 Dados obtidos: {len(dados)} registros")
        print(f"📅 Período: {dados.index[0].strftime('%Y-%m-%d')} a {dados.index[-1].strftime('%Y-%m-%d')}")
        
        # Mostrar features
        features_tecnicas = [col for col in dados.columns if col in [
            'SMA_20', 'SMA_50', 'RSI', 'MACD', 'ATR', 'Volatility'
        ]]
        
        print(f"\n📊 Features técnicas disponíveis: {len(features_tecnicas)}")
        for feature in features_tecnicas[:5]:
            print(f"  - {feature}")
        
        # Preparar dados de treinamento
        print("\n🔧 Preparando dados de treinamento...")
        X_train, X_test, y_train, y_test = backtest.preparar_dados_treinamento(
            features_tecnicas, "Return_5d", janela_temporal=30
        )
        
        print(f"📈 Dados de treinamento: {X_train.shape}")
        print(f"📊 Dados de teste: {X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def exemplo_completo():
    """Exemplo completo do sistema"""
    print("\n🎯 EXEMPLO COMPLETO: SISTEMA INTEGRADO")
    print("=" * 45)
    
    try:
        from transfer_learning_system import SistemaAvancadoDeepLearning
        
        # Configurar sistema completo
        print("🚀 Configurando sistema completo...")
        sistema = SistemaAvancadoDeepLearning("PETR4")
        
        # Executar análise completa
        print("🧠 Executando análise completa...")
        
        # 1. Transfer learning
        print("1️⃣ Transfer learning...")
        resultados_transfer = sistema.configurar_sistema_completo(["VALE3"])
        
        # 2. Validação cruzada
        print("2️⃣ Validação cruzada...")
        validacao = sistema.executar_validacao_completa("PETR4", n_folds=3)
        
        print("\n📊 RESULTADO FINAL")
        print("=" * 25)
        
        # Transfer learning
        if 'modelo_base' in resultados_transfer:
            base_r2 = resultados_transfer['modelo_base']['avaliacao']['r2_score']
            print(f"Modelo Base: R² = {base_r2:.4f}")
        
        # Validação cruzada
        if 'relatorio' in validacao and 'erro' not in validacao['relatorio']:
            relatorio = validacao['relatorio']
            print(f"Validação Cruzada: R² = {relatorio['metricas_medias']['r2_score_medio']:.4f}")
            print(f"Estabilidade: {relatorio['estabilidade']}")
        
        print("\n✅ Sistema de deep learning funcionando perfeitamente!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return False

def main():
    """Função principal"""
    print("🧠 EXEMPLOS DO SISTEMA DE DEEP LEARNING")
    print("=" * 50)
    
    # Testar dependências
    if not testar_imports():
        print("\n❌ Dependências não disponíveis")
        print("💡 Instale: pip install tensorflow keras torch")
        return
    
    # Executar exemplos
    exemplos = [
        ("Sistema Básico", exemplo_sistema_basico),
        ("Transfer Learning", exemplo_transfer_learning),
        ("Validação Cruzada", exemplo_validacao_cruzada),
        ("Backtest Histórico", exemplo_backtest_historico),
        ("Sistema Completo", exemplo_completo)
    ]
    
    resultados = {}
    
    for nome, funcao in exemplos:
        print(f"\n{'='*60}")
        print(f"🎯 EXECUTANDO: {nome}")
        print(f"{'='*60}")
        
        try:
            sucesso = funcao()
            resultados[nome] = "✅ SUCESSO" if sucesso else "❌ FALHOU"
        except Exception as e:
            print(f"❌ Erro inesperado: {str(e)}")
            resultados[nome] = "❌ ERRO"
    
    # Resumo final
    print(f"\n{'='*60}")
    print("📊 RESUMO DOS EXEMPLOS")
    print(f"{'='*60}")
    
    for nome, resultado in resultados.items():
        print(f"{nome}: {resultado}")
    
    sucessos = sum(1 for r in resultados.values() if "✅" in r)
    total = len(resultados)
    
    print(f"\n🎯 Taxa de Sucesso: {sucessos}/{total} ({sucessos/total*100:.1f}%)")
    
    if sucessos == total:
        print("\n🎉 Todos os exemplos executaram com sucesso!")
        print("💡 O sistema de deep learning está funcionando perfeitamente!")
    else:
        print(f"\n⚠️ {total-sucessos} exemplo(s) falharam")
        print("💡 Verifique as dependências e configurações")

if __name__ == "__main__":
    main()
