#!/usr/bin/env python3
"""
Teste da Integração da Análise Avançada
=======================================

Este script testa se a análise avançada foi integrada corretamente
como feature nativa do sistema.
"""

import sys
import os

def testar_imports():
    """Testa se todos os imports necessários estão funcionando"""
    print("🔍 Testando imports...")
    
    try:
        # Imports básicos
        import streamlit as st
        import pandas as pd
        import numpy as np
        import yfinance as yf
        print("✅ Imports básicos: OK")
        
        # Imports avançados
        try:
            from analise_avancada import (
                AnalisadorRiscoAvancado, PreditorPrecosML, SistemaRecomendacaoAvancado
            )
            print("✅ Sistema avançado: DISPONÍVEL")
            return True
        except ImportError as e:
            print(f"⚠️ Sistema avançado: NÃO DISPONÍVEL ({e})")
            return False
            
    except ImportError as e:
        print(f"❌ Erro nos imports básicos: {e}")
        return False

def testar_deteccao_sistema():
    """Testa a detecção automática do sistema"""
    print("\n🔍 Testando detecção do sistema...")
    
    try:
        # Simular a lógica de detecção do sistema principal
        try:
            from analise_avancada import (
                AnalisadorRiscoAvancado, PreditorPrecosML, SistemaRecomendacaoAvancado
            )
            SISTEMA_AVANCADO_DISPONIVEL = True
            print("✅ SISTEMA_AVANCADO_DISPONIVEL = True")
        except ImportError:
            SISTEMA_AVANCADO_DISPONIVEL = False
            print("⚠️ SISTEMA_AVANCADO_DISPONIVEL = False")
        
        # Testar lógica de execução
        if SISTEMA_AVANCADO_DISPONIVEL:
            print("🚀 Sistema usará análise avançada automaticamente")
            return "AVANCADO"
        else:
            print("📊 Sistema usará análise básica")
            return "BASICO"
            
    except Exception as e:
        print(f"❌ Erro na detecção: {e}")
        return "ERRO"

def testar_componentes_avancados():
    """Testa se os componentes avançados estão funcionando"""
    print("\n🔍 Testando componentes avançados...")
    
    try:
        from analise_avancada import (
            AnalisadorRiscoAvancado, PreditorPrecosML, SistemaRecomendacaoAvancado
        )
        
        # Testar inicialização
        analisador_risco = AnalisadorRiscoAvancado()
        preditor_ml = PreditorPrecosML()
        sistema_recomendacao = SistemaRecomendacaoAvancado()
        
        print("✅ AnalisadorRiscoAvancado: OK")
        print("✅ PreditorPrecosML: OK")
        print("✅ SistemaRecomendacaoAvancado: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nos componentes avançados: {e}")
        return False

def testar_sistema_basico():
    """Testa se o sistema básico está funcionando"""
    print("\n🔍 Testando sistema básico...")
    
    try:
        # Importar classes básicas
        from analisador_acoes_completo import YFinanceProvider, AnalisadorAcoes
        
        # Testar inicialização
        fonte_dados = YFinanceProvider()
        analisador = AnalisadorAcoes(fonte_dados)
        
        print("✅ YFinanceProvider: OK")
        print("✅ AnalisadorAcoes: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no sistema básico: {e}")
        return False

def main():
    """Função principal do teste"""
    print("🚀 TESTE DA INTEGRAÇÃO DA ANÁLISE AVANÇADA")
    print("=" * 50)
    
    # Teste 1: Imports
    imports_ok = testar_imports()
    
    # Teste 2: Detecção do sistema
    tipo_sistema = testar_deteccao_sistema()
    
    # Teste 3: Componentes avançados (se disponível)
    if imports_ok:
        componentes_ok = testar_componentes_avancados()
    else:
        componentes_ok = False
    
    # Teste 4: Sistema básico
    sistema_basico_ok = testar_sistema_basico()
    
    # Resultado final
    print("\n" + "=" * 50)
    print("📊 RESULTADO DOS TESTES")
    print("=" * 50)
    
    print(f"🔍 Imports básicos: {'✅ OK' if imports_ok else '❌ ERRO'}")
    print(f"🚀 Sistema avançado: {'✅ DISPONÍVEL' if imports_ok else '⚠️ NÃO DISPONÍVEL'}")
    print(f"📊 Sistema básico: {'✅ OK' if sistema_basico_ok else '❌ ERRO'}")
    print(f"🎯 Tipo de sistema: {tipo_sistema}")
    
    if tipo_sistema == "AVANCADO":
        print("\n🎉 SUCESSO: Sistema avançado integrado e funcionando!")
        print("💡 O sistema usará ML + técnicas quantitativas automaticamente")
    elif tipo_sistema == "BASICO":
        print("\n⚠️ SISTEMA BÁSICO: Funcionando, mas sem análise avançada")
        print("💡 Instale scikit-learn para ativar análise avançada")
    else:
        print("\n❌ ERRO: Problemas na integração")
    
    print("\n🚀 Para executar o sistema:")
    print("   streamlit run analisador_acoes_completo.py")
    print("   Acesse: http://localhost:8501")

if __name__ == "__main__":
    main()
