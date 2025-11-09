#!/usr/bin/env python3
"""
Teste rápido do sistema - Execução sem interface web
"""

def teste_sistema():
    """Testa o sistema sem interface web"""

    print("🧪 TESTE RÁPIDO DO SISTEMA")
    print("=" * 40)

    try:
        # Importar componentes principais
        print("📦 Importando módulos...")
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from datetime import datetime
        print("✅ Módulos importados com sucesso")

        # Testar API do Yahoo Finance
        print("\n🔌 Testando conexão com API...")
        ticker = "PETR4.SA"
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='5d')

        if not hist.empty:
            preco_atual = hist['Close'].iloc[-1]
            print(f"✅ Dados obtidos - PETR4: R$ {preco_atual:.2f}")
        else:
            print("❌ Erro ao obter dados históricos")
            return False

        # Testar indicadores básicos
        print("\n📊 Testando indicadores...")
        pe_ratio = info.get('trailingPE', 'N/A')
        pb_ratio = info.get('priceToBook', 'N/A')
        dividend_yield = info.get('dividendYield', 0)

        print(f"   P/L: {pe_ratio}")
        print(f"   P/VP: {pb_ratio}")  
        print(f"   Dividend Yield: {dividend_yield}")

        # Testar cálculos de risco
        print("\n⚡ Testando cálculos de risco...")
        retornos = hist['Close'].pct_change().dropna()
        if len(retornos) > 1:
            volatilidade = retornos.std() * np.sqrt(252)
            print(f"✅ Volatilidade calculada: {volatilidade:.3f}")
        else:
            print("❌ Dados insuficientes para cálculo de volatilidade")
            return False

        print("\n" + "=" * 40)
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Sistema funcionando corretamente")
        print("\n🚀 Execute o comando abaixo para usar:")
        print("   streamlit run analisador_acoes_completo.py")

        return True

    except ImportError as e:
        print(f"❌ Módulo não encontrado: {e}")
        print("\n💡 Execute: python instalar.py")
        return False

    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        print("\n💡 Verifique sua conexão com a internet")
        return False

if __name__ == "__main__":
    teste_sistema()
