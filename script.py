# Criando requirements.txt simplificado
requirements_simples = '''streamlit>=1.28.0
yfinance>=0.2.18
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
'''

with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(requirements_simples)

# Criando script de instalação automática
script_instalacao = '''#!/usr/bin/env python3
"""
Script de instalação automática para o Sistema de Análise de Ações
Execute este arquivo para instalar todas as dependências automaticamente
"""

import subprocess
import sys
import os

def instalar_dependencias():
    """Instala todas as dependências necessárias"""
    
    print("🚀 INSTALAÇÃO DO SISTEMA DE ANÁLISE DE AÇÕES")
    print("=" * 50)
    
    # Verificar se pip está disponível
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ pip encontrado")
    except subprocess.CalledProcessError:
        print("❌ pip não encontrado. Instale o Python pip primeiro.")
        return False
    
    # Lista de dependências
    dependencias = [
        'streamlit>=1.28.0',
        'yfinance>=0.2.18', 
        'pandas>=1.5.0',
        'numpy>=1.24.0',
        'plotly>=5.15.0'
    ]
    
    print("\\n📦 Instalando dependências...")
    
    for dep in dependencias:
        try:
            print(f"⏳ Instalando {dep.split('>=')[0]}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ], stdout=subprocess.DEVNULL)
            print(f"✅ {dep.split('>=')[0]} instalado com sucesso")
        except subprocess.CalledProcessError:
            print(f"❌ Erro ao instalar {dep}")
            return False
    
    print("\\n🎉 Todas as dependências foram instaladas com sucesso!")
    return True

def verificar_instalacao():
    """Verifica se todas as dependências estão funcionando"""
    
    print("\\n🔍 Verificando instalação...")
    
    modulos = [
        ('streamlit', 'st'),
        ('yfinance', 'yf'),
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('plotly.graph_objects', 'go')
    ]
    
    todos_ok = True
    
    for modulo, alias in modulos:
        try:
            __import__(modulo)
            print(f"✅ {modulo}")
        except ImportError:
            print(f"❌ {modulo} não encontrado")
            todos_ok = False
    
    return todos_ok

def main():
    """Função principal"""
    
    if instalar_dependencias():
        if verificar_instalacao():
            print("\\n" + "=" * 50)
            print("🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO!")
            print("=" * 50)
            print("\\n🚀 Para executar o sistema:")
            print("   streamlit run analisador_acoes_completo.py")
            print("\\n🌐 O sistema abrirá no seu navegador automaticamente")
            print("   Endereço: http://localhost:8501")
            
            # Perguntar se quer executar agora
            resposta = input("\\n❓ Deseja executar o sistema agora? (s/n): ").lower().strip()
            if resposta in ['s', 'sim', 'y', 'yes']:
                print("🚀 Iniciando sistema...")
                try:
                    subprocess.run([
                        sys.executable, "-m", "streamlit", "run", 
                        "analisador_acoes_completo.py"
                    ])
                except KeyboardInterrupt:
                    print("\\n👋 Sistema encerrado pelo usuário")
                except Exception as e:
                    print(f"❌ Erro ao executar: {e}")
        else:
            print("❌ Erro na verificação. Tente instalar manualmente.")
    else:
        print("❌ Erro na instalação. Verifique sua conexão e tente novamente.")

if __name__ == "__main__":
    main()
'''

with open('instalar.py', 'w', encoding='utf-8') as f:
    f.write(script_instalacao)

# Criando guia de execução rápida
guia_execucao = '''# 📈 Sistema de Análise Fundamentalista de Ações

## 🚀 INSTALAÇÃO E EXECUÇÃO RÁPIDA

### Opção 1: Instalação Automática (Recomendado)

1. **Execute o instalador:**
   ```bash
   python instalar.py
   ```

2. **O script fará tudo automaticamente:**
   - Instala todas as dependências
   - Verifica a instalação
   - Oferece executar o sistema

### Opção 2: Instalação Manual

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute o sistema:**
   ```bash
   streamlit run analisador_acoes_completo.py
   ```

3. **Acesse no navegador:**
   ```
   http://localhost:8501
   ```

## 📋 Arquivos do Projeto

```
📁 pasta-do-projeto/
├── analisador_acoes_completo.py    # Sistema principal (ÚNICO ARQUIVO NECESSÁRIO)
├── requirements.txt                # Dependências
├── instalar.py                     # Instalador automático
└── README_EXECUCAO.md             # Este guia
```

## 🎯 Como Usar

1. **Abra o sistema** no navegador
2. **Digite uma ação** na barra lateral (ex: PETR4, VALE3, ITUB4)
3. **Selecione o período** de análise (1, 2 ou 5 anos)
4. **Clique em "Analisar Ação"**
5. **Veja os resultados:**
   - Preço atual e variação
   - Recomendação (COMPRAR/VENDER/MANTER)
   - Análise de risco (BAIXO/MÉDIO/ALTO)
   - Gráficos interativos
   - Indicadores fundamentalistas

## 🔧 Solução de Problemas

### ❌ Erro "ModuleNotFoundError"
```bash
# Reinstale as dependências
pip install --upgrade -r requirements.txt
```

### ❌ Erro "Port 8501 is already in use"  
```bash
# Use outra porta
streamlit run analisador_acoes_completo.py --server.port 8502
```

### ❌ Dados não carregam
- Verifique sua conexão com a internet
- Teste com ações populares: PETR4, VALE3, ITUB4
- Aguarde alguns segundos para carregamento

### ❌ Python não encontrado
1. Instale o Python 3.7+ do site oficial: https://python.org
2. Marque a opção "Add to PATH" durante instalação
3. Reinicie o terminal/prompt

## 📊 Exemplos de Ações para Testar

### 🏦 Bancos
- ITUB4 (Itaú)
- BBDC4 (Bradesco)  
- BBAS3 (Banco do Brasil)

### 🛢️ Petróleo
- PETR4 (Petrobrás)
- PETR3 (Petrobrás)

### ⛏️ Mineração
- VALE3 (Vale)
- GGBR4 (Gerdau)

### 🍺 Consumo
- ABEV3 (Ambev)
- JBSS3 (JBS)

### 🏭 Indústria
- WEGE3 (WEG)
- SUZB3 (Suzano)

## 💡 Recursos do Sistema

- ✅ **Dados em Tempo Real** via Yahoo Finance API
- ✅ **Análise Fundamentalista** com 6+ indicadores
- ✅ **Avaliação de Risco** quantitativa
- ✅ **Recomendações Inteligentes** com justificativas
- ✅ **Gráficos Interativos** com Plotly
- ✅ **Interface Responsiva** com Streamlit
- ✅ **Cache Inteligente** para performance

## 🆘 Suporte

Se encontrar problemas:

1. **Verifique os requisitos:**
   - Python 3.7 ou superior
   - Conexão com internet
   - Todas as dependências instaladas

2. **Teste o exemplo básico:**
   ```python
   import yfinance as yf
   print(yf.Ticker("PETR4.SA").info['regularMarketPrice'])
   ```

3. **Reinstale tudo:**
   ```bash
   pip uninstall -y streamlit yfinance pandas numpy plotly
   python instalar.py
   ```

---
**🎉 Sistema pronto para uso! Boa análise!**
'''

with open('README_EXECUCAO.md', 'w', encoding='utf-8') as f:
    f.write(guia_execucao)

# Criando script de teste rápido
teste_rapido = '''#!/usr/bin/env python3
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
        print("\\n🔌 Testando conexão com API...")
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
        print("\\n📊 Testando indicadores...")
        pe_ratio = info.get('trailingPE', 'N/A')
        pb_ratio = info.get('priceToBook', 'N/A')
        dividend_yield = info.get('dividendYield', 0)
        
        print(f"   P/L: {pe_ratio}")
        print(f"   P/VP: {pb_ratio}")  
        print(f"   Dividend Yield: {dividend_yield}")
        
        # Testar cálculos de risco
        print("\\n⚡ Testando cálculos de risco...")
        retornos = hist['Close'].pct_change().dropna()
        if len(retornos) > 1:
            volatilidade = retornos.std() * np.sqrt(252)
            print(f"✅ Volatilidade calculada: {volatilidade:.3f}")
        else:
            print("❌ Dados insuficientes para cálculo de volatilidade")
            return False
        
        print("\\n" + "=" * 40)
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Sistema funcionando corretamente")
        print("\\n🚀 Execute o comando abaixo para usar:")
        print("   streamlit run analisador_acoes_completo.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Módulo não encontrado: {e}")
        print("\\n💡 Execute: python instalar.py")
        return False
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        print("\\n💡 Verifique sua conexão com a internet")
        return False

if __name__ == "__main__":
    teste_sistema()
'''

with open('testar.py', 'w', encoding='utf-8') as f:
    f.write(teste_rapido)

print("📦 Arquivos auxiliares criados:")
print("   ✅ requirements.txt - Dependências")
print("   ✅ instalar.py - Instalador automático")  
print("   ✅ README_EXECUCAO.md - Guia de execução")
print("   ✅ testar.py - Teste rápido")
print("\n🎯 INSTRUÇÕES PARA EXECUTAR NO SEU COMPUTADOR:")
print("="*60)
print("1️⃣  Baixe todos os arquivos para uma pasta")
print("2️⃣  Abra o terminal/prompt nesta pasta") 
print("3️⃣  Execute: python instalar.py")
print("4️⃣  O sistema instalará tudo e perguntará se quer executar")
print("5️⃣  Acesse: http://localhost:8501")
print("\n💡 OU execute manualmente:")
print("   pip install -r requirements.txt")
print("   streamlit run analisador_acoes_completo.py")
print("\n🧪 Para testar sem interface:")
print("   python testar.py")