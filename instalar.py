#!/usr/bin/env python3
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

    print("\n📦 Instalando dependências...")

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

    print("\n🎉 Todas as dependências foram instaladas com sucesso!")
    return True

def verificar_instalacao():
    """Verifica se todas as dependências estão funcionando"""

    print("\n🔍 Verificando instalação...")

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
            print("\n" + "=" * 50)
            print("🎉 INSTALAÇÃO CONCLUÍDA COM SUCESSO!")
            print("=" * 50)
            print("\n🚀 Para executar o sistema:")
            print("   streamlit run analisador_acoes_completo.py")
            print("\n🌐 O sistema abrirá no seu navegador automaticamente")
            print("   Endereço: http://localhost:8501")

            # Perguntar se quer executar agora
            resposta = input("\n❓ Deseja executar o sistema agora? (s/n): ").lower().strip()
            if resposta in ['s', 'sim', 'y', 'yes']:
                print("🚀 Iniciando sistema...")
                try:
                    subprocess.run([
                        sys.executable, "-m", "streamlit", "run", 
                        "analisador_acoes_completo.py"
                    ])
                except KeyboardInterrupt:
                    print("\n👋 Sistema encerrado pelo usuário")
                except Exception as e:
                    print(f"❌ Erro ao executar: {e}")
        else:
            print("❌ Erro na verificação. Tente instalar manualmente.")
    else:
        print("❌ Erro na instalação. Verifique sua conexão e tente novamente.")

if __name__ == "__main__":
    main()
