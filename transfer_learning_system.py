#!/usr/bin/env python3
"""
Sistema de Transfer Learning e Validação Cruzada Temporal
========================================================

Este módulo implementa técnicas avançadas de transfer learning
e validação cruzada temporal para otimizar o treinamento de
redes neurais em diferentes ações.
"""

import numpy as np
import pandas as pd
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

from deep_learning_system import SistemaDeepLearning, BacktestHistorico

class TransferLearningSystem:
    """Sistema de transfer learning para diferentes ações"""
    
    def __init__(self, acao_base: str = "PETR4"):
        self.acao_base = acao_base
        self.modelo_base = None
        self.features_base = None
        self.historico_treinamento = {}
        
    def treinar_modelo_base(self, periodo_treinamento: int = 5) -> Dict:
        """Treina modelo base usando dados históricos extensos"""
        print(f"🏗️ Treinando modelo base com {self.acao_base}...")
        
        # Configurar sistema base
        sistema_base = SistemaDeepLearning(self.acao_base)
        
        # Usar período mais longo para modelo base
        sistema_base.backtest.periodo_inicio = "2010-01-01"
        
        # Configurar com features robustas
        features_base = [
            'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR',
            'Volume_SMA', 'Price_Change', 'Volatility', 'P_L_ratio',
            'P_VP_ratio', 'ROE', 'Dividend_Yield', 'Debt_to_Equity'
        ]
        
        # Preparar dados
        X_train, X_test, y_train, y_test = sistema_base.configurar_sistema(
            features=features_base,
            target="Return_5d",
            janela_temporal=60,
            split_temporal=0.8
        )
        
        # Treinar com mais épocas para modelo base
        resultados = sistema_base.treinar_sistema(
            X_train, y_train, X_test, y_test,
            epochs=200, batch_size=64
        )
        
        # Salvar modelo base
        self.modelo_base = sistema_base.ensemble
        self.features_base = features_base
        
        # Salvar histórico
        self.historico_treinamento[self.acao_base] = {
            'sistema': sistema_base,
            'resultados': resultados,
            'timestamp': datetime.now()
        }
        
        print(f"✅ Modelo base treinado - R²: {resultados['avaliacao']['r2_score']:.4f}")
        
        return resultados
    
    def transferir_para_acao(self, ticker_destino: str, 
                           fine_tuning: bool = True,
                           epochs_fine_tuning: int = 50) -> Dict:
        """Transfere conhecimento para nova ação"""
        if self.modelo_base is None:
            raise ValueError("Modelo base não foi treinado")
        
        print(f"🔄 Transferindo conhecimento para {ticker_destino}...")
        
        # Configurar sistema destino
        sistema_destino = SistemaDeepLearning(ticker_destino)
        
        # Usar período mais curto para fine-tuning
        sistema_destino.backtest.periodo_inicio = "2020-01-01"
        
        # Preparar dados destino
        X_train, X_test, y_train, y_test = sistema_destino.configurar_sistema(
            features=self.features_base,
            target="Return_5d",
            janela_temporal=60,
            split_temporal=0.8
        )
        
        if fine_tuning:
            # Fine-tuning do modelo base
            print("🔧 Aplicando fine-tuning...")
            
            # Copiar arquitetura do modelo base
            sistema_destino.ensemble = self._copiar_ensemble(self.modelo_base)
            
            # Reduzir learning rate para fine-tuning
            for modelo in sistema_destino.ensemble.models.values():
                if hasattr(modelo, 'model') and modelo.model is not None:
                    modelo.learning_rate *= 0.1  # Reduzir LR para fine-tuning
            
            # Treinar com menos épocas
            resultados = sistema_destino.treinar_sistema(
                X_train, y_train, X_test, y_test,
                epochs=epochs_fine_tuning, batch_size=32
            )
        else:
            # Treinar do zero
            print("🆕 Treinando do zero...")
            resultados = sistema_destino.treinar_sistema(
                X_train, y_train, X_test, y_test,
                epochs=100, batch_size=32
            )
        
        # Salvar histórico
        self.historico_treinamento[ticker_destino] = {
            'sistema': sistema_destino,
            'resultados': resultados,
            'fine_tuning': fine_tuning,
            'timestamp': datetime.now()
        }
        
        print(f"✅ Transferência concluída - R²: {resultados['avaliacao']['r2_score']:.4f}")
        
        return resultados
    
    def _copiar_ensemble(self, ensemble_origem) -> Any:
        """Copia arquitetura do ensemble para fine-tuning"""
        # Esta é uma implementação simplificada
        # Em produção, seria necessário salvar/carregar pesos
        from deep_learning_system import EnsembleDeepLearning, LSTMPredictor, CNNPredictor
        
        # Criar novo ensemble com mesma arquitetura
        input_shape = ensemble_origem.input_shape
        novo_ensemble = EnsembleDeepLearning(input_shape)
        
        # Copiar modelos
        for nome, modelo_origem in ensemble_origem.models.items():
            if "LSTM" in nome:
                novo_modelo = LSTMPredictor(
                    input_shape=input_shape,
                    lstm_units=[50, 50, 25],
                    dropout_rate=0.2,
                    learning_rate=modelo_origem.learning_rate
                )
            elif "CNN" in nome:
                novo_modelo = CNNPredictor(
                    input_shape=input_shape,
                    filters=[32, 64, 128],
                    kernel_size=3,
                    dropout_rate=0.2,
                    learning_rate=modelo_origem.learning_rate
                )
            else:
                continue
            
            novo_ensemble.adicionar_modelo(nome, novo_modelo, ensemble_origem.weights[nome])
        
        return novo_ensemble
    
    def comparar_performance(self, tickers: List[str]) -> Dict:
        """Compara performance entre diferentes ações"""
        resultados_comparacao = {}
        
        for ticker in tickers:
            if ticker in self.historico_treinamento:
                resultados = self.historico_treinamento[ticker]['resultados']
                resultados_comparacao[ticker] = {
                    'r2_score': resultados['avaliacao']['r2_score'],
                    'rmse': resultados['avaliacao']['rmse'],
                    'mae': resultados['avaliacao']['mae'],
                    'fine_tuning': self.historico_treinamento[ticker].get('fine_tuning', False)
                }
        
        return resultados_comparacao

class ValidacaoCruzadaTemporal:
    """Sistema de validação cruzada temporal para backtest"""
    
    def __init__(self, ticker: str, n_folds: int = 5):
        self.ticker = ticker
        self.n_folds = n_folds
        self.resultados_folds = {}
        
    def executar_validacao_cruzada(self, features: List[str], target: str = "Return_5d",
                                 janela_temporal: int = 60) -> Dict:
        """Executa validação cruzada temporal"""
        print(f"🔄 Executando validação cruzada temporal para {self.ticker}...")
        
        # Obter dados históricos
        backtest = BacktestHistorico(self.ticker, "2015-01-01")
        dados = backtest.obter_dados_historicos()
        
        # Remover NaN
        dados_limpos = dados.dropna()
        
        # Preparar dados
        X = dados_limpos[features].values
        y = dados_limpos[target].values
        
        # Criar sequências temporais
        X_seq, y_seq = [], []
        for i in range(janela_temporal, len(X)):
            X_seq.append(X[i-janela_temporal:i])
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Dividir em folds temporais
        fold_size = len(X_seq) // self.n_folds
        resultados_folds = []
        
        for fold in range(self.n_folds):
            print(f"📊 Processando fold {fold + 1}/{self.n_folds}...")
            
            # Definir índices do fold
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < self.n_folds - 1 else len(X_seq)
            
            # Dados de teste (fold atual)
            X_test_fold = X_seq[start_idx:end_idx]
            y_test_fold = y_seq[start_idx:end_idx]
            
            # Dados de treinamento (todos os dados anteriores)
            X_train_fold = X_seq[:start_idx] if start_idx > 0 else np.array([]).reshape(0, X_seq.shape[1], X_seq.shape[2])
            y_train_fold = y_seq[:start_idx] if start_idx > 0 else np.array([])
            
            if len(X_train_fold) == 0:
                print(f"⚠️ Fold {fold + 1}: Dados de treinamento insuficientes")
                continue
            
            # Treinar modelo para este fold
            resultado_fold = self._treinar_fold(
                X_train_fold, y_train_fold, X_test_fold, y_test_fold,
                fold, janela_temporal
            )
            
            resultados_folds.append(resultado_fold)
        
        # Compilar resultados
        self.resultados_folds = self._compilar_resultados(resultados_folds)
        
        return self.resultados_folds
    
    def _treinar_fold(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     fold: int, janela_temporal: int) -> Dict:
        """Treina modelo para um fold específico"""
        try:
            from deep_learning_system import SistemaDeepLearning, LSTMPredictor
            
            # Criar sistema temporário
            sistema_temp = SistemaDeepLearning(self.ticker)
            
            # Configurar ensemble simples (apenas LSTM para validação)
            input_shape = (X_train.shape[1], X_train.shape[2])
            from deep_learning_system import EnsembleDeepLearning
            
            ensemble_temp = EnsembleDeepLearning(input_shape)
            lstm_temp = LSTMPredictor(input_shape, lstm_units=[50, 25])
            ensemble_temp.adicionar_modelo("LSTM", lstm_temp, peso=1.0)
            
            sistema_temp.ensemble = ensemble_temp
            
            # Split para validação
            if len(X_train) > 100:
                split_idx = int(len(X_train) * 0.8)
                X_train_split = X_train[:split_idx]
                X_val_split = X_train[split_idx:]
                y_train_split = y_train[:split_idx]
                y_val_split = y_train[split_idx:]
            else:
                X_train_split = X_train
                X_val_split = X_test[:min(20, len(X_test))]
                y_train_split = y_train
                y_val_split = y_test[:min(20, len(y_test))]
            
            # Treinar
            resultados_treinamento = sistema_temp.ensemble.treinar_ensemble(
                X_train_split, y_train_split, X_val_split, y_val_split,
                epochs=50, batch_size=32
            )
            
            # Avaliar
            resultados_avaliacao = sistema_temp.ensemble.avaliar_ensemble(X_test, y_test)
            
            return {
                'fold': fold,
                'treinamento': resultados_treinamento,
                'avaliacao': resultados_avaliacao,
                'tamanho_treinamento': len(X_train),
                'tamanho_teste': len(X_test)
            }
            
        except Exception as e:
            print(f"❌ Erro no fold {fold}: {str(e)}")
            return {
                'fold': fold,
                'erro': str(e),
                'tamanho_treinamento': len(X_train),
                'tamanho_teste': len(X_test)
            }
    
    def _compilar_resultados(self, resultados_folds: List[Dict]) -> Dict:
        """Compila resultados de todos os folds"""
        folds_validos = [r for r in resultados_folds if 'avaliacao' in r]
        
        if not folds_validos:
            return {'erro': 'Nenhum fold válido'}
        
        # Métricas médias
        r2_scores = [r['avaliacao']['r2_score'] for r in folds_validos]
        rmse_scores = [r['avaliacao']['rmse'] for r in folds_validos]
        mae_scores = [r['avaliacao']['mae'] for r in folds_validos]
        
        return {
            'n_folds_validos': len(folds_validos),
            'n_folds_total': len(resultados_folds),
            'metricas_medias': {
                'r2_score_medio': np.mean(r2_scores),
                'r2_score_std': np.std(r2_scores),
                'rmse_medio': np.mean(rmse_scores),
                'rmse_std': np.std(rmse_scores),
                'mae_medio': np.mean(mae_scores),
                'mae_std': np.std(mae_scores)
            },
            'metricas_por_fold': {
                'r2_scores': r2_scores,
                'rmse_scores': rmse_scores,
                'mae_scores': mae_scores
            },
            'folds_detalhados': resultados_folds
        }
    
    def gerar_relatorio_validacao(self) -> Dict:
        """Gera relatório da validação cruzada"""
        if not self.resultados_folds:
            return {'erro': 'Validação cruzada não foi executada'}
        
        if 'erro' in self.resultados_folds:
            return self.resultados_folds
        
        metricas = self.resultados_folds['metricas_medias']
        
        # Classificar estabilidade
        r2_std = metricas['r2_score_std']
        if r2_std < 0.05:
            estabilidade = "MUITO_ESTAVEL"
        elif r2_std < 0.1:
            estabilidade = "ESTAVEL"
        elif r2_std < 0.2:
            estabilidade = "MODERADAMENTE_ESTAVEL"
        else:
            estabilidade = "INSTAVEL"
        
        return {
            'ticker': self.ticker,
            'n_folds': self.resultados_folds['n_folds_validos'],
            'metricas_medias': metricas,
            'estabilidade': estabilidade,
            'qualidade_geral': self._classificar_qualidade(metricas['r2_score_medio']),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _classificar_qualidade(self, r2_score: float) -> str:
        """Classifica qualidade baseada no R²"""
        if r2_score >= 0.7:
            return "EXCELENTE"
        elif r2_score >= 0.5:
            return "BOM"
        elif r2_score >= 0.3:
            return "REGULAR"
        elif r2_score >= 0.1:
            return "RUIM"
        else:
            return "MUITO_RUIM"

class SistemaAvancadoDeepLearning:
    """Sistema avançado combinando transfer learning e validação cruzada"""
    
    def __init__(self, acao_base: str = "PETR4"):
        self.acao_base = acao_base
        self.transfer_system = TransferLearningSystem(acao_base)
        self.validacao_system = None
        
    def configurar_sistema_completo(self, acoes_secundarias: List[str] = None) -> Dict:
        """Configura sistema completo com transfer learning"""
        if acoes_secundarias is None:
            acoes_secundarias = ["VALE3", "ITUB4", "BBDC4", "ABEV3"]
        
        print("🚀 Configurando sistema avançado de deep learning...")
        
        # Treinar modelo base
        resultado_base = self.transfer_system.treinar_modelo_base()
        
        # Transferir para ações secundárias
        resultados_transfer = {}
        for acao in acoes_secundarias:
            try:
                resultado = self.transfer_system.transferir_para_acao(acao)
                resultados_transfer[acao] = resultado
            except Exception as e:
                print(f"⚠️ Erro ao transferir para {acao}: {str(e)}")
                resultados_transfer[acao] = {'erro': str(e)}
        
        return {
            'modelo_base': resultado_base,
            'transferencias': resultados_transfer,
            'comparacao': self.transfer_system.comparar_performance([self.acao_base] + acoes_secundarias)
        }
    
    def executar_validacao_completa(self, ticker: str, n_folds: int = 5) -> Dict:
        """Executa validação cruzada completa"""
        self.validacao_system = ValidacaoCruzadaTemporal(ticker, n_folds)
        
        features = [
            'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR',
            'Volume_SMA', 'Price_Change', 'Volatility'
        ]
        
        resultados = self.validacao_system.executar_validacao_cruzada(features)
        relatorio = self.validacao_system.gerar_relatorio_validacao()
        
        return {
            'resultados': resultados,
            'relatorio': relatorio
        }

def exemplo_uso_avancado():
    """Exemplo de uso do sistema avançado"""
    print("🚀 EXEMPLO DO SISTEMA AVANÇADO DE DEEP LEARNING")
    print("=" * 60)
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow não está disponível")
        print("💡 Instale: pip install tensorflow")
        return
    
    # Configurar sistema avançado
    sistema_avancado = SistemaAvancadoDeepLearning("PETR4")
    
    try:
        # Configurar sistema completo
        print("🏗️ Configurando sistema completo...")
        resultados_completo = sistema_avancado.configurar_sistema_completo()
        
        print("\n📊 RESULTADOS DO SISTEMA COMPLETO")
        print("=" * 40)
        
        # Modelo base
        base_r2 = resultados_completo['modelo_base']['avaliacao']['r2_score']
        print(f"Modelo Base (PETR4): R² = {base_r2:.4f}")
        
        # Transferências
        print("\n🔄 Transferências:")
        for acao, resultado in resultados_completo['transferencias'].items():
            if 'erro' not in resultado:
                r2 = resultado['avaliacao']['r2_score']
                print(f"  {acao}: R² = {r2:.4f}")
            else:
                print(f"  {acao}: ERRO - {resultado['erro']}")
        
        # Validação cruzada
        print("\n🔄 Executando validação cruzada...")
        validacao = sistema_avancado.executar_validacao_completa("PETR4", n_folds=3)
        
        if 'erro' not in validacao['resultados']:
            relatorio = validacao['relatorio']
            print(f"\n📈 RELATÓRIO DE VALIDAÇÃO CRUZADA")
            print(f"Folds válidos: {relatorio['n_folds']}")
            print(f"R² médio: {relatorio['metricas_medias']['r2_score_medio']:.4f}")
            print(f"Estabilidade: {relatorio['estabilidade']}")
            print(f"Qualidade: {relatorio['qualidade_geral']}")
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}")

if __name__ == "__main__":
    exemplo_uso_avancado()
