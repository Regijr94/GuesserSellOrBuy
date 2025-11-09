#!/usr/bin/env python3
"""
Teste de Configuração de GPU para TensorFlow
============================================

Este script testa se o TensorFlow está configurado corretamente
para usar a GPU NVIDIA.
"""

import os
import warnings
warnings.filterwarnings('ignore')

def configurar_gpu():
    """Configura o TensorFlow para usar GPU"""
    print("🔧 Configurando TensorFlow para GPU...")
    
    # Configurar variáveis de ambiente para GPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduzir logs
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usar primeira GPU
    
    # Configurar memória GPU
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    print("✅ Configurações de GPU aplicadas")

def testar_tensorflow_gpu():
    """Testa se o TensorFlow está usando GPU"""
    try:
        import tensorflow as tf
        
        print(f"\n📊 Informações do TensorFlow:")
        print(f"Versão: {tf.__version__}")
        
        # Verificar se GPU está disponível
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPUs disponíveis: {len(gpus)}")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            
            # Configurar crescimento de memória
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ Crescimento de memória GPU configurado")
            except RuntimeError as e:
                print(f"⚠️ Erro ao configurar memória GPU: {e}")
            
            # Testar operação simples na GPU
            print("\n🧪 Testando operação na GPU...")
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"Resultado da multiplicação: {c.numpy()}")
                print("✅ Operação executada com sucesso na GPU!")
            
            return True
        else:
            print("❌ Nenhuma GPU disponível")
            return False
            
    except ImportError:
        print("❌ TensorFlow não está instalado")
        return False
    except Exception as e:
        print(f"❌ Erro ao testar GPU: {e}")
        return False

def testar_modelo_simples():
    """Testa um modelo simples na GPU"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        print("\n🤖 Testando modelo simples na GPU...")
        
        # Criar modelo simples
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Dados de teste
        import numpy as np
        X_test = np.random.random((100, 10))
        y_test = np.random.randint(0, 2, (100, 1))
        
        # Treinar modelo
        print("📈 Treinando modelo...")
        history = model.fit(
            X_test, y_test,
            epochs=5,
            batch_size=32,
            verbose=0
        )
        
        print(f"✅ Modelo treinado com sucesso!")
        print(f"Loss final: {history.history['loss'][-1]:.4f}")
        print(f"Accuracy final: {history.history['accuracy'][-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar modelo: {e}")
        return False

def main():
    """Função principal"""
    print("🚀 TESTE DE CONFIGURAÇÃO DE GPU PARA TENSORFLOW")
    print("=" * 55)
    
    # Configurar GPU
    configurar_gpu()
    
    # Testar TensorFlow com GPU
    gpu_ok = testar_tensorflow_gpu()
    
    if gpu_ok:
        # Testar modelo simples
        modelo_ok = testar_modelo_simples()
        
        if modelo_ok:
            print("\n🎉 SUCESSO! GPU configurada e funcionando!")
            print("💡 O sistema de deep learning pode usar GPU para treinamento")
        else:
            print("\n⚠️ GPU detectada mas modelo falhou")
    else:
        print("\n❌ GPU não está funcionando")
        print("💡 O sistema usará CPU para treinamento")
    
    print("\n" + "=" * 55)

if __name__ == "__main__":
    main()


