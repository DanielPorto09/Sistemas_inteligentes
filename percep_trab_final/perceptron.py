import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from base_linear import data, labels
from base_espiral import datae, labelse 
from base_normal import datap, labelsp


# Função sigmoide
def degrau(x): 
    return 1 if x >= 0 else -1

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

class Perceptron:
    def __init__(self, epocas=2000, taxa_aprendizado=0.1):
        """#+
        Inicializa um modelo Perceptron com os parâmetros fornecidos.#+
#+
        Parameters:#+
        epocas (int, opcional): O número de épocas de treinamento. Padrão é 2000.#+
        taxa_aprendizado (float, opcional): A taxa de aprendizado. Padrão é 0.1.#+
#+
        Returns:#+
        None#+
        """#+
        self.epocas = epocas
        self.taxa_aprendizado = taxa_aprendizado
        self.pesos = None
        self.bias = None
        self.erros_por_epoca = []

    # Método de treinamento      
    def fit(self, dados, rotulos):
        """#+
        Treina o modelo Perceptron usando o conjunto de dados fornecido.#+
#+
        Parameters:#+
        dados (numpy.ndarray): Um objeto bidimensional semelhante a um array que contém as amostras de entrada. Cada linha representa uma amostra, e cada coluna representa uma característica.#+
        rotulos (numpy.ndarray): Um objeto unidimensional semelhante a um array que contém os rótulos correspondentes para as amostras de entrada.#+
#+
        Returns:#+
        None#+
        """#+
        amostras, caracteristicas = np.shape(dados)
        self.pesos = np.zeros(caracteristicas)
        self.bias = 0
        self.erros_por_epoca = []

        for _ in range(self.epocas):
            erro_total = 0
            for i, dados_i in enumerate(dados):
                comb_linear = np.dot(dados_i, self.pesos) + self.bias
                #rotulo_calculado = np.round(sigmoide(comb_linear)) 
                rotulo_calculado = np.array(degrau(comb_linear))

                erro = rotulos[i] - rotulo_calculado
                erro_total += abs(erro)

                correcao = self.taxa_aprendizado * erro
                self.pesos += correcao * dados_i
                self.bias += correcao

            self.erros_por_epoca.append(erro_total)
    
    def predict(self, dados):
        """#+
        Prevê os rótulos das classes para as amostras de entrada fornecidas usando o modelo Perceptron treinado.#+
#+
        Parameters:#+
        dados (numpy.ndarray): Um objeto bidimensional semelhante a um array que contém as amostras de entrada. Cada linha representa uma amostra, e cada coluna representa uma característica.#+
#+
        Returns:#+
        numpy.ndarray: Um objeto unidimensional semelhante a um array contendo os rótulos das classes previstos para as amostras de entrada. Os rótulos são 1 ou -1.#+
        """#+
        comb_linear = np.dot(dados, self.pesos) + self.bias
        #return np.array(sigmoide(comb_linear)) 
        return np.array([degrau(x) for x in comb_linear])



# Interface de execução
while True:
    print("\nEscolha a visualização:")
    time.sleep(0.2)
    print("1 - Base Linearmente Separável")
    time.sleep(0.2)
    print("2 - Base Não Separavel tipo 1")
    time.sleep(0.2)
    print("3 - Base Quase Normalmente Distribuida")
    time.sleep(0.2)
    print("4 - Base Linearmente Separável Sem Randomização")
    time.sleep(0.2)
    print("0 - Sair")
    opcao = int(input("Digite o número da opção desejada: "))
   
    if opcao == 1:
        # Dividir o dataset em treinamento (40%) e validação (60%)
        np.random.seed(42)
        indices = np.random.permutation(len(data))

        train_size = int(0.4 * len(data))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        X_train, y_train = data[train_indices], labels[train_indices]
        X_val, y_val = data[val_indices], labels[val_indices]

        # Treinar o Perceptron
        perceptron = Perceptron(epocas=1000, taxa_aprendizado=0.1)
        perceptron.fit(X_train, y_train)

        # Avaliar o Perceptron no conjunto de validação
        y_pred = perceptron.predict(X_val)

        # Calcular métricas de desempenho
        #accuracy = accuracy_score(y_val, y_pred) * 100
        accuracy = np.mean(y_val == y_pred) * 100
        print(f"Acurácia no conjunto de validação: {accuracy:.2f}%")

        # Calcular Curva ROC e AUC usando sklearn
        fpr, tpr, thresholds = roc_curve(y_val, y_pred)
        roc_auc = auc(fpr, tpr)

        # Criar os gráficos lado a lado
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # Gráfico de decisão
        axs[0].scatter(X_val[y_val == 1][:, 0], X_val[y_val == 1][:, 1], color='blue', label='Classe 1')
        axs[0].scatter(X_val[y_val == -1][:, 0], X_val[y_val == -1][:, 1], color='red', label='Classe 2')
        x_min, x_max = X_val[:, 0].min() - 1, X_val[:, 0].max() + 1
        y_min, y_max = X_val[:, 1].min() - 1, X_val[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axs[0].contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
        axs[0].set_title("Decisão do Perceptron")
        axs[0].set_xlabel("Dimensão 1")
        axs[0].set_ylabel("Dimensão 2")
        axs[0].legend()
        axs[0].grid(True)

        # Gráfico ROC
        axs[1].plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
        axs[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
        axs[1].set_title("Curva ROC")
        axs[1].set_xlabel("FPR")
        axs[1].set_ylabel("TPR")
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plt.show() 
    
    elif opcao == 2:
        
        # Dividir o dataset em 40% treino e 60% validação
        indices = np.random.permutation(len(datae))
        train_size = int(0.4 * len(datae))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        X_train, y_train = datae[train_indices], labelse[train_indices]
        X_val, y_val = datae[val_indices], labelse[val_indices]

        # Treinar o Perceptron
        perceptron = Perceptron(epocas=1000, taxa_aprendizado=0.1)
        perceptron.fit(X_train, y_train)

        # Avaliar no conjunto de validação
        y_pred = perceptron.predict(X_val)

        # Cálculo de acurácia
        accuracy = np.mean((y_val == y_pred).astype(int)) * 100
        print(f"Acurácia no conjunto de validação (Espiral): {accuracy:.2f}%")

        # Cálculo de TPR (True Positive Rate) e FPR (False Positive Rate)
        fpr, tpr, thresholds = roc_curve(y_val, y_pred)
        roc_auc = auc(fpr, tpr)

        # Visualizar a decisão do Perceptron
        plt.figure(figsize=(12, 6))

        # Gráfico do Perceptron (base espiral)
        plt.subplot(1, 2, 1)
        # Verificar se as classes estão corretamente representadas (exemplo: azul e vermelho)
        plt.scatter(X_val[y_val == 1][:, 0], X_val[y_val == 1][:, 1], color='blue', label='Classe 1 (True)', s=50)
        plt.scatter(X_val[y_val == -1][:, 0], X_val[y_val == -1][:, 1], color='red', label='Classe -1 (True)', s=50)

        # Limites para o plano de decisão
        x_min, x_max = X_val[:, 0].min() - 1, X_val[:, 0].max() + 1
        y_min, y_max = X_val[:, 1].min() - 1, X_val[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plotando a região de decisão
        plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
        plt.title("Decisão do Perceptron - Base Espiral")
        plt.xlabel("Dimensão 1")
        plt.ylabel("Dimensão 2")
        plt.legend()
        plt.grid(True)

        # Curva ROC - TPR vs FPR
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.title("Curva ROC - Espiral")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    elif opcao == 3:
        # Dividir o dataset em 40% treino e 60% validação
        indices = np.random.permutation(len(datap))
        train_size = int(0.4 * len(datap))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        X_train, y_train = datap[train_indices], labelsp[train_indices]
        X_val, y_val = datap[val_indices], labelsp[val_indices]

        # Treinar o Perceptron
        perceptron = Perceptron(epocas=1000, taxa_aprendizado=0.1)
        perceptron.fit(X_train, y_train)

        # Avaliar no conjunto de validação
        y_pred = perceptron.predict(X_val)

        # Cálculo da acurácia
        accuracy = np.mean((y_val == y_pred).astype(int)) * 100
        print(f"Acurácia no conjunto de validação (Normal): {accuracy:.2f}%")

        # Cálculo da curva ROC e AUC
        fpr, tpr, thresholds = roc_curve(y_val, y_pred)
        roc_auc = auc(fpr, tpr)

        # Criar gráficos
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # Gráfico de decisão
        axs[0].scatter(X_val[y_val == 1][:, 0], X_val[y_val == 1][:, 1], color='blue', label='Classe 1')
        axs[0].scatter(X_val[y_val == -1][:, 0], X_val[y_val == -1][:, 1], color='red', label='Classe -1')
        x_min, x_max = X_val[:, 0].min() - 1, X_val[:, 0].max() + 1
        y_min, y_max = X_val[:, 1].min() - 1, X_val[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axs[0].contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
        axs[0].set_title("Decisão do Perceptron - Base Normal")
        axs[0].set_xlabel("Dimensão 1")
        axs[0].set_ylabel("Dimensão 2")
        axs[0].legend()
        axs[0].grid(True)

        # Gráfico ROC
        axs[1].plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
        axs[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
        axs[1].set_title("Curva ROC - Base Normal")
        axs[1].set_xlabel("FPR")
        axs[1].set_ylabel("TPR")
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plt.show()
    elif opcao == 4:
        data_40, data_60, labels_40, labels_60 = train_test_split(
        data, labels, test_size=0.6, stratify=labels, random_state=42   
        )

        #Treinamento
        perceptron = Perceptron(epocas=1000, taxa_aprendizado=0.1)
        perceptron.fit(data_40, labels_40)

        #Avaliação
        predict_labels = perceptron.predict(data_60)

        #acuracia
        y_val_binary = (labels_60 == 1).astype(int)
        y_pred_binary = (predict_labels == 1).astype(int)

        accuracy = np.mean(y_val_binary == y_pred_binary) * 100
        print(f"Acurácia no conjunto de validação: {accuracy:.2f}%")

        # Cálculo de TPR (True Positive Rate) e FPR (False Positive Rate)
        fpr, tpr, thresholds = roc_curve(y_val_binary, y_pred_binary)
        roc_auc = auc(fpr, tpr)

        # Visualizar a decisão do Perceptron
        plt.figure(figsize=(12, 6))

        # Gráfico do Perceptron
        plt.subplot(1, 2, 1)
        # Apenas para ilustração: ajustando as duas primeiras dimensões dos dados para visualização
        if data.shape[1] > 2:
            print("Visualizando apenas as duas primeiras dimensões dos dados.")
            X_val_plot = data_60[:, :2]
        else:
            X_val_plot = data_60

        plt.scatter(X_val_plot[labels_60 == 1][:, 0], X_val_plot[labels_60 == 1][:, 1], color='blue', label='Classe 1 (True)')
        plt.scatter(X_val_plot[labels_60 == -1][:, 0], X_val_plot[labels_60 == -1][:, 1], color='red', label='Classe -1 (True)')

        # Limites para o plano
        x_min, x_max = X_val_plot[:, 0].min() - 1, X_val_plot[:, 0].max() + 1
        y_min, y_max = X_val_plot[:, 1].min() - 1, X_val_plot[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
        plt.title("Decisão do Perceptron")
        plt.xlabel("Dimensão 1")
        plt.ylabel("Dimensão 2")
        plt.legend()
        plt.grid(True)

        # Curva ROC - TPR vs FPR
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.title("Curva ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc="lower right")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    
    elif opcao == 0:
        break
    
    else:
        print("Opção inválida")
