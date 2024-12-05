import numpy as np
import matplotlib.pyplot as plt

# Função degrau (HeavySide)
def degrau(x): 
    return 1 if x >= 0 else -1

class Perceptron:
    def __init__(self, epocas=2000, taxa_aprendizado=0.1):
        self.epocas = epocas
        self.taxa_aprendizado = taxa_aprendizado
        self.pesos = None
        self.bias = None
        self.erros_por_epoca = []

    # Método de treinamento      
    def fit(self, dados, rotulos):
        amostras, caracteristicas = np.shape(dados)
        self.pesos = np.zeros(caracteristicas)
        self.bias = 0
        self.erros_por_epoca = []
        
        for _ in range(self.epocas):
            erro_total = 0
            for i, dados_i in enumerate(dados):
                comb_linear = np.dot(dados_i, self.pesos) + self.bias
                rotulo_calculado = degrau(comb_linear)
                
                erro = rotulos[i] - rotulo_calculado
                erro_total += abs(erro)
                
                correcao = self.taxa_aprendizado * erro
                self.pesos += correcao * dados_i
                self.bias += correcao
            
            # Registrar o erro total por época
            self.erros_por_epoca.append(erro_total)
    
    def predict(self, dados):
        comb_linear = np.dot(dados, self.pesos) + self.bias
        return np.array([degrau(x) for x in comb_linear])

# Gerar o dataset Gaussiano com maior distância entre as classes
num_samples = 1000


mean_class1 = [3, 3]  # Aumentar a distância entre as classes
mean_class2 = [-3, -3]  # Aumentar a distância entre as classes
covariance = [[1, 0], [0, 1]]

# Gerar as classes
class1 = np.random.multivariate_normal(mean_class1, covariance, num_samples // 2)
class2 = np.random.multivariate_normal(mean_class2, covariance, num_samples // 2)

labels_class1 = np.ones(len(class1))
labels_class2 = -np.ones(len(class2))

# Adicionar outliers - pontos aleatórios distantes
outliers_class1 = np.random.uniform(low=0.5, high=2, size=(8, 2))  # Outliers para a classe 1
outliers_class2 = np.random.uniform(low=0.7, high=1.2, size=(8, 2))  # Outliers para a classe 2

# Adicionar outliers aos dados
class1 = np.vstack((class1, outliers_class1))
class2 = np.vstack((class2, outliers_class2))

labels_class1 = np.ones(len(class1))
labels_class2 = -np.ones(len(class2))

data = np.vstack((class1, class2))
labels = np.hstack((labels_class1, labels_class2))

# Dividir o dataset em treinamento (40%) e validação (60%)
np.random.seed(42)  # Para reprodutibilidade
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

# Converter os rótulos para {0, 1} para calcular acurácia
y_val_binary = (y_val == 1).astype(int)
y_pred_binary = (y_pred == 1).astype(int)

accuracy = np.mean(y_val_binary == y_pred_binary) * 100
print(f"Acurácia no conjunto de validação: {accuracy:.2f}%")

# Função para calcular métricas de avaliação
def calcular_metricas(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
    
    return TPR, FPR, FN

# Inicializar listas para armazenar os valores de TPR, FPR e FN ao variar o limiar de decisão
TPR_values = []
FPR_values = []
FN_values = []

# Testar diferentes limiares de decisão para calcular as métricas
limiares = np.linspace(-5, 5, 100)
for limiar in limiares:
    y_pred_limiar = (np.dot(X_val, perceptron.pesos) + perceptron.bias) >= limiar
    y_pred_binary = y_pred_limiar.astype(int) * 2 - 1  # Convertendo de 0/1 para -1/1
    TPR, FPR, FN = calcular_metricas(y_val, y_pred_binary)
    TPR_values.append(TPR)
    FPR_values.append(FPR)
    FN_values.append(FN)

# Escolher a visualização
while True:
    
    print("Escolha a visualização:")
    print("1 - Gráfico de erro por época")
    print("2 - Gráfico de acurácia")
    print("3 - Classes divididas no plano cartesiano")
    print("4 - TPR por FPR")
    print("5 - FN por FPR")
    print("0 - Para sair")
    opcao = int(input("Digite o número da opção desejada: "))

    if opcao == 1:
        # Gráfico de erro por época
        plt.figure(figsize=(8, 6))
        plt.plot(perceptron.erros_por_epoca, label="Erro total por época")
        plt.title("Erro por época durante o treinamento")
        plt.xlabel("Épocas")
        plt.ylabel("Erro total")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    elif opcao == 2:
        # Gráfico de acurácia e erro no conjunto de validação
        plt.figure(figsize=(8, 6))
        plt.bar(["Acurácia (%)", "Erro (%)"], [accuracy, 100 - accuracy], color=['green', 'red'])
        plt.title("Acurácia e erro no conjunto de validação")
        plt.ylabel("Porcentagem")
        plt.grid(axis='y')
        plt.show()

    elif opcao == 3:
        # Visualização do conjunto de validação com decisão do Perceptron
        plt.figure(figsize=(8, 6))
        plt.scatter(X_val[y_val == 1][:, 0], X_val[y_val == 1][:, 1], color='blue', label='Classe 1 (True)')
        plt.scatter(X_val[y_val == -1][:, 0], X_val[y_val == -1][:, 1], color='red', label='Classe 2 (True)')
        
        # Decisão do perceptron
        x_min, x_max = X_val[:, 0].min() - 1, X_val[:, 0].max() + 1
        y_min, y_max = X_val[:, 1].min() - 1, X_val[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
        plt.title("Decisão do Perceptron no Conjunto de Validação")
        plt.xlabel("Dimensão 1")
        plt.ylabel("Dimensão 2")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif opcao == 4:
        # Gráfico TPR vs FPR
        plt.figure(figsize=(8, 6))
        plt.plot(FPR_values, TPR_values, label="Curva ROC", color='blue')
        plt.title("TPR por FPR")
        plt.xlabel("FPR (False Positive Rate)")
        plt.ylabel("TPR (True Positive Rate)")
        plt.grid(True)
        plt.legend()
        plt.show()

    elif opcao == 5:
        # Gráfico FN vs FPR
        plt.figure(figsize=(8, 6))
        plt.plot(FPR_values, FN_values, label="Falsos Negativos por FPR", color='red')
        plt.title("FN por FPR")
        plt.xlabel("FPR (False Positive Rate)")
        plt.ylabel("Falsos Negativos (FN)")
        plt.grid(True)
        plt.legend()
        plt.show()

    elif opcao == 0:
        print("Volte sempre! : )")
    
    else:
        print("\nUSUARIO BOBO, LEIA LEIA\n")
        
    if opcao == 0:
        break
