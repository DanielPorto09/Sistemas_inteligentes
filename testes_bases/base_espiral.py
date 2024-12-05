import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Função degrau (Heaviside)
def degrau(x): 
    return 1 if x >= 0 else -1

class Perceptron:
    def __init__(self, epocas=2000, taxa_aprendizado=0.1):
        self.epocas = epocas
        self.taxa_aprendizado = taxa_aprendizado
        self.pesos = None
        self.bias = None
        self.erros_por_epoca = []

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
                
            self.erros_por_epoca.append(erro_total)
    
    def predict(self, dados):
        comb_linear = np.dot(dados, self.pesos) + self.bias
        return np.array([degrau(x) for x in comb_linear])

# Gerar base de dados em espiral
def gerar_espiral(n_pontos, ruido=0.5):
    theta = np.sqrt(np.random.rand(n_pontos)) * 2 * np.pi
    r_a = 2 * theta + np.pi
    r_b = -2 * theta - np.pi
    data_a = np.c_[r_a * np.cos(theta) + np.random.normal(0, ruido, n_pontos),
                   r_a * np.sin(theta) + np.random.normal(0, ruido, n_pontos)]
    data_b = np.c_[r_b * np.cos(theta) + np.random.normal(0, ruido, n_pontos),
                   r_b * np.sin(theta) + np.random.normal(0, ruido, n_pontos)]
    labels_a = np.ones(n_pontos)
    labels_b = -np.ones(n_pontos)
    return np.vstack((data_a, data_b)), np.hstack((labels_a, labels_b))

# Criar base de dados em espiral
np.random.seed(42)
data, labels = gerar_espiral(250, ruido=0.3)

# Dividir o dataset em 40% treino e 60% validação
indices = np.random.permutation(len(data))
train_size = int(0.4 * len(data))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

X_train, y_train = data[train_indices], labels[train_indices]
X_val, y_val = data[val_indices], labels[val_indices]

# Treinar o Perceptron
perceptron = Perceptron(epocas=1000, taxa_aprendizado=0.1)
perceptron.fit(X_train, y_train)

# Avaliar no conjunto de validação
y_pred = perceptron.predict(X_val)

# Cálculo de acurácia
accuracy = np.mean((y_val == y_pred).astype(int)) * 100
print(f"Acurácia no conjunto de validação (Espiral): {accuracy:.2f}%")

# Matriz de confusão
cm = confusion_matrix(y_val, y_pred)
print("Matriz de Confusão:")
print(cm)

# Cálculo de TPR (True Positive Rate) e FPR (False Positive Rate)
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)

# Visualizar a decisão do Perceptron
plt.figure(figsize=(12, 6))

# Gráfico do Perceptron
plt.subplot(1, 2, 1)
plt.scatter(X_val[y_val == 1][:, 0], X_val[y_val == 1][:, 1], color='blue', label='Classe 1 (True)')
plt.scatter(X_val[y_val == -1][:, 0], X_val[y_val == -1][:, 1], color='red', label='Classe -1 (True)')

# Limites para o plano
x_min, x_max = X_val[:, 0].min() - 1, X_val[:, 0].max() + 1
y_min, y_max = X_val[:, 1].min() - 1, X_val[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
plt.title("Decisão do Perceptron - Base Espiral")
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

# Gráfico de barras com erro e acurácia
plt.figure(figsize=(6, 4))
epochs = np.arange(1, perceptron.epocas + 1)
erro_medio = np.array(perceptron.erros_por_epoca) / len(X_train)
plt.bar(epochs, erro_medio, color='red', label="Erro")
plt.plot(epochs, [accuracy / 100] * perceptron.epocas, color='blue', label="Acurácia")
plt.title("Erro e Acurácia ao Longo das Épocas")
plt.xlabel("Época")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.show()
