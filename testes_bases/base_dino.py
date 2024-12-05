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
        self.acuracia_por_epoca = []

    def fit(self, dados, rotulos):
        amostras, caracteristicas = np.shape(dados)
        self.pesos = np.zeros(caracteristicas)
        self.bias = 0
        self.erros_por_epoca = []
        self.acuracia_por_epoca = []
        
        for _ in range(self.epocas):
            erro_total = 0
            acuracia_total = 0
            for i, dados_i in enumerate(dados):
                comb_linear = np.dot(dados_i, self.pesos) + self.bias
                rotulo_calculado = degrau(comb_linear)
                erro = rotulos[i] - rotulo_calculado
                erro_total += abs(erro)
                correcao = self.taxa_aprendizado * erro
                self.pesos += correcao * dados_i
                self.bias += correcao
                
                acuracia_total += (rotulo_calculado == rotulos[i])
            
            erro_medio = erro_total / len(dados)
            acuracia_media = acuracia_total / len(dados)
            self.erros_por_epoca.append(erro_medio)
            self.acuracia_por_epoca.append(acuracia_media)
    
    def predict(self, dados):
        comb_linear = np.dot(dados, self.pesos) + self.bias
        return np.array([degrau(x) for x in comb_linear])

# Base de dados com forma de dinossauro
def gerar_dinossauro():
    dinossauro_coords = [
        (-3, 1), (-2.5, 2), (-2, 3), (-1.5, 3.5), (-1, 3.7), (0, 3.5), (1, 3), (1.5, 2.5),
        (2, 1.8), (2.3, 1), (2, 0), (1.5, -1), (1, -1.8), (0.5, -2.5), (0, -3), (-0.5, -2.5),
        (-1, -1.8), (-1.5, -1), (-2, 0), (-2.5, 1)
    ]
    background_coords = np.random.uniform(-4, 4, size=(1000, 2))

    dinossauro_coords = np.array(dinossauro_coords)
    labels_dino = np.ones(len(dinossauro_coords))  # Classe 1 para o dinossauro
    labels_back = -np.ones(len(background_coords))  # Classe -1 para o fundo

    data = np.vstack((dinossauro_coords, background_coords))
    labels = np.hstack((labels_dino, labels_back))

    return data, labels

# Gerar a base de dados
datap, labelsp = gerar_dinossauro()

# Divisão em treino e validação
np.random.seed(42)
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
accuracy = np.mean((y_val == y_pred).astype(int)) * 100
print(f"Acurácia no conjunto de validação (Distribuida): {accuracy:.2f}%")

# Cálculo de TPR (True Positive Rate) e FPR (False Positive Rate)
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)

# Visualizar a decisão do Perceptron
plt.figure(figsize=(12, 6))

# Gráfico de decisão do Perceptron
plt.subplot(1, 2, 1)
plt.scatter(X_val[y_val == 1][:, 0], X_val[y_val == 1][:, 1], color='blue', label='Fundo (True)')
plt.scatter(X_val[y_val == -1][:, 0], X_val[y_val == -1][:, 1], color='red', label='Fundo (True)')

# Limites para o plano
x_min, x_max = X_val[:, 0].min() - 1, X_val[:, 0].max() + 1
y_min, y_max = X_val[:, 1].min() - 1, X_val[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
plt.title("Decisão do Perceptron - Base Distribuida")
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

# Gráfico de erro e acurácia ao longo das épocas
plt.figure(figsize=(6, 4))
epochs = np.arange(1, perceptron.epocas + 1)
plt.plot(epochs, perceptron.erros_por_epoca, color='red', label="Erro")
plt.plot(epochs, perceptron.acuracia_por_epoca, color='blue', label="Acurácia")
plt.title("Erro e Acurácia ao Longo das Épocas")
plt.xlabel("Época")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.show()
