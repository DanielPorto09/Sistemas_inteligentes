import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# Função degrau (HeavySide)
def degrau(x): 
    return 1 if x >= 0 else -1

# Classe Perceptron
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

# Carregar dataset
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets

# Transformar os rótulos em binário (-1, 1)
y = np.where(y == np.unique(y)[0], -1, 1)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instanciar e treinar o Perceptron
modelo = Perceptron(epocas=2000, taxa_aprendizado=0.1)
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Calcular a acurácia
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acuracia * 100:.2f}%")

# Curva ROC e AUC
y_scores = np.dot(X_test, modelo.pesos) + modelo.bias  # Combinação linear
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plotar curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Plotar gráfico da decisão
# Usando apenas as duas primeiras features para visualização em 2D
X_vis = X_test[:, :2]
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Obter previsões para a grade
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_test, edgecolors='k', marker='o', cmap=plt.cm.Paired)
plt.title('Região de Decisão do Perceptron')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.show()
