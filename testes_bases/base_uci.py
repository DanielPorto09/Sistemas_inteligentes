import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Função sigmoide
def degrau(x): 
    return 1 if x >= 0 else -1

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

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
                #rotulo_calculado = np.round(sigmoide(comb_linear)) 
                rotulo_calculado = np.array(degrau(comb_linear))
                
                erro = rotulos[i] - rotulo_calculado
                erro_total += abs(erro)
                
                correcao = self.taxa_aprendizado * erro
                self.pesos += correcao * dados_i
                self.bias += correcao
            
            self.erros_por_epoca.append(erro_total)
    
    def predict(self, dados):
        comb_linear = np.dot(dados, self.pesos) + self.bias
        #return np.array(sigmoide(comb_linear)) 
        return np.array([degrau(x) for x in comb_linear])
    
"------------------------------------------------------------------------------------------------"



# Carregar o arquivo CSV
df = pd.read_csv("/home/daniel-porto/estudo_perceptron/bin/meus_codigos/testes_bases/student_num.csv")

last_column = df.iloc[1:, -1]
labelsu = np.array(last_column.values)

matriz_menos_3_colum = df.iloc[1:, :-3]
datau = np.array(matriz_menos_3_colum.values)

#Dividir base de dados
data_40, data_60, labels_40, labels_60 = train_test_split(
    datau, labelsu, test_size=0.6, stratify=labelsu, random_state=42
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
if datau.shape[1] > 2:
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
