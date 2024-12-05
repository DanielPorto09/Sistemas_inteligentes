import numpy as np
import matplotlib.pyplot as plt

# Função de ativação Heaviside
def heaviside(x):
    return 1 if x >= 0 else 0

# Classe Perceptron
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = heaviside(linear_output)

                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([heaviside(x) for x in linear_output])

# Gerar o dataset Gaussiano
num_samples = 200
mean_class1 = [2, 2]
mean_class2 = [-2, -2]
covariance = [[1, 0], [0, 1]]

class1 = np.random.multivariate_normal(mean_class1, covariance, num_samples // 2)
class2 = np.random.multivariate_normal(mean_class2, covariance, num_samples // 2)

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
perceptron = Perceptron(learning_rate=0.1, epochs=1000)
perceptron.fit(X_train, y_train)

# Avaliar o Perceptron no conjunto de validação
y_pred = perceptron.predict(X_val)

# Converter os rótulos para {0, 1} para calcular acurácia
y_val_binary = (y_val == 1).astype(int)
y_pred_binary = (y_pred == 1).astype(int)

accuracy = np.mean(y_val_binary == y_pred_binary) * 100

print(f"Acurácia no conjunto de validação: {accuracy:.2f}%")

# Visualização do conjunto de validação
plt.figure(figsize=(8, 6))
plt.scatter(X_val[y_val == 1][:, 0], X_val[y_val == 1][:, 1], color='blue', label='Classe 1 (True)')
plt.scatter(X_val[y_val == -1][:, 0], X_val[y_val == -1][:, 1], color='red', label='Classe 2 (True)')

# Decisão do perceptron
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
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
