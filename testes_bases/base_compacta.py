import numpy as np
import matplotlib.pyplot as plt

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

# Função para criar base de dados em espiral
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

# Função para criar base de dados com pontos equidistantes
def gerar_equidistantes(n_pontos):
    data = np.random.randn(n_pontos, 2)
    labels = np.where(data[:, 0] + data[:, 1] > 0, 1, -1)
    return data, labels

# Bases de dados
bases = {
    "Mais Compacta (Não Linear)": {
        "data": np.random.multivariate_normal([0, 0], [[0.2, 0], [0, 0.2]], 500),
        "labels": np.hstack((np.ones(250), -np.ones(250)))
    },
    "Espiral": gerar_espiral(250),
    "Equidistantes": gerar_equidistantes(500)
}

# Treinar e avaliar o perceptron em cada base
for nome, base in bases.items():
    data, labels = base["data"], base["labels"]

    # Dividir o dataset em 40% treino e 60% validação
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

    # Avaliar no conjunto de validação
    y_pred = perceptron.predict(X_val)
    accuracy = np.mean((y_val == y_pred).astype(int)) * 100
    print(f"Acurácia para a base '{nome}': {accuracy:.2f}%")

    # Visualizar a decisão do Perceptron
    plt.figure(figsize=(8, 6))
    plt.scatter(X_val[y_val == 1][:, 0], X_val[y_val == 1][:, 1], color='blue', label='Classe 1 (True)')
    plt.scatter(X_val[y_val == -1][:, 0], X_val[y_val == -1][:, 1], color='red', label='Classe -1 (True)')
    
    x_min, x_max = X_val[:, 0].min() - 1, X_val[:, 0].max() + 1
    y_min, y_max = X_val[:, 1].min() - 1, X_val[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")
    plt.title(f"Decisão do Perceptron: {nome}")
    plt.xlabel("Dimensão 1")
    plt.ylabel("Dimensão 2")
    plt.legend()
    plt.grid(True)
    plt.show()
