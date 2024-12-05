import numpy as np

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
datae, labelse = gerar_espiral(250, ruido=0.3)

