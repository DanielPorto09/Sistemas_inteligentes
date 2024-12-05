import numpy as np

# Gerar o dataset Gaussiano com maior distância entre as classes
num_samples = 1000

mean_class1 = [3, 3]  # Centro da classe 1
mean_class2 = [-3, -3]  # Centro da classe 2
covariance = [[1, 0], [0, 1]]  # Covariância para ambas as classes

# Gerar as amostras
class1 = np.random.multivariate_normal(mean_class1, covariance, num_samples // 2)
class2 = np.random.multivariate_normal(mean_class2, covariance, num_samples // 2)

# Adicionar rótulos
labels_class1 = np.ones(len(class1))
labels_class2 = -np.ones(len(class2))

# Gerar outliers próximos à região crítica (em torno de [0, 0])
num_outliers = 25  # Número total de outliers (8 por classe)
mean_outliers = [0, 0]  # Região crítica entre as classes
cov_outliers = [[0.5, 0], [0, 0.5]]  # Outliers concentrados perto da origem

outliers_class1 = np.random.multivariate_normal(mean_outliers, cov_outliers, num_outliers // 2)
outliers_class2 = np.random.multivariate_normal(mean_outliers, cov_outliers, num_outliers // 2)

# Adicionar outliers aos dados
class1 = np.vstack((class1, outliers_class1))
class2 = np.vstack((class2, outliers_class2))

labels_class1 = np.ones(len(class1))
labels_class2 = -np.ones(len(class2))

# Criar o dataset final
data = np.vstack((class1, class2))
labels = np.hstack((labels_class1, labels_class2))
