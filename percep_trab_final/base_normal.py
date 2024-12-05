import numpy as np

def gerar_normal():
    normal_coords = [
        (-3, 1), (-2.5, 2), (-2, 3), (-1.5, 3.5), (-1, 3.7), (0, 3.5), (1, 3), (1.5, 2.5),
        (2, 1.8), (2.3, 1), (2, 0), (1.5, -1), (1, -1.8), (0.5, -2.5), (0, -3), (-0.5, -2.5),
        (-1, -1.8), (-1.5, -1), (-2, 0), (-2.5, 1)
    ]
    background_coords = np.random.uniform(-4, 4, size=(1000, 2))

    normal_coords = np.array(normal_coords)
    labels_normal = np.ones(len(normal_coords))  # Classe 1 para o normal
    labels_back = -np.ones(len(background_coords))  # Classe -1 para o fundo

    data = np.vstack((normal_coords, background_coords))
    labels = np.hstack((labels_normal, labels_back))

    return data, labels

datap, labelsp = gerar_normal()