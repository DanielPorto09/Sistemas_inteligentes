import numpy as np
from scipy.optimize import fsolve

# Simula a função desconhecida f(x)
class FuncaoDesconhecida:
    def __init__(self, tipo, coeficientes):
        """
        Inicializa a função desconhecida.
        :param tipo: 1, 2 ou 3 (grau da função).
        :param coeficientes: Lista de coeficientes [A, B, C, D].
        """
        self.tipo = tipo
        self.coeficientes = coeficientes

    def avaliar(self, x):
        """
        Avalia a função em x.
        """
        if self.tipo == 1:
            return self.coeficientes[0] * x + self.coeficientes[1]
        elif self.tipo == 2:
            return (
                self.coeficientes[0] * x**2
                + self.coeficientes[1] * x
                + self.coeficientes[2]
            )
        elif self.tipo == 3:
            return (
                self.coeficientes[0] * x**3
                + self.coeficientes[1] * x**2
                + self.coeficientes[2] * x
                + self.coeficientes[3]
            )
        else:
            raise ValueError("Tipo inválido de função.")


# Determina a função com base nos pontos e no tipo
def determina_funcao():
    """
    Determina a função ajustada com base nos pontos fornecidos.
    :return: Coeficientes da função ajustada e pontos utilizados.
    """
    # Pontos conhecidos (x, y)
    pontos = [
        (-2, funcao.avaliar(-2)),
        (0, funcao.avaliar(0)),
        (2, funcao.avaliar(2)),
        (3, funcao.avaliar(3)),
    ]

    num_pontos = len(pontos)
    if num_pontos < 2:
        raise ValueError("Pontos insuficientes para determinar uma função.")
    
    if num_pontos >= 2:
        tipo = 1  # 1º grau
    if num_pontos >= 3:
        tipo = 2  # 2º grau
    if num_pontos >= 4:
        tipo = 3  # 3º grau

    n = tipo + 1  # Número de coeficientes necessários
    pontos_usados = pontos[: n + 1]
    
    # Monta o sistema linear
    X = np.array([[x**i for i in range(n - 1, -1, -1)] for x, _ in pontos_usados])
    y = np.array([y for _, y in pontos_usados])
    coeficientes = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeficientes, pontos_usados


# Determina a raiz da função ajustada
def determina_raiz():
    """
    Determina uma raiz da função ajustada.
    :return: Raiz de menor valor.
    """
    coeficientes, _ = determina_funcao()

    # Define a função baseada nos coeficientes
    def f(x):
        return sum(c * x**i for i, c in enumerate(reversed(coeficientes)))

    # Usando o método `fsolve` para encontrar raízes
    guess = 0  # Palpite inicial
    raiz = fsolve(f, guess)[0]
    return raiz


# Simulação
# Função desconhecida: 2º grau, 3x² + 2x - 5
funcao = FuncaoDesconhecida(3, [3, 2, -5,1])

# Determina a raiz
raiz = determina_raiz()
print(f"Raiz de menor valor: {raiz}")
