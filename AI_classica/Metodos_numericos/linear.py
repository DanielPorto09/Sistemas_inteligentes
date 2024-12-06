import numpy as np

def y(x):
    """ Essa parte é responsavel por simular a função que não conhecemos, se quiser mude a função mas mantendo a linearidade"""
    return 2 * x + 4

def verifica_raiz(X):
    """Verifica se o valor X é uma raiz da função y(x)."""
    if y(X) == 0:
        return X
    return False

def verifica_convergência(ponto_1, ponto_2):
    return ponto_1[1] * ponto_2[1] < 0

def pede_pontos():
    """Solicita dois pontos ao usuário e verifica se atendem aos critérios para o método da bissecção."""
    x_1 = float(input("Digite um número X_1 para obter o primeiro ponto: "))
    ponto_inicial_1 = [x_1, y(x_1)]
    
    if verifica_raiz(x_1):
        return x_1

    while True:
        x_2 = float(input("Digite um número X_2 para obter o segundo ponto: "))
        if verifica_raiz(x_2):
            return x_2
        
        ponto_inicial_2 = [x_2, y(x_2)]

        if verifica_convergência(ponto_inicial_1, ponto_inicial_2):
            break
        print("\nEsses dois pontos não atenderam aos critérios:")
        print(f"O resultado de y_1 foi ({y(x_1)}, {x_1}) e y_2 foi ({y(x_2)}, {x_2})")
    
    return ponto_inicial_1, ponto_inicial_2

def bisec(Tol=1e-6, max_iter=500):
    pontos = pede_pontos()
    
    if isinstance(pontos, float):  # Raiz encontrada diretamente
        return pontos
    
    ponto_1, ponto_2 = pontos
    erro = 10
    i = 0
    
    while erro > Tol:
        assert i < max_iter, "Número máximo de iterações atingido."
        
        raiz = (ponto_1[0] + ponto_2[0]) / 2
        ponto_temp = [raiz, y(raiz)]
        i += 1
        
        if verifica_convergência(ponto_temp, ponto_1):
            erro = abs(ponto_temp[0] - ponto_1[0]) / abs(ponto_temp[0])
            ponto_2 = ponto_temp
        elif verifica_convergência(ponto_temp, ponto_2):
            erro = abs(ponto_temp[0] - ponto_2[0]) / abs(ponto_temp[0])
            ponto_1 = ponto_temp
    
    return raiz

raiz = bisec()
print("\nRaiz encontrada:", raiz)
