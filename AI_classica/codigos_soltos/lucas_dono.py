import spade
import numpy as np
from scipy.optimize import fsolve
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.template import Template
from spade.message import Message
import random

# Credenciais dos agentes
GERADOR = "lucasgessner@magicbroccoli.de"
SENHA_GERADOR = "f-22raptor123"

RESOLVEDOR = "Kombatwar@magicbroccoli.de"
SENHA_RESOLVEDOR = "Kom12345"

# AGENTE GERADOR
class Gerador(Agent):
    def __init__(self, jid, password):
        super().__init__(jid, password)
        self.tipo = None
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0

    async def setup(self):
        """
        Configuração inicial do agente.
        """
        print("Iniciando Gerador de Funções...")

        # Sorteia o tipo de função
        self.tipo = random.randint(1, 3)
        print(f"Tipo de função escolhido: {self.tipo}º grau")

        # Gera coeficientes aleatórios
        while self.a == 0:
            self.a = random.randint(-3, 3)
        self.b = random.randint(-3, 3)
        self.c = random.randint(-3, 3)
        self.d = random.randint(-3, 3)

        # Define o comportamento com base no tipo
        if self.tipo == 1:
            print(f"Função gerada: f(x) = {self.a}(x - ({self.b}))")
            funcao = Funcao1Grau(self)
        elif self.tipo == 2:
            print(f"Função gerada: f(x) = {self.a}(x - ({self.b}))(x - ({self.c}))")
            funcao = Funcao2Grau(self)
        elif self.tipo == 3:
            print(f"Função gerada: f(x) = {self.a}(x - ({self.b}))(x - ({self.c}))(x - ({self.d}))")
            funcao = Funcao3Grau(self)

        # Adiciona o comportamento da função
        t = Template(metadata={"performative": "subscribe"})
        self.add_behaviour(funcao, t)

        # Adiciona o comportamento para responder ao tipo da função
        tipo_funcao = TipoFuncao(self)
        t_tipo = Template(metadata={"performative": "request"})
        self.add_behaviour(tipo_funcao, t_tipo)
        
        # Adiciona comportamento para finalizar e validar a raiz
        finaliza_behaviour = FinalizaGerador(self)
        t_finaliza = Template(metadata={"performative": "inform"})
        self.add_behaviour(finaliza_behaviour, t_finaliza)


class FinalizaGerador(CyclicBehaviour):
    def __init__(self, agent):
        self.agent = agent
        super().__init__()

    async def run(self):
        msg = await self.receive(timeout=10)
        if msg:
            print(f"Recebi a mensagem: {msg.body}")

            # Verifica se a mensagem contém uma raiz
            if "Raiz encontrada:" in msg.body:
                raiz = float(msg.body.split(":")[1].strip())
                resultado = self.agent.a * (raiz - self.agent.b)

                # Para funções de grau maior, é necessário validar mais raízes
                if self.agent.tipo >= 2:
                    resultado *= (raiz - self.agent.c)
                if self.agent.tipo == 3:
                    resultado *= (raiz - self.agent.d)

                if abs(resultado) < 1e-5:  # Tolerância para validar a raiz
                    print(f"A raiz {raiz} é válida!")
                else:
                    print(f"A raiz {raiz} não é válida. Resultado: {resultado}")

            # Finaliza o agente Gerador
            print("Descansar que o gerador não é de ferro...")
            await self.agent.stop()


# Comportamentos de funções
class Funcao1Grau(CyclicBehaviour):
    def __init__(self, agent):
        self.agent = agent
        super().__init__()

    async def run(self):
        resposta = await self.receive(timeout=5)
        if resposta:
            x = float(resposta.body)
            resultado = self.agent.a * (x - self.agent.b)
            print(f"f({x}) = {resultado}")
            msg = Message(to=str(resposta.sender), metadata={"performative": "inform"})
            msg.body = str(resultado)
            await self.send(msg)


class Funcao2Grau(CyclicBehaviour):
    def __init__(self, agent):
        self.agent = agent
        super().__init__()

    async def run(self):
        resposta = await self.receive(timeout=5)
        if resposta:
            x = float(resposta.body)
            resultado = self.agent.a * (x - self.agent.b) * (x - self.agent.c)
            print(f"f({x}) = {resultado}")
            msg = Message(to=str(resposta.sender), metadata={"performative": "inform"})
            msg.body = str(resultado)
            await self.send(msg)


class Funcao3Grau(CyclicBehaviour):
    def __init__(self, agent):
        self.agent = agent
        super().__init__()

    async def run(self):
        resposta = await self.receive(timeout=5)
        if resposta:
            x = float(resposta.body)
            resultado = (
                self.agent.a * (x - self.agent.b) * (x - self.agent.c) * (x - self.agent.d)
            )
            print(f"f({x}) = {resultado}")
            msg = Message(to=str(resposta.sender), metadata={"performative": "inform"})
            msg.body = str(resultado)
            await self.send(msg)


class TipoFuncao(CyclicBehaviour):
    def __init__(self, agent):
        self.agent = agent
        super().__init__()

    async def run(self):
        msg = await self.receive(timeout=5)
        if msg:
            tipo = {1: "1grau", 2: "2grau", 3: "3grau"}[self.agent.tipo]
            print(f"Respondendo tipo da função: {tipo}")
            resposta = Message(to=str(msg.sender), metadata={"performative": "inform"})
            resposta.body = tipo
            await self.send(resposta)


# AGENTE RESOLVEDOR
class Resolvedor(Agent):
    async def setup(self):
        print("Iniciando Resolvedor...")
        resolver_behaviour = self.ResolveFuncao()
        self.add_behaviour(resolver_behaviour)

    class ResolveFuncao(OneShotBehaviour):
        async def run(self):
            print("Iniciando resolução...")

            # Descobre o tipo da função
            tipo = await self.descobrir_tipo()
            print(f"Tipo de função descoberto: {tipo}")

            # Obtém os pontos necessários
            pontos = await self.obter_pontos(tipo)
            print(f"Pontos recebidos: {pontos}")

            # Determina os coeficientes da função
            coeficientes = self.interpolar_funcao(tipo, pontos)
            print(f"Coeficientes determinados: {coeficientes}")

            # Encontra a raiz
            raiz = self.encontrar_raiz(coeficientes)
            print(f"Raiz encontrada: {raiz}")

            # Envia a raiz para o Gerador
            await self.enviar_raiz(raiz)

        async def descobrir_tipo(self):
            msg = Message(to=GERADOR, metadata={"performative": "request"})
            msg.body = "Qual é o tipo da função?"
            await self.send(msg)
            resposta = await self.receive(timeout=10)
            if resposta:
                return resposta.body
            else:
                raise TimeoutError("Não foi possível obter o tipo da função.")

        async def obter_pontos(self, tipo):
            num_pontos = {"1grau": 2, "2grau": 3, "3grau": 4}[tipo]
            pontos = []
            for i in range(num_pontos):
                x = i - 1
                msg = Message(to=GERADOR, metadata={"performative": "subscribe"})
                msg.body = str(x)
                await self.send(msg)
                resposta = await self.receive(timeout=10)
                if resposta:
                    pontos.append((x, float(resposta.body)))
                else:
                    raise TimeoutError(f"Não foi possível obter f({x}).")
            return pontos

        def interpolar_funcao(self, tipo, pontos):
            grau = {"1grau": 1, "2grau": 2, "3grau": 3}[tipo]
            X = np.array([[x**i for i in range(grau, -1, -1)] for x, _ in pontos])
            y = np.array([y for _, y in pontos])
            coeficientes = np.linalg.lstsq(X, y, rcond=None)[0]
            return coeficientes

        def encontrar_raiz(self, coeficientes):
            def f(x):
                return sum(c * x**i for i, c in enumerate(reversed(coeficientes)))
            raiz = fsolve(f, 0)[0]
            return raiz

        async def enviar_raiz(self, raiz):
            msg = Message(to=GERADOR, metadata={"performative": "inform"})
            msg.body = f"Raiz encontrada: {raiz}"
            await self.send(msg)
            print("Raiz enviada ao Gerador.")


# Inicialização dos agentes
async def main():
    gerador = Gerador(GERADOR, SENHA_GERADOR)
    await gerador.start()
    resolvedor = Resolvedor(RESOLVEDOR, SENHA_RESOLVEDOR)
    await resolvedor.start()
    await spade.wait_until_finished(gerador)
    await spade.wait_until_finished(resolvedor)


if __name__ == "__main__":
    spade.run(main())
