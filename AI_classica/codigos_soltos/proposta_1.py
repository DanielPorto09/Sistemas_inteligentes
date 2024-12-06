import spade
import numpy as np
from scipy.optimize import fsolve
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
import random

# Credenciais dos agentes
GERADOR = "lucasgessner@magicbroccoli.de"
SENHA_GERADOR = "f-22raptor123"

RESOLVEDOR = "Kombatwar@magicbroccoli.de"
SENHA_RESOLVEDOR = "Kom12345"

# AGENTE GERADOR
class Gerador(Agent):
    async def setup(self):
        print("Iniciando Gerador de Funções...")

        # Sorteia o tipo de função
        self.tipo = random.randint(1, 3)
        print(f"Tipo de função escolhido: {self.tipo}º grau")

        # Gera coeficientes aleatórios
        self.a = random.randint(1, 3) * random.choice([-1, 1])
        self.b = random.randint(-3, 3)
        self.c = random.randint(-3, 3)
        self.d = random.randint(-3, 3)

        # Raízes da função para validação posterior
        self.raizes = self.gerar_raizes()

        # Exibe a função gerada
        if self.tipo == 1:
            print(f"Função gerada: f(x) = {self.a}(x - ({self.b}))")
        elif self.tipo == 2:
            print(f"Função gerada: f(x) = {self.a}(x - ({self.b}))(x - ({self.c}))")
        elif self.tipo == 3:
            print(f"Função gerada: f(x) = {self.a}(x - ({self.b}))(x - ({self.c}))(x - ({self.d}))")

        # Adiciona comportamentos
        self.add_behaviour(FuncaoHandler())
        self.add_behaviour(ValidarRaiz())
        self.add_behaviour(ResponderTipo())

    def gerar_raizes(self):
        """Gera as raízes exatas da função baseada nos coeficientes."""
        if self.tipo == 1:
            return [self.b]
        elif self.tipo == 2:
            return [self.b, self.c]
        elif self.tipo == 3:
            return [self.b, self.c, self.d]


class ResponderTipo(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=5)
        if msg and "Qual é o tipo da função?" in msg.body:
            print("Respondendo o tipo da função...")
            resposta = Message(to=str(msg.sender), metadata={"performative": "inform"})
            resposta.body = str(self.agent.tipo)
            await self.send(resposta)

class FuncaoHandler(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=5)
        if msg:
            try:
                # Tenta converter o valor recebido em float
                x = float(msg.body)
                if self.agent.tipo == 1:
                    resultado = self.agent.a * (x - self.agent.b)
                elif self.agent.tipo == 2:
                    resultado = self.agent.a * (x - self.agent.b) * (x - self.agent.c)
                elif self.agent.tipo == 3:
                    resultado = (
                        self.agent.a
                        * (x - self.agent.b)
                        * (x - self.agent.c)
                        * (x - self.agent.d)
                    )
                print(f"f({x}) = {resultado}")
                resposta = Message(to=str(msg.sender), metadata={"performative": "inform"})
                resposta.body = str(resultado)
                await self.send(resposta)
            except ValueError:
                # Mensagens não numéricas são ignoradas
                print(f"Mensagem não numérica recebida: {msg.body}")


class ValidarRaiz(CyclicBehaviour):
    async def run(self):
        msg = await self.receive(timeout=10)
        if msg:
            try:
                raiz_enviada = float(msg.body)
                print(f"Raiz recebida do resolvedor: {raiz_enviada}")

                # Verifica se a raiz está dentro da precisão esperada
                for raiz in self.agent.raizes:
                    if abs(raiz_enviada - raiz) <= 1e-2:
                        print("Raiz correta, parabéns resolvedor!")
                        await self.agent.stop()
                        return

                print("Raiz incorreta ou fora da precisão esperada.")
                await self.agent.stop()
            except ValueError:
                print(f"Mensagem não numérica recebida: {msg.body}")


# AGENTE RESOLVEDOR
class Resolvedor(Agent):
    async def setup(self):
        print("Iniciando Resolvedor...")
        self.add_behaviour(self.ResolveFuncao())

    class ResolveFuncao(OneShotBehaviour):
        async def run(self):
            print("Iniciando resolução...")
            tipo = await self.descobrir_tipo()
            print(f"Tipo de função descoberto: {tipo}")

            pontos = await self.obter_pontos(tipo)
            print(f"Pontos recebidos: {pontos}")

            coeficientes = self.interpolar_funcao(tipo, pontos)
            print(f"Coeficientes determinados: {coeficientes}")

            raiz = self.encontrar_raiz(coeficientes)
            print(f"Raiz encontrada: {raiz}")

            msg = Message(to=GERADOR, metadata={"performative": "finalize"})
            msg.body = str(raiz)
            await self.send(msg)
            await self.agent.stop()

        async def descobrir_tipo(self):
            msg = Message(to=GERADOR, metadata={"performative": "request"})
            msg.body = "Qual é o tipo da função?"
            await self.send(msg)
            resposta = await self.receive(timeout=10)
            return int(resposta.body) if resposta and resposta.body.isdigit() else None

        async def obter_pontos(self, tipo):
            num_pontos = {1: 2, 2: 3, 3: 4}[tipo]
            pontos = []
            for i in range(num_pontos):
                x = i - 1
                msg = Message(to=GERADOR, metadata={"performative": "subscribe"})
                msg.body = str(x)
                await self.send(msg)
                resposta = await self.receive(timeout=10)
                pontos.append((x, float(resposta.body)))
            return pontos

        def interpolar_funcao(self, tipo, pontos):
            grau = tipo
            X = np.array([[x**i for i in range(grau, -1, -1)] for x, _ in pontos])
            y = np.array([y for _, y in pontos])
            coeficientes = np.linalg.lstsq(X, y, rcond=None)[0]
            return coeficientes

        def encontrar_raiz(self, coeficientes):
            def f(x):
                return sum(c * x**i for i, c in enumerate(reversed(coeficientes)))
            raiz = fsolve(f, 0)[0]
            return raiz


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
