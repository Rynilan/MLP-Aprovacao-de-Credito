from random import random
from math import inf


def derivada(
    funcao: callable,
    valor: float,
    delta: float = 1e-6
):
    return (funcao(valor + delta) - funcao(valor - delta))/(2 * delta)


class Neuronio:
    __pesos: tuple[float, ...]  # valor para ajuste linear
    __bias: float  # valor para ajuste linear
    __ativacao: callable  # função de ativação
    # definida em float -> float

    # Valores da última entrada
    __saida: float  # pós ativação
    __somado: float  # aplicado os pesos e bias
    __entrada: float  # valor original

    __gradiente: float  # o gradiente descendente de cada peso

    def __init__(
        self,
        ativacao: callable,
        tamanho_da_camada_anterior: int,
    ):
        # Inicialização aleatória de pesos e bias
        self.__pesos = tuple(
            random() for _ in range(tamanho_da_camada_anterior)
        )
        self.__bias = random()

        self.__ativacao = ativacao

        self.__saida = None
        self.__somado = None
        self.__entrada = None

        self.__gradiente = 0

    def __avaliar_entrada(self, entrada: tuple[float, ...]):
        if (len(entrada) != len(self.__pesos)):
            raise ValueError(
                'Tamanho da entrada difere da quantia de pesos em {}'
                .format(len(entrada) - len(self.__pesos))
            )

    def aplicar(self, entrada: tuple[float, ...]):
        self.__avaliar_entrada(entrada)

        self.__entrada = entrada
        self.__somado = sum(  # Perceptron
            self.__pesos[i] * entrada[i] for i in range(len(entrada))
        ) + self.__bias
        self.__saida = self.__ativacao(self.__somado)

        return self.__saida

    def atualizar_pesos(
        self,
        taxa_de_aprendizado: float,
    ):
        self.__pesos = list(self.__pesos)
        for i in range(len(self.__pesos)):
            self.__pesos[i] -= float(
                taxa_de_aprendizado *
                self.__gradiente *
                self.__entrada[i]
            )
        self.__pesos = tuple(self.__pesos)

    def atualizar_bias(
        self,
        taxa_de_aprendizado: float,
    ):
        self.__bias -= taxa_de_aprendizado * self.__gradiente

    def gradiente_saida(self, derivada_da_perda: float):
        self.__gradiente = float(
            derivada(self.__ativacao, self.__somado) * derivada_da_perda
        )

    def gradiente_oculto(self, contribuicao: float):
        self.__gradiente = float(
            derivada(self.__ativacao, self.__somado) * contribuicao
        )

    def pegar_entrada(self):
        return self.__entrada

    def pegar_saida(self):
        return self.__saida

    def pegar_peso(self, peso: int):
        return self.__pesos[peso]

    def pegar_gradiente(self):
        return self.__gradiente


class Camada:
    __neuronios: tuple[Neuronio]
    __de_entrada: bool
    __tamanho_da_camada_anterior: int
    __buffer: tuple[float, ...] = None

    def __init__(
        self,
        quantos_neuronios: int,
        funcoes_de_ativacao: tuple[callable],
        tamanho_da_camada_anterior: int,
        de_entrada: bool = False
    ):
        self.__de_entrada = de_entrada
        if (de_entrada):
            self.__buffer = tuple()
            return

        self.__neuronios = tuple(
            Neuronio(funcoes_de_ativacao[i], tamanho_da_camada_anterior)
            for i in range(quantos_neuronios)
        )
        self.__tamanho_da_camada_anterior = tamanho_da_camada_anterior

    def aplicar(self, entrada: tuple[float, ...]):
        if (self.__de_entrada):
            self.__buffer = entrada
            return entrada

        saidas = list()
        for neuronio in self.__neuronios:
            saidas.append(
                neuronio.aplicar(entrada)
            )
        return tuple(saidas)

    def pegar_saida(self):
        if (not self.__de_entrada):
            return tuple(
                neuronio.pegar_saida() for neuronio in self.__neuronios
            )
        else:
            return self.__buffer

    def pegar_neuronios(self):
        return self.__neuronios

    def calcular_gradiente_de_saida(
        self,
        esperado: tuple[int, ...],
        perda: callable
    ):
        for i in range(len(self.__neuronios)):
            neuronio = self.__neuronios[i]
            obtido = self.pegar_saida()

            def perda_componente(valor):
                modificados = list(obtido)
                modificados[i] = valor

                return perda(
                    esperado,
                    tuple(modificados)
                )

            derivada_perda = derivada(
                perda_componente,
                obtido[i]
            )

            neuronio.gradiente_saida(derivada_perda)

    def calcular_gradiente_ocultos(self, proxima_camada: 'Camada'):
        proximos = proxima_camada.pegar_neuronios()

        for i, neuronio in enumerate(self.__neuronios):

            contribuicao = sum(
                proximo.pegar_peso(i) * proximo.pegar_gradiente()
                for proximo in proximos
            )

            neuronio.gradiente_oculto(contribuicao)

    def atualizar(self, taxa_de_aprendizado: float):
        for neuronio in self.__neuronios:
            neuronio.atualizar_pesos(taxa_de_aprendizado)
            neuronio.atualizar_bias(taxa_de_aprendizado)


class MLP:

    __camadas: tuple[Camada, ...]

    __erro: float  # valor prévio do erro
    __perda: callable  # função de perda
    # perda{esperado, obtido}
    # tuple[float, ...], tuple[float, ...] -> float

    __epocas: float  # quantas épocas se passaram desde a sua criação
    __taxa_de_aprendizado: float  # valor que define o passo de aprendizado
    # domínio entre 0 e 1

    __dados: tuple[tuple[float, ...], ...]  # dados de treinamento
    __saidas: tuple[int]  # Valor esperado para cada entrad

    def __init__(
        self,
        profundidade: int,
        definicao: tuple[tuple[int, tuple[callable, ...]], ...],
        perda: callable,
        taxa_de_aprendizado: float,
        dados: tuple[tuple[float, ...], ...],
        saidas: tuple[int]
    ):
        self.__camadas = tuple(
            Camada(
                definicao[i][0],
                definicao[i][1],
                definicao[i - 1][0] if (i != 0) else 0,
                i == 0
            )
            for i in range(profundidade)
        )

        self.__erro = inf
        self.__perda = perda

        self.__epocas = 0
        self.__taxa_de_aprendizado = taxa_de_aprendizado

        self.__dados = dados
        self.__saidas = saidas

    def aplicar(self, entrada: tuple[float, ...]):
        for camada in self.__camadas:
            entrada = camada.aplicar(entrada)  # feed-foward
        return entrada

    def aprender(self, esperado: tuple[int, ...], verbose=False):
        obtido = self.__camadas[-1].pegar_saida()

        self.__erro = self.__perda(esperado, obtido)

        self.__camadas[-1].calcular_gradiente_de_saida(esperado, self.__perda)

        for i in range(len(self.__camadas) - 2, 0, -1):
            self.__camadas[i].calcular_gradiente_ocultos(
                self.__camadas[i + 1]  # backpropagation
            )

        for camada in self.__camadas[1:]:
            camada.atualizar(self.__taxa_de_aprendizado)  # aprendizado

    def treinar(self, verbose=False, epocas='2*len(self.__dados)'):
        if (epocas == '2*len(self.__dados)'):
            epocas = 2 * len(self.__dados)
        for n in range(epocas):
            erro_t = 0
            for i in range(len(self.__dados)):
                self.aplicar(self.__dados[i])
                self.aprender(self.__saidas[i])
                erro_t += self.__erro
            erro_medio = erro_t / len(self.__dados)
            self.__epocas += 1
            if (verbose):
                print("Epoca: {} erro: {:.2f}".format(self.__epocas, erro_medio))

    def pegar_erro(self):
        return self.__erro

    def pegar_epocas(self):
        return self.__epocas


if __name__ == '__main__':
    print('Classes MLP, Camada e Neurônio criadas')
