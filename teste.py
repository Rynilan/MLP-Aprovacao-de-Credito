import pandas as pd
from rede_neural import MLP
from math import exp, log


# Leitura dos dados de entrada
colunas: tuple[str] = (
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
    'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16'
)
data = pd.read_csv(
    'data/crx.data', names=colunas, header=None, na_values='?'
)

pre_x = data[[
    'A3',   # Dívida atual
    'A8',   # Outros investimentos
    'A11',  # Compormissos de crédito (quantia)
    'A14'   # Despezas mensais
]].astype(float)
normalizado = (pre_x - pre_x.mean()) / pre_x.std()
# Dados de entrada
x = tuple(
    normalizado.itertuples(index=False, name=None)
)

# Dados de saídas
# Normalização da entrada 1 ou 0
y = tuple(
    (valor,)
    for valor in (data['A16'] == '+').astype(int)
)


def bce(Y, y):
    epsilon = 1e-15

    return -sum(
        Y[i] * log(max(y[i], epsilon))
        + (1 - Y[i]) * log(max(1 - y[i], epsilon))
        for i in range(len(Y))
    ) / len(Y)


rede: MLP = MLP(
    4,  # Profundidae (quantia de camadas)
    (  # Definições de cada camada da rede
        (4, None),  # Camada de entrada, 4 parâmetros de entrada
        (8, tuple(lambda x: max(0, x) for _ in range(8))),  # Camada oculta 1
        (4, tuple(lambda x: max(0, x) for _ in range(4))),  # Camada oculta 2
        (1, (lambda x: 1 / (1 + exp(-x)), ))  # Camada de saída
    ),  # Funções de atiação: ReLu e Sigmoide
    # Função perda BCE
    bce,
    0.0001,  # learning rate
    x,  # dados de treinamento (entradas)
    y  # dados de treinamento (saídas)
)

rede.treinar(
    verbose=True,
    epocas=1994
)
