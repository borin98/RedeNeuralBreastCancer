"""

Script que faz a classificação do dataset do câncer de mama

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from ann_visualizer.visualize import ann_viz

def criaRedeNeural ( optimizer, loss, kernelInitializer, activation, neurons ) :
    """
    Função que cria a rede neural
    de 1 camada oculta

    Com Dropout na primeira e
    seguda camada escondida

    :return:
    """

    aNN = Sequential()

    aNN.add(Dense(
        units = neurons,
        activation = activation,
        kernel_initializer = kernelInitializer,
        input_dim = 30 ) )

    # camada de dropout de 20 %
    aNN.add( Dropout ( 0.2 ) )

    aNN.add ( Dense(
        units = neurons,
        activation = activation,
        kernel_initializer = kernelInitializer) )

    aNN.add ( Dropout ( 0.2 ) )

    aNN.add(Dense(
        units = 1,
        activation = "sigmoid"))

    # compilando a rede neural
    aNN.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = ["binary_accuracy"])

    #ann_viz(model=aNN, view=True, title="Rede Neural")

    return aNN

def main() :

    # criando os dados
    dadosEntrada = pd.read_csv("entradas-breast.csv")
    dadosSaida = pd.read_csv("saidas-breast.csv")

    # setando os previsões de saída da rede
    previsaoEntradaTreinamento, previsaoEntradaTeste, previsaoSaidaTreinamento, previsaoSaidaTeste = train_test_split(
        dadosEntrada,
        dadosSaida,
        test_size = 0.25
    )

    # criando a rede neural Sequecial ( rede neural artificial -rna- )
    aNN = KerasClassifier ( build_fn = criaRedeNeural )

    parametros = {
        "batch_size" : [10, 30],
        "epochs" : [100, 500],
        "optimizer" : ["adam", "sgd"],
        "loss" : ["binary_crossentropy", "hinge"],
        "kernelInitializer" : ["random_uniform", "normal"],
        "activation" : ["relu", "tanh"],
        "neurons" : [16, 8]
    }

    dropoutRate = [0.0,0.2,0.2]
    paramGrid = dict ( dropoutRate = dropoutRate )

    gridSearch = GridSearchCV ( estimator = aNN,
                                param_grid = parametros,
                                scoring = "accuracy",
                                cv = 10)

    gridSearch = gridSearch.fit(dadosEntrada, dadosSaida)

    resultadoParam = gridSearch.best_params_
    resultadoPresc = gridSearch.best_score_

if __name__ == '__main__':
    main()

