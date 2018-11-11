import pandas as pd
import keras
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def criaRedeNeural (  ) :
    """
       Função que cria a rede neural
       de 1 camada oculta

       Com Dropout na primeira e
       seguda camada escondida

       :return:
       """

    aNN = Sequential()

    aNN.add(Dense(
        units = 32,
        activation = "relu",
        kernel_initializer = "random_uniform",
        input_dim = 30))

    # camada de dropout de 20 %
    aNN.add(Dropout(0.2))

    aNN.add(Dense(
        units = 32,
        activation = "relu",
        kernel_initializer = "random_uniform"))

    aNN.add(Dropout(0.2))

    aNN.add(Dense(
        units=1,
        activation="sigmoid"))

    # valores com precisão de 90,73212174940899 % de acerto
    #  lr = 0.004,
    # decay = 0.0003,
    # clipvalue = 0.7
    otimizador = keras.optimizers.Adam ( lr = 0.004, decay = 0.0003, clipvalue = 0.7 )

    # compilando a rede neural
    aNN.compile(
        optimizer = otimizador,
        loss = "binary_crossentropy",
        metrics=["accuracy"])

    return aNN

def main (  ) :

    dadosEntrada = pd.read_csv("entradas-breast.csv")
    dadosSaida = pd.read_csv("saidas-breast.csv")

    # melhores parâmetros
    # epochs = 600,
    # batch_size = 50
    aNN = KerasClassifier ( build_fn = criaRedeNeural,
                            epochs = 600,
                            batch_size = 50 )

    # melhores parâmetros
    # cv = 12
    resultado = cross_val_score ( estimator = aNN,
                                  X = dadosEntrada,
                                  y = dadosSaida,
                                  cv = 12,
                                  scoring = "accuracy")

    print("\nMédia de acerto calculada : {}".format(resultado.mean()))
    print("\nDesvio parão de erro : {}".format(resultado.std()))

if __name__ == '__main__':
    main()