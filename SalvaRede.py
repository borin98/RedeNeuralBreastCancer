import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

def salvaRede ( aNNJson, aNN ) :
    """
    Função que salva a estrutura da rede neural e seus
     pesos em disco

    :param aNNJson:
    :return:
    """

    with open("aNNBreast.json", "w") as jsonFile :
        jsonFile.write ( aNNJson )

    aNN.save_weights("aNNWeights.h5")

def main (  ):

    dadosEntrada = pd.read_csv("entradas-breast.csv")
    dadosSaida = pd.read_csv("saidas-breast.csv")

    aNN = Sequential()

    aNN.add(Dense(
        units=8,
        activation="relu",
        kernel_initializer="normal",
        input_dim=30))

    # camada de dropout de 20 %
    aNN.add(Dropout(0.2))

    aNN.add(Dense(
        units=8,
        activation="relu",
        kernel_initializer="normal"))

    aNN.add(Dropout(0.2))

    aNN.add(Dense(
        units=1,
        activation="sigmoid"))

    # compilando a rede neural
    aNN.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"])

    aNN.fit(dadosEntrada, dadosSaida, batch_size=10, epochs=100)

    # salvando a rede em formato json
    aNNJson = aNN.to_json()

    salvaRede(aNNJson, aNN)

if __name__ == '__main__':
    main()