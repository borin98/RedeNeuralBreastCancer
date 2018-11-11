import numpy as np
import pandas as pd
from keras.models import model_from_json

def main (  ) :

    dadosEntrada = pd.read_csv("entradas-breast.csv")
    dadosSaida = pd.read_csv("saidas-breast.csv")

    arquivo = open("aNNBreast.json", "r")
    estruturaAnn = arquivo.read()
    arquivo.close()

    # carregando a estrutura da rede neural
    aNN = model_from_json  ( estruturaAnn )

    # carregando os pesos da rede neural
    aNN.load_weights("aNNWeights.h5")

    previsaoEntrada = np.array([np.random.uniform(low=0, high=200,
                                                  size=30)])

    aNN.compile(loss = "binary_crossentropy",
               optimizer = "adam",
               metrics = ["binary_accuracy"])

    resultado = aNN.evaluate ( dadosEntrada, dadosSaida )

    previsaoSaida = aNN.predict(previsaoEntrada)
    resultado = (previsaoSaida > 0.7)
    print("\nPrecisão de saída : {}\n".format(previsaoSaida))
    print("Classificação : {}".format(resultado))

if __name__ == '__main__':
    main()