import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from ann_visualizer.visualize import ann_viz

def main (  ) :

    dadosEntrada = pd.read_csv("entradas-breast.csv")
    dadosSaida = pd.read_csv("saidas-breast.csv")

    aNN = Sequential()

    aNN.add(Dense(
        units = 8,
        activation = "relu",
        kernel_initializer = "normal",
        input_dim=30))

    # camada de dropout de 20 %
    aNN.add(Dropout(0.2))

    aNN.add(Dense(
        units = 8,
        activation = "relu",
        kernel_initializer = "normal" ))

    aNN.add(Dropout(0.2))

    aNN.add(Dense(
        units=1,
        activation="sigmoid"))

    # compilando a rede neural
    aNN.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics=["binary_accuracy"])

    ann_viz(model=aNN, view=True, title="Rede Neural2")

    aNN.fit ( dadosEntrada, dadosSaida, batch_size = 10, epochs = 100 )

    previsaoEntrada = np.array (  [ np.random.uniform ( low = 0, high = 200,
                                                         size = 30) ]  )

    previsaoSaida = aNN.predict ( previsaoEntrada )
    resultado = ( previsaoSaida > 0.7 )
    print("\nPrecisão de saída : {}\n".format(previsaoSaida))
    print("Classificação : {}".format(resultado))



if __name__ == '__main__':
    main()