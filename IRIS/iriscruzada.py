import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from keras.utils import np_utils

dados = pd.read_csv('iris.csv')
previsores = dados.iloc[:,0:4].values
classe = dados.iloc[:,4].values

labEncoder = LabelEncoder()
classe = labEncoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 4, activation = 'relu',))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 3, activation = 'softmax', input_dim = 4))
    classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['categorical_accuracy'])
    return classificador
 
classificador = KerasClassifier(build_fn = criarRede, epochs = 1000, batch_size = 10)

resultados = cross_val_score(estimator=classificador, X= previsores, y = classe, cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()