import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dados = pd.read_csv('personagens.csv')

classe = dados.iloc[:,6].values
previsores = dados.iloc[:,[0,1,2,3,4,5]].values

lb = LabelEncoder()
classe = lb.fit_transform(classe)

classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 6))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(previsores,classe, epochs = 2000, batch_size = 100 )
classificador.evaluate(previsores,classe)