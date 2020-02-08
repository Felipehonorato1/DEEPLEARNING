from keras.models import Sequential
from keras.layers import Dense,Dropout
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


dados = pd.read_csv('iris.csv')
previsores = dados.iloc[:,0:4].values
classe = dados.iloc[:,4].values

labEncoder = LabelEncoder()
classe = labEncoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'tanh', kernel_initializer = 'normal', input_dim = 4))
classificador.add(Dropout(0.1))
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['binary_accuracy'])

classificador.fit(x = previsores, y= classe_dummy, batch_size = 10, epochs=100)



classificador_json = classificador.to_json()
with open ('classificador_breast.json','w') as json_file:
    json_file.write(classificador_json) 
classificador.save_weights('classificador_breast.h5')

resultados = classificador.evaluate(previsores,classe_dummy)