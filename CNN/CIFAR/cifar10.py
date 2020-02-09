from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization,Dropout, Flatten
from keras.utils import np_utils

(X_treinamento,y_treinamento),(X_teste,y_teste) = cifar10.load_data()
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],32,32,3)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = X_teste.reshape(X_teste.shape[0],32,32,3)
previsores_teste = previsores_teste.astype('float32')
previsores_treinamento /= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento,10)
classe_teste = np_utils.to_categorical(y_teste,10)

classificador = Sequential()

classificador.add(Conv2D(32,(4,4),activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (3,3)))

classificador.add(Conv2D(32,(4,4),activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (3,3)))
classificador.add(Flatten())

classificador.add(Dense(units = 256, activation = 'relu'))
classificador.add(Dropout(0.1))
classificador.add(Dense(units = 256, activation = 'relu'))
classificador.add(Dropout(0.1))
classificador.add(Dense(units = 10, activation = 'softmax'))
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])

classificador.fit(previsores_treinamento,classe_treinamento, batch_size= 100, epochs = 10,
                  validation_data = (previsores_teste,classe_teste))

resultados = classificador.evaluate(previsores_teste,classe_teste)


