import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization 
import numpy as np
from keras.preprocessing import image

(X_treinamento,y_treinamento),(X_teste, y_teste) = mnist.load_data()

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],28,28,1)
previsores_teste = X_teste.reshape(X_teste.shape[0],28,28,1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento/= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento,10)
classe_teste = np_utils.to_categorical(y_teste,10)

classificador = Sequential()
classificador.add(Conv2D(32, (3,3), input_shape=(28,28,1),activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Conv2D(32, (3,3),activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size= (2,2)))
classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128,activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, activation = 'softmax'))
classificador.compile(optimizer = 'adam' , loss = 'categorical_crossentropy',metrics = ['accuracy'])
classificador.fit(previsores_treinamento,classe_treinamento, batch_size= 128, epochs = 5,
                  validation_data = (previsores_teste,classe_teste))

imagem_teste = X_teste.reshape(1,28,28,1)
imagem_teste = imagem_teste.astype('float32')

imagem_teste /= 255

preview = classificador.predit(imagem_teste)

result = np.argmax(preview)

