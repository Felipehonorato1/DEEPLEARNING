from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

classificador = Sequential()
classificador.add(Conv2D(32,(3,3), activation = 'relu',input_shape=(64,64,3)))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Conv2D(32,(3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten())
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units =  1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                      metrics = ['binary_accuracy'])

img_gen = ImageDataGenerator(rotation_range = 7, horizontal_flip = True,
                             rescale = 1./255, zoom_range = 0.2,
                             height_shift_range = 0.07, shear_range = 0.2)

test_gen = ImageDataGenerator(rescale = 1./255)

base_treinamento = img_gen.flow_from_directory('dataset_personagens/training_set',
                                               target_size = (64,64), batch_size = 64, 
                                               class_mode = 'binary')

base_teste = test_gen.flow_from_directory('dataset_personagens/test_set', batch_size = 64, 
                                         target_size = (64,64), class_mode = 'binary')

classificador.fit_generator(base_treinamento,steps_per_epoch = 196, validation_steps = 73,
                            validation_data = base_teste, epochs = 10)

imagem_teste = image.load_img('dataset_personagens/test_set/bart/bart1.bmp',target_size = (64,64))
imagem_teste = image.img_to_array(imagem_teste)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste,axis = 0)


previsao = classificador.predict(imagem_teste)
