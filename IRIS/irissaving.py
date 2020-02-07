from keras.models import Sequential
from keras.layers import Dense,Dropout


classificador = Sequential()
classificador.add(Dense())
classificador.add(Dropout())
classificador.compile(optimizer = '',loss = '', metrics = [''])

classificador.compile(optimizer = 'adam' loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

classificador.fit(x = previsores, y= classe, batch_size = , epochs= )


classificador_json = classificador.to_json()
with open('irisneuralnetwork','w'):
    json_file.write(classificador_json)
classificador.save_weights('irisweights.h5')

