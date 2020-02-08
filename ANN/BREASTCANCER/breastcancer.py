import pandas as pd

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')
# BASICAMENTE ATRIBUIMOS VARIAVEIS A LEITURA DOS DADOS

from sklearn.model_selection import train_test_split
previsores_treinamento,previsores_teste,classe_treinamento,classe_teste = train_test_split(previsores,classe,test_size=0.25)
# SEPARAMOS 25% DOS DADOS PARA O TREINAMENTO DA MÁQUINA 

# PROBLEMA DE CLASSIFICACAO BINARIA - BENIGNO OU MALIGNO!

import keras 
from keras.models import Sequential
from keras.layers import Dense

# CAMADA DENSA SIGNIFICA QUE CADA NEURONIO É LIGADO A TODOS OS OUTROS DA CAMADA SUBSEQUENTE

classificador = Sequential()
#CAMADA OCULTA
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform',input_dim = 30))
classificador.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
#CAMADA DE SAIDA COM UM NEURONIO PELA CLASSIFICACAO BINARIA E SIGMOIDE PELA PROBABILIDADE ENTRE 0 E 1
classificador.add(Dense(units = 1, activation = 'sigmoid'))

#optmizer é relacionado a descida do gradiente, loss: calculo de erro. Metrics = metodos 



#classificador.compile(optimizer='adam',loss= 'binary_crossentropy', metrics = ['binary_accuracy'])

otimizador = keras.optimizers.Adam(lr = 0.001,decay = 0.0001,clipvalue = 0.5)

classificador.compile(optimizer=otimizador,loss= 'binary_crossentropy', metrics = ['binary_accuracy'])

#batch size = numero de registros analisados antes da reclassificacao de pesos, epochs = numero de ajustes de pesos
classificador.fit(previsores_treinamento,classe_treinamento,batch_size=10, epochs = 100) 


# Pegando os pesos p[os treinamento na primeira camada oculta
pesos0 = classificador.layers[0].get_weights()
print(pesos0)
# pegando os pesos da segunda camada oculta
pesos1 = classificador.layers[1].get_weights()
print(pesos1)


previsoes = classificador.predict(previsores_teste)

previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classe_teste,previsoes)
print (f"\nPrecisao: {precisao}")
matriz = confusion_matrix(classe_teste,previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)

print(f'\nResultado: {resultado}')