import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from keras.utils import np_utils
from sklearn.model_selection import GridSearchCV

dados = pd.read_csv('iris.csv')
previsores = dados.iloc[:,0:4].values
classe = dados.iloc[:,4].values

labEncoder = LabelEncoder()
classe = labEncoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criarRede(neurons,activation,Dropoutparam, kernel_initializer):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer,input_dim = 4))
    classificador.add(Dropout(Dropoutparam))
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    classificador.add(Dropout(Dropoutparam))
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy' ,metrics = ['categorical_accuracy'])
    return classificador
 
classificador = KerasClassifier(build_fn=criarRede)

parametros = {'neurons':[2,4,8],'activation':['relu','tanh'], 
              'Dropoutparam' : [0.1,0.2,0,3], 
              'kernel_initializer': ['random_uniform','normal'],'batch_size': [10,30], 'epochs': [50,100]}

grid_search = GridSearchCV(estimator = classificador,param_grid = parametros, 
                           scoring = 'accuracy',cv = 10)

grid_search = grid_search.fit(previsores,classe)

melhores_params = grid_search.best_params_
melhores_resultados = grid_search.best_score_ 
