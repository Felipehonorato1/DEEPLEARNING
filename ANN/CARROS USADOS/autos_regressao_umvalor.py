import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV

dados = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

dados = dados.drop('dateCrawled', axis = 1)
dados = dados.drop('dateCreated', axis = 1)
dados = dados.drop('postalCode', axis = 1)
dados = dados.drop('lastSeen', axis = 1)
dados = dados.drop('name',axis = 1)
dados = dados.drop('seller',axis = 1)
dados = dados.drop('offerType',axis = 1)
dados = dados.drop('nrOfPictures',axis = 1)

############################################## TRATAMENTO DE DADOS FALTOSOS / INCONSISTENTES DELETANDO OU COLOCANDO OS VALORES MAIS COMUNS NA BASE
i1 = dados.loc[dados.price <= 10]
dados = dados[dados.price>10] 
i2 = dados.loc[dados.price >= 400000]
dados = dados[dados.price<400000]
dados.loc[pd.isnull(dados['vehicleType'])]
dados['vehicleType'].value_counts() # LIMOSINE
dados.loc[pd.isnull(dados['gearbox'])]
dados['gearbox'].value_counts() # MANUELL 
dados.loc[pd.isnull(dados['model'])]
dados['model'].value_counts() # GOLF 
dados.loc[pd.isnull(dados['fuelType'])]
dados['fuelType'].value_counts() # benzin
dados.loc[pd.isnull(dados['notRepairedDamage'])]
dados['notRepairedDamage'].value_counts() # NEIN 

valores = {'vehicleType': 'limousine', 'gearbox': 'manuell',
           'model': 'golf', 'fuelType': 'benzin', 'notRepairedDamage': 'nein'}

dados = dados.fillna(value = valores)

previsores = dados.iloc[:,1:13].values
preco_real = dados.iloc[:,0].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lEncoder = LabelEncoder()
previsores[:,0] = lEncoder.fit_transform(previsores[:,0])
previsores[:,1] = lEncoder.fit_transform(previsores[:,1])
previsores[:,3] = lEncoder.fit_transform(previsores[:,3])
previsores[:,5] = lEncoder.fit_transform(previsores[:,5])
previsores[:,8] = lEncoder.fit_transform(previsores[:,8])
previsores[:,9] = lEncoder.fit_transform(previsores[:,9])
previsores[:,10] = lEncoder.fit_transform(previsores[:,10])

onehotencoder = OneHotEncoder(categorical_features = [0,1,3,5,9,10])
previsores = onehotencoder.fit_transform(previsores).toarray()

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

def criarRede():
    regressor = Sequential()
    regressor.add(Dense(units = 158, activation = 'relu', input_dim = 310))
    regressor.add(Dense(units = 158, activation = 'relu'))
    regressor.add(Dense(units = 1, activation = 'linear'))
    regressor.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics = ['mean_absolute_error'])
    return regressor

regressor = KerasRegressor(build_fn=criarRede,epochs = 100, batch_size = 300)

resultados_cruzado = cross_val_score(estimator = regressor, X = previsores, y = preco_real, cv = 10, scroing = 'mean_absolute_error')
media = resultados.mean()
desvio = resultados.std()

