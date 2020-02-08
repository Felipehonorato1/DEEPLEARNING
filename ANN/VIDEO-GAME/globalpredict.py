import pandas as pd
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Input
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

dados = pd.read_csv('games.csv')
dados = dados.drop('Name',axis = 1)
dados = dados.drop('NA_Sales',axis = 1)
dados = dados.drop('EU_Sales',axis = 1)
dados = dados.drop('JP_Sales',axis = 1)
dados = dados.drop('Other_Sales',axis = 1)
dados = dados.drop('Developer',axis = 1)
dados = dados.dropna(axis = 0)
dados = dados.loc[dados['Global_Sales']> 1]
previsores = dados.iloc[:,[0,1,2,3,5,6,7,8,9]].values
valor = dados.iloc[:,4]

lb = LabelEncoder()
previsores[:,0] = lb.fit_transform(previsores[:,0])
previsores[:,2] = lb.fit_transform(previsores[:,2])
previsores[:,3] = lb.fit_transform(previsores[:,3])
previsores[:,8] = lb.fit_transform(previsores[:,8])

onehot = OneHotEncoder(categorical_features = [0,2,3,8])
previsores = onehot.fit_transform(previsores).toarray()

camada_entrada = Input(shape =(99,))
camada_oculta1 = Dense(units = 50, activation = 'sigmoid')(camada_entrada)
camada_saida = Dense(units = 1, activation = 'linear')(camada_entrada)

regressor = Model(inputs = camada_entrada, outputs = [camada_saida])
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(previsores,valor,batch_size = 100, epochs = 5000)

previsoes = regressor.predict(previsores)