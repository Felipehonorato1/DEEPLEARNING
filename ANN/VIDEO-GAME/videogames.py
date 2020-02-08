import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model


dados = pd.read_csv('games.csv')

nome_jogos = dados.Name
dados = dados.drop('Name',axis = 1)
dados = dados.drop('Other_Sales',axis = 1)
dados = dados.drop('Global_Sales',axis = 1)
dados = dados.drop('Developer', axis = 1)

dados = dados.dropna(axis = 0)

dados = dados.loc[dados['NA_Sales']>1]
dados = dados.loc[dados['EU_Sales']>1]

previsores = dados.iloc[:,[0,1,2,3,7,8,9,10,11]].values 
eu_sales = dados.iloc[:,5]
jp_sales = dados.iloc[:,6]
na_sales = dados.iloc[:,4]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

lb = LabelEncoder()
previsores[:,0] = lb.fit_transform(previsores[:,0])
previsores[:,2] = lb.fit_transform(previsores[:,2])
previsores[:,3] = lb.fit_transform(previsores[:,3])
previsores[:,8] = lb.fit_transform(previsores[:,8])

onehotencoder = OneHotEncoder(categorical_features = [0,2,3,8])
previsores = onehotencoder.fit_transform(previsores).toarray()

camada_entrada = Input(shape=(61,))
camada_oculta1 = Dense(units = 32, activation = 'sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 32, activation = 'sigmoid')(camada_oculta1)
camada_saida1 = Dense(units = 1, activation = 'linear')(camada_oculta2)
camada_saida2 = Dense(units = 1, activation = 'linear')(camada_oculta2)
camada_saida3 = Dense(units = 1, activation = 'linear')(camada_oculta2)

regressor = Model(inputs = camada_entrada, outputs = [camada_saida1,camada_saida2,camada_saida3])
regressor.compile(optimizer = 'adam', loss = 'mse')

regressor.fit(previsores,[eu_sales,jp_sales,na_sales],epochs = 5000, batch_size = 100)

previsao_jp,previsao_na,previsao_eu = regressor.predict(previsores)