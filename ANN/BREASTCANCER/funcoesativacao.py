import numpy as np 

#Transfer function

def stepFunction(soma):
    if(soma>=1):
        return 1
    return 0

teste = stepFunction(150)
print("\nSTEP FUNCTION: ")
print(teste)

#sigmoid function
# usada para calcular probabilidades e problemas binários
def sigmoidFunction(soma):
    return 1/(1+np.exp(-soma))

teste = sigmoidFunction(0.358)
print("\nSIGMOID: ")
print(teste)

#tangente hiperbolica 
# e^x - e ^-x / e^x + e^-x

def TanFunction(soma):
    return (np.exp(soma) - np.exp(-soma))/(np.exp(soma)+np.exp(-soma))

teste = TanFunction(-0.358)
print("\nTAN FUNCTION: ")
print(teste)

#RELU FUNCTION 
# Valores 0 ou maiores que 0. Sem valor máximo.
# Qualquer valor negativo se torna 0.
# Usadas em redes neurais convulocionais.
def ReluFunction(soma):
    if(soma>=0):
        return soma
    return 0;

print("\nRELU FUNCTION w/ negatives: ")
print(ReluFunction(-0.358))

print("RELU FUNCTION w/ positives: ")
print(ReluFunction(0.358))

#LINEAR FUNCTION 
# Não faz nada, retorna o mesmo valor inserido.
def LinearFunction(soma):
    return soma

print("\nLINEAR FUNCTION")
print(LinearFunction(-0.358))

#SOFTMAX FUNCTION 
# Utilizada para retornar probabilidades com problemas com mais de duas classes
# ex: separação de cores.
def SoftmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

print("\nSOFTMAX FUNCTION: ")
valores = [5.0,2.0,1.3]
print(SoftmaxFunction(valores))