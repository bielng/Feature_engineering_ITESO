from sklearn.datasets import load_wine
import pandas as pd

LW = load_wine()
df_wine=pd.DataFrame(LW.data, columns=LW.features_names)
df_wine['class']=LW.target
df_wine.head()


###

df_wine.shape
df_wine['class'].dtype
df_wine.info()
df_wine.isnull().sum()
df_wine['class'].nunique()
df_wine['class'].unique()


###
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import optimize

minimize?


#Ayudin
'''Argumentos de 'minimize'
fun: funcion objectivo a minimizr fun(x, *args) -> float
x0: parametros a encontrar [a0, a1]
args: argumentos extras que necesita fun (x,y)
'''


#Encontramos la relacion entre las variables y='ash' y x='alcalinity_of_ash'

def ecm(a, x, y):
    return np.mean((a[0]+a[1]*x-y)**2)


#Para agregar una celda es con 'esc' + 'b' o 'esc' +'c'
a_inicial=np.random.rand(2) #valores aleatorios de dimension 2
solucion = minimize(ecm, a_inicial, args=(df_wine['alcalinity_of_ash'],df_wine['ash']))
solucion
