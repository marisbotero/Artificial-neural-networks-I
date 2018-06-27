
# coding: utf-8

# # Aprendamos ciencia de datos con python 🐍

# ## Conceptos básicos de Python para el análisis de datos

# Estructuras en python
# 
# 
# - Listas: Son una de las estructuras de datos más versátiles en Python. Una lista simplemente se puede definir escribiendo una lista de valores separados por comas entre corchetes.Estas pueden contener elementos de diferentes tipos, pero generalmente todos los artículos tienen el mismo tipo. Las listas de Python son mutables y se pueden cambiar los elementos individuales de una lista.
# 

# In[6]:


list=[0,1,4,9,16,25]


# In[7]:


list


# ## 💛 Crea tu propia lista de numeros primeros que se encuentren entre 0 y 10

# - Tuplas: se representa mediante una serie de valores separados por comas. Las tuplas son inmutables y el resultado está rodeado por paréntesis para que las tuplas anidadas se procesen correctamente.Como las tuplas son inmutables y no pueden cambiar, su procesamiento es más rápido que las listas. Por lo tanto, si es poco probable que su lista cambie, debe usar tuplas, en lugar de listas. 🤗

# In[8]:


tuple_example= 0,1,2,3,4


# In[9]:


tuple_example


# In[10]:


tuple_example[2]


# In[11]:


tuple_example[2]=7


# ## 💛 Crea tu propia tupla con los elementos que quieras
# 

# - Dicionarios: son un tipo de estructuras de datos que permite guardar un conjunto no ordenado de pares clave-valor, siendo las claves únicas dentro de un mismo diccionario (es decir que no pueden existir dos elementos con una misma clave).

# In[16]:


dictionary_example ={'Piloto 1':'Fernando Alonso', 'Piloto 2':'Juan Pablo Montoya', 'Piloto 3':'Felipe Massa'}


# In[17]:


dictionary_example


# In[18]:


dictionary_example.keys()


# In[19]:


dictionary_example['piloto 4']= 'Michael Schumacher'


# In[20]:


dictionary_example


# ## 💛 Crea tu propio diccionario con los elementos y claves que quieras
# 

# ## Iteración y construcciones condicionales

# - Al igual que la mayoría de los lenguajes, Python también tiene un bucle FOR que es el método más utilizado para la iteración.

# In[28]:


fact=1
N = 1
for i in range(1,N+1):
  fact *= i
print(fact)


# - Las declaraciones condicionales se usan para ejecutar fragmentos de código basados en una condición. La construcción más comúnmente utilizada es if-else.

# In[30]:


if N%2 == 0:
  print ('Even')
else:
  print ('Odd')


# Ahora que está familiarizado con los fundamentos de Python, demos un paso más. ¿Qué sucede si tiene que realizar las siguientes tareas?
# 
#  - Multiplicar 2 matrices
#  - Encuentra la raíz de una ecuación cuadrática
#  - Trazar gráficos de barras e histogramas
#  - Hacer modelos estadísticos
#  - Acceda a páginas web

# ## Librerias de python

# Hay muchas bibliotecas predefinidas que podemos importar directamente a nuestro código y hacer nuestra vida más fácil.🎉Vamos a dar un paso adelante en nuestro viaje para aprender Python conociendo algunas bibliotecas útiles. El primer paso es obviamente aprender a importarlos a nuestro entorno. Hay varias formas de hacerlo en Python:

# In[31]:


import math as m


# In[32]:


from math import *


# De la primera manera, hemos definido un alias m para la biblioteca math. Ahora podemos usar varias funciones de la biblioteca matemática (por ejemplo, factorial) haciendo referencia a ella utilizando el alias m.factorial ().
# 
# De la segunda manera, ha importado todo el espacio de nombres en matemáticas, es decir, puede usar directamente factorial () sin hacer referencia a las matemáticas.

# ## Resumen de bibliotecas, que necesitará para cualquier cálculo científico y análisis de datos:

# - NumPy: La característica más poderosa de NumPy es la matriz n-dimensional. Esta biblioteca también contiene funciones básicas de álgebra lineal, transformadas de Fourier, capacidades avanzadas de números aleatorios y herramientas para la integración con otros lenguajes de bajo nivel como Fortran, C y C ++.

# - SciPy está basada en NumPy. Es una de las bibliotecas más útiles para la variedad de módulos de ingeniería y ciencia de alto nivel, como la transformada discontinua de Fourier, el álgebra lineal, la optimización y las matrices dispersas.

# - Matplotlib: Es para trazar una gran variedad de gráficos, comenzando desde histogramas a gráficos de líneas Puede utilizar la función Pylab en el cuaderno ipython (cuaderno ipython -pylab = en línea) para utilizar estas funciones de trazado en línea. Si ignora la opción en línea, pylab convierte el entorno ipython a un entorno, muy similar a Matlab. También puede usar los comandos Latex para agregar matemática a su trazado.

# - Pandas: para operaciones de datos estructurados y manipulaciones. Se usa ampliamente para la manipulación y preparación de datos y han sido fundamentales para impulsar el uso de Python en la comunidad de científicos de datos.

# - Scikit Learn: Es para aprendizaje automático. Basada en NumPy, SciPy y matplotlib, esta biblioteca contiene una gran cantidad de herramientas eficaces para el aprendizaje automático y el modelado estadístico, que incluyen clasificación, regresión, clustering y reducción de dimensionalidad.

# ## Hagamos un Ejercicio práctico 
#    - Exploración de datos: descubriendo más sobre los datos que tenemos
#    - Data Munging: limpiar los datos y jugar con ellos para adaptarlos mejor al modelado estadístico
#    - Modelado Predictivo: ejecutar los algoritmos reales y divertirse 🙂

# 🐼Pandas es una de las bibliotecas de análisis de datos más útiles en Python (sé que estos nombres suenan raros, ¡pero espera!). Han sido fundamentales para aumentar el uso de Python en la comunidad de ciencia de datos. Ahora usaremos Pandas para leer un conjunto de datos de una competencia Analytics Vidhya, realizar un análisis exploratorio y construir nuestro primer algoritmo básico de categorización para resolver este problema.
# 
# Antes de cargar los datos, vamos a entender las 2 estructuras de datos clave en Pandas: Series y DataFrames

# ## Introducción a Series y Dataframes

# La serie se puede entender como una matriz unidimensional etiquetada / indexada. Puede acceder a elementos individuales de esta serie a través de estas etiquetas.
# 
# Un marco de datos es similar al libro de Excel: tiene nombres de columnas que hacen referencia a columnas y tiene filas, a las que se puede acceder mediante el uso de números de fila. La diferencia esencial es que los nombres de columna y los números de fila se conocen como índice de columna y fila, en el caso de marcos de datos.
# 
# Las series y los marcos de datos forman el modelo de datos básicos para Pandas en Python. Los conjuntos de datos se leen primero en estos cuadros de datos y luego se pueden aplicar fácilmente varias operaciones (por ejemplo, agrupar por, agregación, etc.) a sus columnas.

# - Conjunto de datos de práctica: problema de predicción de préstamos

# Importación de bibliotecas y el conjunto de datos:
# 
# Las siguientes son las bibliotecas que usaremos:
# 
# - numpy
# - matplotlib
# - pandas
# 

# Después de importar la biblioteca, hay que leer el conjunto de datos usando la función read_csv (). 

# In[42]:


import pandas as pd
import numpy as np
import matplotlib as plt
get_ipython().magic('matplotlib inline')


df = pd.read_csv("Downloads/datasets/train.csv") #Reading the dataset in a dataframe using Pandas


# In[43]:


df.head(10)


# Una vez que haya leído el conjunto de datos, puede ver algunas  de las filas superiores utilizando la función head ()
# Esto debería imprimir 10 filas. Ademas, puede ver el resumen de los campos numéricos utilizando la función describe ()

# In[44]:


df.describe()


# La función describe () proporcionaría: Conteo total, la media, la desviacón desviación estándar (std), cuartiles max y min

# ## Análisis de distribución

# Ahora que estamos familiarizados con las características básicas de los datos, estudiemos la distribución de diversas variables. Comencemos con las variables numéricas, a saber, ApplicantIncome y LoanAmount
# 
# Comencemos trazando el histograma de ApplicantIncome usando los siguientes comandos:

# In[45]:


df['Property_Area'].value_counts()


# In[46]:


df['ApplicantIncome'].hist(bins=50)





# En el histograma observamos que hay pocos valores extremos. Esta es también la razón por la cual se requieren 50 contenedores para representar claramente la distribución.
# 
# Ahora observamos los diagramas de caja para comprender las distribuciones.

# In[47]:


df.boxplot(column='ApplicantIncome')


# In[48]:


df.boxplot(column='ApplicantIncome', by = 'Education')


# In[49]:


df['LoanAmount'].hist(bins=50)


# ## 🧡Ahora carga tu propio datase y realiza un análisis de la distribución🧡

# Mira todo lo que hemos aprendido en tan poco tiempo, ahora es tu turno. La única manera de aprender es intentándolo por eso te invito a que cargues tu propio dataset de datos del titanic y que realices dos graficas que nos ayuden a analizar los datos. Lo harás increíble

# ## Análisis de variables categóricas

# 
# 

# El análisis que se está realizando en este ejemplo es para analizar el préstamo de una persona y las diferentes variables que influyen la aprobación del mismo. Para este ejemplo el préstamo ha sido codificado como 1 para Sí y 0 para No. Por lo tanto, la media representa la probabilidad de obtener un préstamo.
# 
# 
# 

# In[51]:


temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print ('Frequency Table for Credit History:') 
print (temp1)

print ('\nProbility of getting loan for each Credit History class:' )
print (temp2)


# ahora vamos a trazar un gráfico de barras usando la biblioteca "matplotlib" con el siguiente código:

# In[52]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")


# Esto muestra que las posibilidades de obtener un préstamo son ocho veces mayores si el solicitante tiene un historial crediticio válido. Puede trazar gráficos similares por Casado, Independiente, Propiedad_Area, etc.
# 
# Alternativamente, estos dos gráficos también se pueden visualizar combinándolos en un gráfico apilado:

# In[53]:


temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# Acabamos de crear dos algoritmos básicos de clasificación, uno basado en el historial de crédito, mientras que otro en 2 variables categóricas (incluido el género). 
# Realizamos un análisis exploratorio en Python usando Pandas. Espero que su amor por los pandas (el animal) se haya incrementado a esta altura, dada la cantidad de ayuda que la biblioteca puede brindarle al analizar los conjuntos de datos.
# 
# Ahora te toca a ti, realiza tu propio algoritmo de clasificación para el dataset de datos del titanic.
# 

# ## 💚Construyendo un modelo predictivo en Python💚

# In[ ]:




