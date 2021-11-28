# Algoritmo K means

""" Sirve esencialmente para clasificar valores buscando los puntos de datos “más similares” (por cercanía) aprendidos en la etapa de entrenamiento.
    Clasifica cada punto en una categoría, basándose en la categoría de sus vecinos más cercanos. 
    Suele utilizarse en sistemas de recomendación, búsqueda semántica y detección de anomalías. 
    
    Pros: Sencillo de aprender e implementar
    Contras: Utiliza todo el dataset para entrenar “cada punto” y por eso requiere de uso de mucha memoria y recursos CPU. 
    
    Cómo funciona:
        1. Calcula la distancia entre el item a clasificar y el resto de items del dataset de entrenamiento.
        2. Selecciona los “k” elementos más cercanos (con menor distancia, según la función que se use)
        3. Realiza una “votación de mayoría” entre los k puntos: los de una clase/etiqueta que <<dominen>> decidirán su clasificación final. """

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

df = pd.read_csv("pokemon.csv")

df.columns
# En este algoritmo, hemos decidido ver 

