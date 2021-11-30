# Algoritmo K means

""" Sirve esencialmente para clasificar valores buscando los puntos de datos “más similares” (por cercanía) aprendidos en la etapa de entrenamiento.
    Clasifica cada punto en una categoría, basándose en la categoría de sus vecinos más cercanos. 
    Suele utilizarse en sistemas de recomendación, búsqueda semántica y detección de anomalías. 

    https://www.aprendemachinelearning.com/clasificar-con-k-nearest-neighbor-ejemplo-en-python/#:~:text=K%2DNearest%2DNeighbor%20es%20un,el%20mundo%20del%20Aprendizaje%20Autom%C3%A1tico.
    
    Pros: Sencillo de aprender e implementar. Robusto y versátil.
    Contras: Utiliza todo el dataset para entrenar “cada punto” y por eso requiere de uso de mucha memoria y recursos CPU. 
    
    Cómo funciona:
        1. Calcula la distancia entre el item a clasificar y el resto de items del dataset de entrenamiento.
        2. Selecciona los “k” elementos más cercanos (con menor distancia, según la función que se use)
        3. Realiza una “votación de mayoría” entre los k puntos: los de una clase/etiqueta que <<dominen>> decidirán su clasificación final. """

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

dfP = pd.read_csv("pokemon.csv")

dfP.columns

# Podemos utilizar este algoritmo para ver cuáles son los Pokémon más fuertes, tratando la suma de estadísticas (base_total):
dfP2 = dfP[["base_total"]].dropna()
dfP2.describe()

# Vemos cuántos Pokémon tienen un sumatorio de 500 puntos de estadísticas, divididos en Ataque, At.Esp., Defensa, Def.Esp. y Velocidad
filtro = dfP2["base_total"] > 500

# Etiquetas
dfP2["base_total"][filtro] = "Fuerte"
dfP2["base_total"][filtro == False] = "Normal-Bajo"

# Sugiero utilizar un número impar para la comparación entre vecinos, de forma que podamos dilucidar posibles empates...
nbrs_3 = KNeighborsClassifier(n_neighbors = 3, n_jobs = -1)