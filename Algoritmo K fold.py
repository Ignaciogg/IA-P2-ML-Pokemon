# Algoritmo K fold

""" Este algoritmo divide un conjunto de datos en un número K. Tomemos por ejemplo K+5. 
    En este escenario, el método dividirá el conjunto de datos en cinco pliegues. 
    Utiliza el primer pliegue en la primera iteración para probar el modelo. Utiliza los conjuntos de datos restantes para entrenar el modelo.
    
    https://www.aprendemachinelearning.com/sets-de-entrenamiento-test-validacion-cruzada/ """

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd