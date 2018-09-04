# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:37:03 2018

@author: alu2015111446

Algoritmo genético para ajustar hiperparâmetros de MLP

Número de camadas ocultas. Mínimo 1 e máx 3 
Representação binária. Tamanho máximo 2, pois é o valor máximo (11)

Número de neurônios. Mínimo 1 e máx 1000
Representação binária. Tamanho máximo 10, pois é o valor máximo (1111101000)

1. Geração aleatória de indivíduos
2. Avaliados por função de aptidão
3. Cruzamento de informações (crossover)
4. Mutação
5. Seleção

"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def generate_population(n):
    """
    Gera a população inicial
        
        Parâmetros
        
        n = Número de indivíduos
    """
    population = []
    
    for k in range(n):
        n_camadas = np.random.randint(1, 4)
        n_neuronios = np.random.randint(1, 1001)
    
        cromossomo = np.binary_repr(n_camadas, 2) + np.binary_repr(n_neuronios, 10)
    
        population.append(cromossomo)
    
    return population

def avaliar_individuo(cromossomo, x_train, y_train, x_test, y_test):
    """
    Avalia o indivíduo.
    
        Parâmetros
        
        cromossomo indivíduo
        x_train: dados de treino
        y_train: labels de treino
        x_test: dados de teste
        y_test: labels de teste
    """
    n_camadas = cromossomo[:2]
    n_neuronios = cromossomo[2:]
    
    n_camadas = int(n_camadas, 2)
    n_neuronios = int(n_neuronios, 2)
    
    hidden_layers = (n_neuronios for i in range(n_camadas))
    
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers)
    
    mlp.fit(x_train, y_train)
    
    y_pred = mlp.predict(x_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    return acc

def crossover(cromossomo1, cromossomo2):
    
    corte = np.random.randint(1, 11)
    
    individuo_3 = cromossomo1[:corte] + cromossomo2[corte:]
    individuo_4 = cromossomo2[:corte] + cromossomo1[corte:]
    
    return individuo_3, individuo_4

pop = generate_population(10)



