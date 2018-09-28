# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:41:01 2018

@author: ethomas

Classificador Adaboost

"""
import numpy as np

class Adaboost:
    
    def __init__(self, T):
        """
        Implementação de boosting para classificadores
        
        Parâmetros:
            
            T = Array de classificadores
        """
        self.T = T
        self.alphas = None
    
    def fit(self, X, y):
        """
        Treina os modelos
        
        Parâmetros:
            
            X = Dados de treino
            
            y = Labels de treino
        """
        
        #inicializa os pesos
        weights = np.ones(len(X)) / float(len(X))
        
        i = 0
        
        alphas = []
        
        #para cada classificador
        for clf in self.T:
            
            i += 1
            
            clf.fit(X, y, sample_weight=weights)
            
            pred_train = clf.predict(X)
            
            miss = [int(d) for d in (pred_train != y)]
            err   = np.dot(weights, miss)/np.sum(weights)
            alpha = np.log((1 - err)/err)
            
            #guarda o alpha para o classificador
            alphas.append(alpha)
            
            #ajusta os pesos
            weights = np.multiply(weights, np.exp([alpha * float(x) for x in miss]))/float(weights.sum())
            
            accuracy = sum([int(d) for d in (pred_train == y)]) / len(pred_train)
            
            print('Accuracy on model {0}: {1}', i, accuracy)
        
        self.alphas = alphas