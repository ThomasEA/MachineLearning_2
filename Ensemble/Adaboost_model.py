# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:41:01 2018

@author: ethomas

Classificador Adaboost

"""
import numpy as np

class Adaboost:
    
    alpha = []
    
    def __init__(self, T):
        """
        Implementação de boosting para classificadores
        
        Parâmetros:
            
            T = Array de classificadores
        """
        self.T = T
    
    def fit(self, X, y):
        
        #inicializa os pesos
        weights = np.ones(len(X)) / float(len(X))
        
        i = 1
        
        #para cada classificador
        for clf in self.T:
            
            print('Classificador: ', i)
            
            i += 1
            
        