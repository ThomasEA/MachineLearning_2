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
        fit_predicted = np.zeros(len(y))
        
        #para cada classificador
        for clf in self.T:
            
            i += 1
            
            clf.fit(X, y, sample_weight=weights)
            
            predicted = clf.predict(X)
            
            miss = [int(d) for d in (predicted != y)]
            err   = np.dot(weights, miss)/np.sum(weights)
            alpha = np.log((1 - err)/err)
            
            #guarda o alpha para o classificador
            alphas.append(alpha)
            
            #ajusta os pesos
            weights = np.multiply(weights, np.exp([alpha * float(x) for x in miss]))/float(weights.sum())
            
            accuracy = sum([int(d) for d in (predicted == y)]) / len(predicted)
            
            fit_predicted = [sum(x) for x in zip(fit_predicted, [x * alpha for x in predicted])]
        
        self.alphas = alphas
        
        return np.sign(fit_predicted)
        
    def predict(self, X, y):
        
        predicted = np.zeros(len(X))
        
        for model, alpha in zip(self.T, self.alphas):
            
            pred = model.predict(X)
            
            accuracy = sum([int(d) for d in (pred == y)]) / len(pred)
            
            #print(accuracy)
            
            predicted = [sum(x) for x in zip(predicted, [x * alpha for x in pred])]
            
        predicted = np.sign(predicted)
        
        return predicted