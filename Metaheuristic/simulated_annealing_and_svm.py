# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:09:17 2018

@author: alu2015111446

Metaheurísticas para seleção de parâmetros

Simulated Annealing + SVM

Implementração de Simulated Annealing para regularizar parâmetros
    - gamma (Kernel) >= 0
    - C >= 0
    
******* ATENÇÃO ********
Este script não está concluído
Há um problema nas iterações
************************
"""
from __future__ import division
from copy import copy

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

"""
Gera a vizinhança em torno de um ponto
    
    s: solution ([gamma, C])
"""
def generate_neighbors(s, r=1):
    k = copy(s)
    
    k[0] += (2 * np.random.rand() - 1) * r * s[0] 
    k[1] += (2 * np.random.rand() - 1) * r * s[1] 
    
    k[0] = max(1e-2, k[0])
    k[1] = max(1e-2, k[1])
    
    return k

def f(y_true, y_pred):
    """
    Função de avaliação que deve maximizar a acurácia
    """
    #return accuracy_score(y_true, y_pred)
    return np.linalg.norm(y_true - y_pred)

"""
Calcula a probabilidade de aceitação
    f: função de aceitação
    f_: 
    t: temperatura
"""
def prob_aceitacao(f, f_, t):
    return np.exp( - ( f - f_ ) / t)

def apply_model(model, x, y, x_val, y_val):
    """
    Aplica o modelo
    """
    model.fit(x, y)
    
    y_pred = model.predict(x_val)
    
    return f(y_val, y_pred)

"""
    initial: solução inicial
    temp: temperatura
    max_iter: numero máximo de iterações
    alfa: regula a temperatura
"""
def simulated_annealing(initial, temp, max_iter, alpha, X_train, y_train, X_val, y_val):
    
    accuracy_hist = []
    
    accuracy_now = 0
    accuracy_old = -1
    iterT = 0

    while (accuracy_now - accuracy_old > 1e-3):
        while (iterT <= max_iter):
            iterT += 1
            #Gera solução aleatória na vizinhança da solução atual
            nova_solucao = generate_neighbors(initial)
            
            #Treina os modelos com ambos os parâmetros para comparar
            m1 = SVC(C=initial[1], gamma=initial[0])
            f_old = apply_model(m1, X_train, y_train, X_val, y_val)
            
            m2 = SVC(C=nova_solucao[1]*100, gamma=nova_solucao[0]*100)
            f_new = apply_model(m2, X_train, y_train, X_val, y_val)
            
            print('Old -> accur: ', f_old, initial)
            print('New -> accur: ', f_new, nova_solucao)
            
            #avaliação
            if (f_new >= f_old):
                initial = copy(nova_solucao)
                
                accuracy_old = f_old
                accuracy_now = f_new
            else:
                #dá uma chance. gera um valor randômico entre 0 e 1
                v = np.random.rand()
                #calcula a probabilidade de aceitação
                prob = prob_aceitacao(accuracy_now, accuracy_old, temp)
                
                if (v <= prob):
                    initial = copy(nova_solucao)
                    
                    accuracy_old = f_old
                    accuracy_now = f_new
            
        iterT = 0
        temp *= alpha
        
        accuracy_hist.append(accuracy_now)
    
    return accuracy_hist


solucao_inicial = [10, 10]
temperatura = 100
max_iteracoes = 100
alpha = 0.9

df = np.genfromtxt('../datasets/diabetes.csv', delimiter=',')
data = df[:,:-1]
labels = df[:,-1]

#Para descobrir os hyperparâmetros, não se deve usar os dados que serão utilizados
#na validação do modelo
#
#Neste caso ficou:
# Separa treino <-> teste
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
#Separa pporção do treino para validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

ret = simulated_annealing(solucao_inicial, temperatura, max_iteracoes, alpha, X_train, y_train, X_val, y_val)
#, 6.4520052904296925 -> 242640687119285
print(apply_model(SVC(), X_train, y_train, X_val, y_val))

model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
    
print(f(y_val, y_pred))
