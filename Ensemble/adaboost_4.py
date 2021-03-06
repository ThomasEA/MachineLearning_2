# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:37:36 2018

@author: ethomas

Implementação AdaBoost sem Cross-Validation

"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold

#Importando os classificadores
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def load_data():
    df = pd.read_csv('../datasets/sonar.all-data.csv', delimiter=',', header=None)
    return df

def split_train_test(data, test_size):
    train, test = train_test_split(data, test_size=test_size, shuffle=True, random_state = 50)
    return train.iloc[:,:-1], train.iloc[:,-1], test.iloc[:,:-1], test.iloc[:,-1]

def get_model(name):
    return [item['model'] for item in models if item['name'].lower == name.lower][0]

def calc_erro(pred_y, true_y, weights):
    check = [int(d) for d in (pred_y != true_y)]
    return weights.dot(check) / float(weights.sum())

def calc_alpha(err):
    return np.log((1 - err)/float(err))

def change_weights(weights, pred_y, true_y):
    
    err_m = calc_erro(pred_y, true_y, weights)
    
    alpha = calc_alpha(err_m)
    
    weights = np.multiply(weights, np.exp([alpha * float(x) for x in (pred_y != true_y)]))
    
    return weights

def error_rate(pred_y, true_y):
    return sum(pred_y != true_y)/float(len(true_y))

def adaboost(data, test_size, model, max_iter, n_folds):

    X_train, y_train, X_test, y_test = split_train_test(data, test_size=test_size)

    #gera os pesos iniciais
    w = np.ones(len(X_train)) / len(X_train)
    
    #para referência, conforme documentação, treina uma vez sem os pesos
    model.fit(X_train, y_train)

    err_rate = [np.zeros(max_iter)]

    err_rate_iter = np.zeros(max_iter)
        
    for i in range(max_iter):
        
        train_X, train_y = X_train, y_train
        test_X,  test_y  = X_test, y_test
        
        model.fit(train_X, train_y, sample_weight=w)
        
        predict_train = model.predict(train_X)
        
        predict_test = model.predict(test_X)
        
        w = change_weights(w, predict_train, train_y)
        
        err_rate_iter[i] = error_rate(predict_test, test_y)
        
    
    #err_rate = [sum(x) for x in zip(err_rate, err_rate_iter)]
        
    return err_rate_iter

            

#-------------------------------------------------------
    
metric = { 'metrics': { 'accuracy': [], 'precision': [], 'recall': [] } }

models = [ 
        { 'name': 'TREE', 'model': DecisionTreeClassifier(max_depth=8, max_leaf_nodes=2), 'metrics': metric.copy() },
        { 'name': 'MLP' , 'model': MLPClassifier(), 'metrics': metric.copy() },
        { 'name': 'NB'  , 'model': GaussianNB(), 'metrics': metric.copy() },
        { 'name': 'SVM' , 'model': SVC(C=10, gamma=0.1), 'metrics': metric.copy() }
        ]

data = load_data()

model = get_model('TREE')

err_rate = adaboost(data, 0.2, model, max_iter=50, n_folds=10)

print(err_rate)

plt.plot(err_rate)

#print(models[models['name'] == 'MLP'])

#max_iter = 10

#for i in range(max_iter):
    
    



