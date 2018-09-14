# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:37:36 2018

@author: ethomas

Implementação AdaBoost

"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#Importando os classificadores
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def load_data():
    #carrega o dataset
    df = pd.read_csv('../datasets/sonar.all-data.csv', delimiter=',', header=None)

    return df

def split_train_test(data, test_size):
    
    train, test = train_test_split(data, test_size=test_size, shuffle=True, random_state = 50)
    
   
    return train.iloc[:,:-1], train.iloc[:,-1], test.iloc[:,:-1], test.iloc[:,-1]

#-------------------------------------------------------
    
metric = { 'metrics': { 'accuracy': [], 'precision': [], 'recall': [] } }

models = [
    { 'name': 'MLP', 'model': MLPClassifier(), 'metrics': metric.copy() },
    { 'name': 'NB' , 'model': GaussianNB(), 'metrics': metric.copy() },
    { 'name': 'SVM', 'model': SVC(C=10, gamma=0.1), 'metrics': metric.copy() }
]

data = load_data()

X_train, y_train, X_test, y_test = split_train_test(data, test_size=0.2)

print([d.get('name') == 'MLP' for d in models])

m = [d.get('name') == 'MLP' for d in models]

x = models[(m)]

#print(models[models['name'] == 'MLP'])

#max_iter = 10

#for i in range(max_iter):
    
    



