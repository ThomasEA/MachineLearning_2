# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 21:58:20 2018

@author: alu2015111446

Dados -> Diabetes
Modelos -> MLP, SVM, RANDOM FOREST (BAGGING), BOOSTING TREE (BOOSTING)
Avaliação -> CROSS VALIDATION (10 FOLDS), ACURÁCIA, PRECISÃO, RECALL
"""
import numpy as np

df = np.genfromtxt('../datasets/diabetes.csv', delimiter=',')

data = df[:,:-1]
labels = df[:,-1]

"""
Normalização
"""
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

np.min(data[:,0]), np.max(data[:,0])

"""
Treinamento
"""
#Modelos
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#K-fold
from sklearn.model_selection import KFold

#Métricas
from sklearn.metrics import accuracy_score, recall_score, precision_score

performance = {}

modelos = ['mlp','svm','random_forest','gradient_boosting']

for modelo in modelos:
    performance[modelo] = {
                'acuracia': [],
                'recall': [],
                'precision': []
            }

kf = KFold(n_splits=10)

for train_idx, test_idx in kf.split():
    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    mlp = MLPClassifier()
    svm = SVC(C=10, gamma=0.1)
    rf = RandomForestClassifier()
    bt = GradientBoostingClassifier(n_estimators=10)
    
    mlp.fit(X_train, y_train)
    #... para cada classificador
    #...
    
    predict = mlp.predict(X_test)
    
    mlp_acc = accuracy_score(y_test, predict)
    recall_score(y_test, predict)
    precision_score(y_test, predict)
    
    performance['mlp']['acuracia'].append(mlp_acc)