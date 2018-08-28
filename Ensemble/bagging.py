# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 21:58:20 2018

@author: alu2015111446

Dados -> Diabetes
Modelos -> MLP, SVM, RANDOM FOREST (BAGGING), BOOSTING TREE (BOOSTING)
Avaliação -> CROSS VALIDATION (10 FOLDS), ACURÁCIA, PRECISÃO, RECALL
"""
import numpy as np

#Para o voto majoritário
from collections import Counter


def balancear_dados(x, y, estrategia='oversampling'):
    
    if estrategia == 'oversampling':
        
        cnt = Counter()
        
        for cat in y:
            cnt[cat] += 1
        
        classe_majo = cnt.most_common()[0][0]
        num_samples = cnt.most_common()[0][1]
        
        dados_bal = []
        labels_bal = []
        
        for classe in np.unique(y):
            
            if not classe == classe_majo:
                dados = x[y == classe]
                labels = y[y == classe]
                
                sampled_dados, sampled_label = resample(dados, labels, n_samples=num_samples)
                
                dados_bal.append(sampled_dados)
                labels_bal.append(sampled_label)
            
            else:
                dados_bal.append(x[y == classe_majo])
                labels_bal.append(y[y == classe_majo])
                
        return np.vstack(dados_bal), np.hstack(labels_bal)


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

#Para fazer a amostragem dos dados para cada modelo
from sklearn.utils import resample

num_samples = 400

modelos = ['mlp','svm']#,'random_forest','gradient_boosting']

kf = KFold(n_splits=10)

performance = { 'bagging': {
        'accuracy': [],
        'recall': [],
        'precision': []
        }
            }

for train_idx, test_idx in kf.split(data):
    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    predictions = []
    ensemble_predictions = []
    
    for modelo in modelos:
        sample_X_train, sample_y_train = resample(X_train, 
                                                  y_train, 
                                                  n_samples=num_samples)
        
        #Balanceamento
        sample_X_train, sample_y_train = balancear_dados(sample_X_train, sample_y_train)
        
        if modelo == 'mlp':
            md = MLPClassifier()
        elif modelo == 'svm':
            md = SVC(C=10, gamma=0.1)
            
            
        md.fit(sample_X_train, sample_y_train)
        
        predictions.append(md.predict(X_test))
        
    predictions = np.vstack(predictions)

    #Voto majoritário
    cnt = Counter()
    for col in range(predictions.shape[1]):
        votes = predictions[:,col]
        for vote in votes:
            cnt[vote] += 1
        
        ensemble_predictions.append(cnt.most_common()[0][0]) #Categoria mais votada
        
    ens_acc  = accuracy_score(y_test, ensemble_predictions)
    ens_rec  = recall_score(y_test, ensemble_predictions)
    ens_prec = precision_score(y_test, ensemble_predictions)
    
    performance['bagging']['accuracy'].append(ens_acc)
    performance['bagging']['recall'].append(ens_rec)
    performance['bagging']['precision'].append(ens_prec)
    
print(performance)    


    
    

