# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:37:36 2018

@author: ethomas

AdaBoost Teste 4

1) Given (x_1,y_1),…..,(x_m,y_m) where x_i ∈ X, y_i ∈ {-1, +1}
2) Initialize: D1(i) = 1/m for i = 1, …,m.

"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_hastie_10_2

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import accuracy_score, recall_score, precision_score

#Importando os classificadores
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def load_data():
    df = pd.read_csv('../datasets/sonar.all-data.csv', delimiter=',', header=None)
    
    #Ajusta os labels conforme especificação do algoritmo Adaboost
    df.iloc[:,-1] = [-1 if x == 'R' else 1 for x in df.iloc[:, -1]]
    
    return df

def split_train_test(data, test_size):
    train, test = train_test_split(data, test_size=test_size, random_state=1)#, shuffle=True)#, random_state = 1)
    return train.iloc[:,:-1], train.iloc[:,-1], test.iloc[:,:-1], test.iloc[:,-1]


def get_model(name):
    return [item['model'] for item in models if item['name'].lower == name.lower][0]


def initial_fit(model, X_train, y_train, X_test, y_test):
    """
        Faz uma predição inicial e calcula a taxa de erro no início do processo
        
        Return:
            err_train = Taxa de erro no treinamento
            
            err_test = Taxa de erro no teste
    """
    model.fit(X_train, y_train)
    
    pred_train = model.predict(X_train)
    
    pred_test = model.predict(X_test)
    
    return error_rate(pred_train, y_train), error_rate(pred_test, y_test), calc_metrics(pred_train, y_train), calc_metrics(pred_test, y_test)


def calc_erro(pred_y, true_y, weights):
    check = [int(d) for d in (pred_y != true_y)]
    
    miss_v = [x if x == 1 else -1 for x in check]
    
    return np.dot(weights, check) / float(weights.sum()), miss_v

def calc_alpha(err):
    return 0.5 * np.log((1 - err)/float(err))

def change_weights(weights, pred_y, true_y):
    
    err_m, miss_v = calc_erro(pred_y, true_y, weights)
    
    alpha = calc_alpha(err_m)
    
    weights = np.multiply(weights, np.exp([alpha * float(x) for x in miss_v]))/float(weights.sum())
    
    return weights, alpha

def error_rate(pred_y, true_y):
    return sum(pred_y != true_y)/float(len(true_y))

def calc_metrics(pred_y, true_y):
    accuracy = accuracy_score(true_y, pred_y)
    recall = recall_score(true_y, pred_y)
    precision = precision_score(true_y, pred_y)
    return { 'accuracy': accuracy, 'recall': recall, 'precision': precision }

def adaboost(train_X, train_y, test_X, test_y, model, max_iter, n_folds):

    err_rate_iter_train = []#np.zeros(max_iter + 1)
    err_rate_iter_test = []#np.zeros(max_iter + 1)
    
    #X_train, y_train, X_test, y_test = split_train_test(data, test_size=test_size)

    #gera os pesos iniciais
    w = np.ones(len(X_train)) / len(X_train)
    
    #para referência, treina uma vez sem os pesos e pega a taxa de erro inicial
    #err_m_train, err_m_test = initial_fit(model, X_train, y_train, X_test, y_test)

    #err_rate_iter_train[0] = err_m_train
    #err_rate_iter_test[0] = err_m_test

    pred_train, pred_test = [np.zeros(len(train_X)), np.zeros(len(test_X))]
    
    for i in range(max_iter):
        
        model.fit(train_X, train_y, sample_weight=w)
        
        predict_train = model.predict(train_X)
        
        predict_test = model.predict(test_X)
        
        w, alpha_m = change_weights(w, predict_train, train_y)
        
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [x * alpha_m for x in predict_train])]
        pred_test = [sum(x) for x in zip(pred_test, 
                                         [x * alpha_m for x in predict_test])]
        
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
        
    err_rate_iter_train.append(error_rate(pred_train, train_y))
    err_rate_iter_test.append(error_rate(pred_test, test_y))
    
    metrics_train, metrics_test = calc_metrics(pred_train, train_y), calc_metrics(pred_test, test_y)
    
    #err_rate = [sum(x) for x in zip(err_rate, err_rate_iter)]
        
    return err_rate_iter_train, err_rate_iter_test, metrics_train, metrics_test

#-------------------------------------------------------
    
metric = { 'metrics': { 'accuracy': [], 'precision': [], 'recall': [] } }

models = [ 
        { 'name': 'TREE', 'model': DecisionTreeClassifier(max_depth=1, random_state=1), 'metrics': metric.copy() },
        { 'name': 'MLP' , 'model': MLPClassifier(), 'metrics': metric.copy() },
        { 'name': 'NB'  , 'model': GaussianNB(), 'metrics': metric.copy() },
        { 'name': 'SVM' , 'model': SVC(C=10, gamma=0.1), 'metrics': metric.copy() }
        ]

data = load_data()

model = get_model('TREE')

max_iter = 15

X_train, y_train, X_test, y_test = split_train_test(data, test_size=0.2)

#para referência, treina uma vez sem os pesos e pega a taxa de erro inicial
err_m_train, err_m_test, m_tr, m_tst = initial_fit(model, X_train, y_train, X_test, y_test)

    
err_train = [err_m_train]
err_test = [err_m_test]

acc_train = [m_tr.get('accuracy')]
acc_test = [m_tst.get('accuracy')]

for i in range(10,105,5):
    er_tr, er_tst, m_tr, m_ts = adaboost(X_train, y_train, X_test, y_test, model, max_iter=i, n_folds=10)
    err_train.append(er_tr[0])
    err_test.append(er_tst[0])
    acc_train.append(m_tr.get('accuracy'))
    acc_test.append(m_ts.get('accuracy'))
    #er_i = adaboost_clf(y_train, X_train, y_test, X_test, i, model)
    #err_train.append(er_i[0])
    #err_test.append(er_i[1])
    
    

#print(err_train, err_test)

df = pd.DataFrame([err_train, err_test]).T
df.columns = ['Treino', 'Teste']
plt.plot(np.arange(0,100,5), df['Treino'])
plt.plot(np.arange(0,100,5), df['Teste'])
plt.ylabel('Taxa de erro')
plt.xlabel('Iterações')
plt.legend(['Treino','Teste'])
plt.grid()
plt.xticks(np.arange(0,105,5))
plt.axhline(y=err_test[0], linewidth=1, color = 'red', ls = 'dashed')
plt.axhline(y=err_train[0], linewidth=1, color = 'g', ls = 'dashed')
plt.show()

df_accur = pd.DataFrame([acc_train, acc_test]).T
df_accur.columns = ['Treino', 'Teste']
plt.plot(np.arange(0,100,5), df_accur['Treino'])
plt.plot(np.arange(0,100,5), df_accur['Teste'])
plt.ylabel('Acurácia')
plt.xlabel('Iterações')
plt.legend(['Treino','Teste'])
plt.grid()
plt.xticks(np.arange(0,105,5))
plt.show()



#plt.xticks(range(0,1,1))

#print(models[models['name'] == 'MLP'])

#max_iter = 10

#for i in range(max_iter):
    
    



