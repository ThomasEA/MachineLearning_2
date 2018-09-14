# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 10:37:36 2018

@author: ethomas

Implementação AdaBoost

"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

#Importando os classificadores
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def plot_fold_balance(balance):
    
    #monta o array para plotar
    data_r = []
    data_m = []
    
    qt_folds = 0
    
    for item in balance:
        qt_folds += 1
        for c in item['class']:
            if c['label'] == 'R':
                data_r.append((c['train'], c['test']))
            elif c['label'] == 'M':
                data_m.append((c['train'], c['test']))
    
    ind = np.arange(1,qt_folds + 1, 1)
    width = 0.25
    
    fig = plt.figure()
    #ax = fig.subplots(111)
    
    for i in range(len(data_r[0])):
        y = [d[i] for d in data_r]
        b = plt.bar(ind + i * width, y, width)
        
    fig.legend(['Treino', 'Teste'])
    """
    plt.title('Balanceamento FOLD Treino')
    plt.xlabel('Folds')    
    plt.xticks(np.arange(1,10,1))
    """
    """
    plt.bar(np.arange(1, qt_folds + 1, 1), data_r_train)
    plt.bar(np.arange(1, qt_folds + 1, 1), data_m_train)
    plt.show()
    
    plt.title('Balanceamento FOLD Teste')
    plt.xlabel('Folds')    
    plt.xticks(np.arange(1,10,1))
    
    plt.bar(np.arange(1, qt_folds + 1, 1), data_r_test)
    plt.bar(np.arange(1, qt_folds + 1, 1), data_m_test)
    plt.show()
    """
    """
    
    plt.bar(train_r)
    """
    

    return


metric = { 'metrics': { 'accuracy': [], 'precision': [], 'recall': [] } }

models = [
    { 'name': 'MLP', 'model': MLPClassifier(), 'metrics': metric.copy() },
    { 'name': 'NB' , 'model': GaussianNB(), 'metrics': metric.copy() },
    { 'name': 'SVM', 'model': SVC(C=10, gamma=0.1), 'metrics': metric.copy() }
]

#carrega o dataset
df = pd.read_csv('../datasets/sonar.all-data.csv', delimiter=',', header=None)

data  = np.array(df.iloc[:,:-1])
labels = np.array(df.iloc[:,-1])

#verificação do balancemamento das classes
labels_values, count_labels = np.unique(labels, return_counts=True)

plt.bar(labels_values, count_labels)
plt.title('Balanceamento de classes')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.show()

#Configura para 10 folds, e usa Shuffle pq os dados estão
#claramente dividos entre suas classes no dataset
kf = KFold(n_splits=10, shuffle=True, random_state = 50)

kfold_balance = []
cnt_fold = 0

for train_idx, test_idx in kf.split(data):
    
    X_train, y_train = data[train_idx], labels[train_idx]
    X_test , y_test  = data[test_idx], labels[test_idx]

    


    """
    cnt_fold += 1
    
    kfold_balance.append({'fold': cnt_fold,
                          'class':[
                                   { 'label': 'R',  
                                    'train': len(y_train[y_train[:] == 'R']) / len(y_train) * 100, 
                                    'test': len(y_test[y_test[:] == 'R']) / len(y_test) * 100 
                                   },
                                   { 'label': 'M',  
                                    'train': len(y_train[y_train[:] == 'M']) / len(y_train) * 100, 
                                    'test': len(y_test[y_test[:] == 'M']) / len(y_test) * 100 
                                   }
                                   ]})

    """
    
    
    
    
plot_fold_balance(kfold_balance)

"""
print('Fold treino: ', X_train.shape, ' / Fold Teste: ', X_test.shape)
print('\t\t\tR: %d (%.2f%%) / M: %d (%.2f%%)' % 
          (len(y_train[y_train[:] == 'R']), 
           len(y_train[y_train[:] == 'R']) / len(y_train) * 100, 
           len(y_train[y_train[:] == 'M']),
           len(y_train[y_train[:] == 'M']) / len(y_train) * 100))
"""

