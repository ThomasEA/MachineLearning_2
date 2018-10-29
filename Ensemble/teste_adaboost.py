# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:37:23 2018

@author: ethomas
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split, StratifiedKFold
from sklearn.naive_bayes      import GaussianNB
from sklearn.tree             import DecisionTreeClassifier
from sklearn.metrics          import accuracy_score, recall_score, precision_score, f1_score
from sklearn                  import metrics

from Adaboost_model           import Adaboost

import copy

def load_data():
    """
        Carrega o dataset e ajusta os labels para utilização pelo Adaboost
        
        Return:
            
            data = Dataframe
    """
    df = pd.read_csv('../datasets/sonar.all-data.csv', delimiter=',', header=None)
    
    #Ajusta os labels conforme especificação do algoritmo Adaboost
    df.iloc[:,-1] = [-1 if x == 'R' else 1 for x in df.iloc[:, -1]]
    
    return df

def calc_recall_score(true_y, pred_y, pos_label):
    tp = 0.
    fn = 0.
    
    for ty, py in zip(true_y, pred_y):
        if ty == pos_label and ty == py:
            tp += 1
        elif ty == pos_label and ty != py:
            fn += 1
            
    return tp / (tp + fn)
    
def calc_precision_score(true_y, pred_y, pos_label):
    tp = 0.
    fp = 0.
    
    for ty, py in zip(true_y, pred_y):
        if (ty == pos_label and ty == py):
            tp += 1
        elif (py == pos_label and ty != py):
            fp += 1
            
    return tp / (tp + fp)

def calc_f1_score(true_y, pred_y, pos_label):
    precision = calc_precision_score(true_y, pred_y, pos_label)
    recall = calc_recall_score(true_y, pred_y, pos_label)
    
    return 2* (precision*recall)/(precision+recall)


#------------------------------------------------------------------------------


random_state = 1
n_folds = 10
n_clfs = 40

df = load_data()

train, test = train_test_split(df, test_size=0.2, stratify=df.iloc[:,-1], random_state=random_state)

clfs = []

accuracy = []

max_idx_accur = 0

#armazena o melhor modelo do treinamento
best_model = None

for i in range(1,n_clfs + 1, 1):
    clfs.append(DecisionTreeClassifier(criterion='entropy', max_depth=1))
    #clfs.append(GaussianNB())

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    boosting = Adaboost(copy.copy(clfs))
    
    acc_cv = []
    
    #Cross-validation
    for train_idx, test_idx in kf.split(train, train.iloc[:, -1]):
        
        X_train, y_train = df.iloc[train_idx, :-1], df.iloc[train_idx, -1]
        X_test , y_test  = df.iloc[test_idx , :-1], df.iloc[test_idx, -1]
    
        pred = boosting.fit(X_train, y_train)
        
        pred_test = boosting.predict(X_test)
        
        acc_cv.append(accuracy_score(y_test, pred_test))
    
    if len(accuracy) == 0 or np.mean(acc_cv) > max(accuracy):
        max_idx_accur = i
        best_model = boosting
    
    accuracy.append(np.mean(acc_cv))
        
fig = plt.figure(figsize=(6, 6)) 
ax  = fig.add_subplot(111)

ax.set_title('Performance on trainning with 10-fold validation')
ax.plot(range(1, len(accuracy) + 1), accuracy, '-b')
ax.set_xlabel('Number of estimators')
ax.set_xticks(np.arange(1, len(accuracy) +1, len(accuracy) / 10))
ax.set_yticks(np.arange(min(accuracy), max(accuracy), 0.025))
ax.set_ylabel('Accuracy')
plt.axhline(max(accuracy), linestyle='--', color='red', linewidth=1)
plt.axvline(max_idx_accur, linestyle='--', color='red', linewidth=1)
plt.show()

test_X, test_y = test.iloc[:, :-1], test.iloc[:, -1]
predicted = boosting.predict(test_X)

accuracy_test = accuracy_score(test_y, predicted)

rM = calc_recall_score(test_y, predicted, pos_label=1)
rR = calc_recall_score(test_y, predicted, pos_label=-1)
pM = calc_precision_score(test_y, predicted, pos_label=1)
pR = calc_precision_score(test_y, predicted, pos_label=-1)

print('-------------------------------------------------')
print('Appliyng the best model found on the test dataset ')
print('-------------------------------------------------')
print('> Number of estimators: ', len(best_model.T))
print('> Accuracy on test dataset: ', accuracy_test)
print('-------------------------------------------------')
print('> Class M (1): ')
print('\t> Recall: ', rM, ' / Precision: ', pM)
print('> Class R (-1): ')
print('\t> Recall: ', rR, ' / Precision: ', pR)
print('-------------------------------------------------')
print('F1 Score: ', calc_f1_score(test_y, predicted, pos_label=1))

