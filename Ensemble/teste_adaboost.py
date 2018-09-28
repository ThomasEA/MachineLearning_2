# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:37:23 2018

@author: ethomas
"""
import pandas as pd

from sklearn.model_selection  import train_test_split, KFold
from sklearn.naive_bayes      import GaussianNB
from sklearn.tree             import DecisionTreeClassifier
from sklearn.metrics          import accuracy_score

from Adaboost_model      import Adaboost

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

#------------------------------------------------------------------------------

n_folds = 10

clfs = []

for i in range(0,100):
    #clfs.append(DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2))
    clfs.append(GaussianNB())

df = load_data()

train, test = train_test_split(df, test_size=0.2)

kf = KFold(n_splits=n_folds, shuffle=True)

boosting = Adaboost(clfs)

i = 1

for train_idx, test_idx in kf.split(train):
    
    X_train, y_train = df.iloc[train_idx, :-1], df.iloc[train_idx, -1]
    X_test , y_test  = df.iloc[test_idx , :-1], df.iloc[test_idx, -1]

    pred = boosting.fit(X_train, y_train)
    
    pred_test = boosting.predict(X_test, y_test)
    
    #acc = sum(pred == y_train)/len(y_train)
    acc = accuracy_score(y_train, pred)
    acc_test = accuracy_score(y_test, pred_test)
    
    #print(boosting.alphas)
    print('Boost accuracy in train - Fold: ', i, ' - ', acc)
    print('Boost accuracy in train - Fold: ', i, ' - ', acc_test)
    print('---------------------------------------------')

    i += 1
