# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 21:45:11 2018

@author: alu2015111446
"""

import requests
import time

list = open('lista.txt')
lista = list.read()
list.close

for linha in lista.split('\n'):
    print('https://www.nomedosite.com/{}'.format(linha))
    #response = requests.get('https://www.nomedosite.com/{}'.format(linha))
    time.sleep(3)


     #if response.status_code == '200' :


       #print response