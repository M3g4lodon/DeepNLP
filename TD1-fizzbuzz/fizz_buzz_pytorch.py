# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:23:25 2018

@author: Herve Le Borgne

Fizz Buzz in pyTorch
see http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
"""

import numpy as np
import torch
from torch.autograd import Variable

NUM_DIGITS = 10

# codage binaire d'un chiffre (max NUM_DIGITS bits)
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode(i):
    if   i % 15 == 0: return 3
    elif i % 5  == 0: return 2
    elif i % 3  == 0: return 1
    else:             return 0

def fizz_buzz_index(i):
    if   i % 15 == 0: return 3
    elif i % 5  == 0: return 2
    elif i % 3  == 0: return 1
    else:             return 0

# données d'entraînement (X) et labels (Y)
X=Variable(torch.FloatTensor(np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])))
Y=Variable(torch.LongTensor(np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])).squeeze())

# données de test
X_test=Variable(torch.FloatTensor(np.array([binary_encode(i, NUM_DIGITS) for i in range(1,101)])))

# nombre de neurones dans la couche cachée
NUM_HIDDEN = 100

# définition du MLP à 1 couche cachée (non linearite ReLU)
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4)
    )

# fonction de coût 
loss_fn = torch.nn.CrossEntropyLoss()

# optimiseur
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)

# affichage attendu par l'application
def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

# on lance les calculs
BATCH_SIZE = 128
for epoch in range(10000):
    for start in range(0, len(X), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = X[start:end]
        batchY = Y[start:end]

        # prediction et calcul loss
        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)
    
        # mettre les gradients à 0 avant la passe retour (backward)
        optimizer.zero_grad()
    
        # rétro-propagation
        loss.backward()
        optimizer.step()

    # calcul coût  (et affichage)
    loss = loss_fn( model(X), Y)
    if epoch%100 == 0:
        print(epoch, 1-loss.data[0])

    # visualisation des résultats en cours d'apprentissage
    if(epoch%1000==0):
        Y_test_pred = model(X_test)
        val, idx = torch.max(Y_test_pred,1)
        ii=idx.data.numpy()
        numbers = np.arange(1, 101)
        output = np.vectorize(fizz_buzz)(numbers, ii)
        print(output)

# Sortie finale (calcul plus compact des predictions)
Y_test_pred = model(X_test)
print(Y_test_pred)
predictions = zip(range(1, 101), list(Y_test_pred.max(1)[1].data.tolist()))
print("============== Final result ============")
print([fizz_buzz(i, x) for (i, x) in predictions])
print(f"{np.average(Y_test_pred==np.vectorize(fizz_buzz_index)(numbers)):0.0%} erreurs sur les données de test")

