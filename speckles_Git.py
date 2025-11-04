# -*- coding: utf-8 -*-
"""Speckles iterations.ipynb

"""

print('Importando os módulos python necessários...')
import cv2
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import pinv, inv
import requests, gzip, os, hashlib
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import tensorflow as tf
# from google.colab import files

print('Importados com sucesso!')

print('Setando alguns funções úteis')

#Reshape em matrizes
def reshape1(X, size):
    X= X.reshape(-1, size, size)
    return X

#Redução de dimensão
def resize(X, size):
    X= np.array([cv2.resize(X[i], (size,size), interpolation= cv2.INTER_NEAREST) for i in tqdm(range(0, X.shape[0]))])
    return X

#Normaliza os valores de intensidade para [0, 1]
def norm(x):
    x= x-np.min(x)
    x= x/np.max(x)
    return x

#Seleciona apenas as classes 0 e 1 do MNIST
def class2_MNIST(X_train, Y_train, X_test, Y_test):
    train_filter = np.where((Y_train == 0 ) | (Y_train == 1))
    test_filter = np.where((Y_test == 0 ) | (Y_test == 1))
    X_train, Y_train = X_train[train_filter], Y_train[train_filter]
    X_test, Y_test = X_test[test_filter], Y_test[test_filter]
    return X_train, Y_train, X_test, Y_test

#Filtragem da quantidade de dados do MNIST

def filter_data(X_train, Y_train, X_test, Y_test, a, b):
    i_train= a
    i_val= b

    X_train= X_train[0:i_train]
    Y_train= Y_train[0:i_train]
    X_test= X_test[0:i_val]
    Y_test= Y_test[0:i_val]
    return X_train, Y_train, X_test, Y_test

#Função para plotar várias imagens

def plot(arr, label, n):
    plt.figure(figsize=(20, 4))
    for i in range(0, n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(arr[i], cmap= 'gray')
        y= label[i]
        plt.title(f'Rótulo = {y}')
        plt.colorbar()
        plt.axis('off')
        #plt.gray()
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)

#Macropixels
def img_macrop(img, dx):
    #print(img.shape)
    mac= []
    imx= int(img.shape[0]/dx)
    imy= int(img.shape[1]/dx)

    for i in range(0, img.shape[0] , dx):
        for j in range(0, img.shape[1], dx):
            m= img[i:i+dx, j:j+dx].mean()
            mac.append(m)

    mac= np.array(mac).reshape(imx, imy)
    #print(mac.shape)
    return mac

#Filtro espacial numérico
def spatial_f(D, N_pix):
    B= np.zeros((N_pix, N_pix))
    for i in range(1, N_pix):
        for j in range(1, N_pix):
            if (D/2)**2 <= (i-0.5-N_pix/2)**2 + (j-0.5-N_pix/2)**2:
                B[i,j]= 0
            else:
                B[i,j]= np.power(-1, (i+j))
    return B

#Funções de ativação

def relu(x):
    return np.maximum(x, 0, x)

print('Fazendo download do banco de dados do Mnist... ')

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#Filtragem da quantidade de dados do MNIST
i_train= 3000
i_val= 600

X_train, Y_train, X_test, Y_test= filter_data(X_train, Y_train, X_test, Y_test, i_train, i_val)

#plot(X_train, Y_train, 7)

#Redução de dimensão
n_pix= 100

X_train= resize(X_train, n_pix)
X_test= resize(X_test, n_pix)

#Normalização
X_train= X_train/255
X_test= X_test/255


#Codficação One-hot dos labels
onehotencoder = OneHotEncoder(categories='auto')
y_train= onehotencoder.fit_transform(Y_train.reshape(-1, 1)).toarray()
y_test= onehotencoder.fit_transform(Y_test.reshape(-1, 1)).toarray()

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print('Arquivos prontos! ')

# Função para calcular a acurácia
def calculate_accuracy(predictions, labels):
    correct = np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
    accuracy = correct / labels.shape[0]
    return accuracy

#Função para gerar Speckles com entradas codificas na fase dos speckles
def speckle(X, ph, D):
    E= np.exp(1j*(ph-X))
    Es= np.fft.fft2(E)*D
    Es= np.fft.ifft2(Es)
    return Es

def calcular_beta(H, T, C):
    num_neuronios = H.shape[1]
    num_amostras = H.shape[0]

    if num_neuronios > num_amostras:
        # Solução rápida 1: beta = H^T (I/C + H H^T)^(-1) T
        I = np.eye(num_amostras)
        regularizacao = I / C
        H_H_T = np.dot(H, H.T)
        tmp = np.dot(H.T, inv(regularizacao + H_H_T))
        beta = np.dot(tmp, T)

    else:
        # Solução rápida 2: beta = (I/C + H^T H)^(-1) H^T T 
        I = np.eye(num_neuronios)
        regularizacao = I / C
        H_T_H = np.dot(H.T, H)
        inversa = inv(regularizacao + H_T_H)
        beta = np.dot(np.dot(inversa, H.T), T)

    return beta

# Parâmetro de Weibull
alfa = 0.4

# Tamanho e número de pixels
Ap = 60
D = spatial_f(Ap, n_pix)

def ElM_interactive(X_train, X_test, y_train, y_test, D, alfa, C= 0.1, n_interactions=3):

    train_accuracies = []
    test_accuracies = []
    contrast= []
    #phases = []

    for interaction in tqdm(range(n_interactions)):
        # Gerar uma fase aleatória
        ph = np.random.uniform(-np.pi, np.pi, size=[n_pix, n_pix])
        #phases.append(ph)

        # Treino
        E_train = speckle(X_train, ph, D)** alfa
        E_test = speckle(X_test, ph, D)** alfa

        I_train = np.abs(E_train) ** 2
        I_train = I_train / I_train.max()
        I_test = np.abs(E_test) ** 2
        I_test = I_test / I_test.max()
    
        C_= np.std(I_test)/np.mean(I_test)

        I_train_flat = I_train.reshape(I_train.shape[0], -1)
        I_test_flat = I_test.reshape(I_test.shape[0], -1)

        beta= calcular_beta(I_train_flat, y_train, C)
        #beta = pinv(I_train_flat) @ y_train

        y_pred_train = I_train_flat @ beta
        y_pred_test = I_test_flat @ beta

        acc_train = calculate_accuracy(y_pred_train, y_train)
        acc_test = calculate_accuracy(y_pred_test, y_test)

        train_accuracies.append(acc_train)
        test_accuracies.append(acc_test)
        contrast.append(C_)

        print(f'Interaction {interaction + 1}/{n_interactions}: Train Accuracy = {acc_train:.4f}, Test Accuracy = {acc_test:.4f}, Contrast= {C_:.4f}')

    return train_accuracies, test_accuracies, contrast

interactions= 10
train_accuracies, test_accuracies, contrast = ElM_interactive(X_train, X_test, y_train, y_test, D, alfa, n_interactions= interactions)

#Media das acurárcias e dos contrastes
M_= np.array(test_accuracies)
print(M_)
C_= np.array(contrast)

Med_acc= np.mean(M_)
Med_cont= np.mean(C_)
print('Média das Acurácias :')
print(Med_acc)

print('Média dos contrastes:')
print(Med_cont)

It= np.linspace(1,interactions, interactions)

zipped=zip(It, M_,C_)
header = ['Iteration','Test_Accuracy', 'Contrast']
import csv
with open('Iterations_alpha_{}_D_{}_Train_{}_Test_{}.csv'.format(alfa, Ap, i_train, i_val), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(i for i in header)
    for j in zipped:
        writer.writerow(j)

# Plotando o gráfico da acurácia em função do número de interações
plt.figure(figsize=(10, 5))
#plt.plot(range(1, 16), train_accuracies, label='Train Accuracy')
plt.plot(range(1, interactions+1), test_accuracies, label='Test Accuracy')
plt.xlabel('Number of Interactions')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Interactions')
plt.legend()
plt.grid(True)
plt.show()

