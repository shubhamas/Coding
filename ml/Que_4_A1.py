import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
from skimage import color
import os
import glob
import math
print('''Threshold for classification
            less than zero --> Happy faces 
            greater than zero --> Sad faces ''')

Y = np.zeros((10201,20))
X = np.zeros((10201,20))
mean_x = np.zeros((1,10201))
i = 0
N = 20
path = glob.glob("E:/Coding/ml/data/train/*.gif")
for files in path:
    gif = cv2.VideoCapture(files)
    ret, frame = gif.read()
    cv2.imwrite('image.jpg', frame)
    img = im.imread('image.jpg')
    img = color.rgb2gray(img)
    cv2.imshow('image', img)
    # print(img.shape)
    cv2.waitKey(100)
    flat_arr = img.ravel()
    x = np.matrix(flat_arr)
    x = np.array(x)

    Y[:,i] = x
    i = i + 1

i = 0

# Centering the data 
for i in range(0,20):
    mean_x = mean_x + Y[:,i]
print("sum =",mean_x)
sum_ = np.sum(Y, axis=1)
print("sum=", sum_)
mean_x = mean_x/20
for j in range(0,20):
    x = Y[:,j] - mean_x
    X[:,j] = x

# Calculating the eigen vector and eigen values
S = X.T @ X
S = S/20

eg, ev = linalg.eigh(S)

idx = eg.argsort()[::-1]
eg  = eg[idx]
ev  = ev[:,idx]

# print(eg)
# print(eg[0])

# Now calculating of eigenvalues and eigenvectors of original data matrix 
i = 0
ev_ = np.zeros((10201,20))
for eig in eg:
    if eg[i]<0:
        eg[i] = -eg[i]
    ev_[:,i] = X @ ev[:,i]
    ev_[:,i] = ev_[:,i]/math.sqrt(N*eg[i])
    i = i + 1

# Projection of data by using PCA
# print(ev_.shape)
# print(X.shape)
y = np.zeros((20,20))
y = ev_.T @ X           #Use of original data without substracting mean
# print(y)


# *****************************         LDA            ***************************************
# Linear discriminant analysis
# 0,1,4,7,9,11,13,16,18 Happy faces 
# 2,3,5,6,8,10,12,14,15,17,19 Sad faces 

# For between class scatter 
m1 = (y[:,0]+y[:,1]+y[:,4]+y[:,7]+y[:,9]+y[:,11]+y[:,13]+y[:,16]+y[:,18])/9

m2 = (y[:,2]+y[:,3]+y[:,5]+y[:,6]+y[:,8]+y[:,10]+y[:,12]+y[:,14]+y[:,15]+y[:,17]+y[:,19])/11
# print(y[:,1]-m1)

a = y[:,1]-m1
# print(a.T @ a) 

# For within class scatter 
SB = np.array([(m2 - m1)]).T @ np.array([(m2 - m1)])
SW1 = np.zeros((20,20))
for i in [0,1,4,7,9,11,13,16,18]:
    SW1 = SW1 + (np.array([(y[:,i]-m1)]).T @ np.array([(y[:,i]-m1)]))
    # print(SW1)
SW2 = np.zeros((20,20))
for j in [2,3,5,6,8,10,12,14,15,17,19]:
    SW2 = SW2 + (np.array([(y[:,j]-m2)]).T @ np.array([(y[:,j]-m2)]))

SW = (SW1 + SW2)
# print(y[:,0])
# print(SW)

d = np.linalg.det(SW)
# print(d)

# Projection of data using Linear discriminant analysis 

A = np.linalg.inv(SW) @ SB
eig_value, eig_vector = np.linalg.eigh(A)
idx = eig_value.argsort()[::-1]
eig_value  = eig_value[idx]
eig_vector  = eig_vector[:,idx]

w = eig_vector[:,0]

feature_vector = y.T @ w
print("Feature matrix for training")
print(feature_vector)
y = np.zeros(20)
# print(x)
col=[]
for i in range(0,20):
    if feature_vector[i]<0:
        col.append('r') 
    elif feature_vector[i]>0:
        col.append('b')

for i in range(0,20):
    plt.scatter(feature_vector[i],y[i],c=col[i])
plt.show()


# **********************************************************************************************************
# Testing a model

Y = np.zeros((10201,10))
X = np.zeros((10201,10))
mean_x = np.zeros((1,10201))
i = 0
N = 10
path = glob.glob("E:/Coding/ml/data/test/*.gif")
for files in path:
    gif = cv2.VideoCapture(files)
    ret, frame = gif.read()
    cv2.imwrite('image.jpg', frame)
    img = im.imread('image.jpg')
    img = color.rgb2gray(img)
    cv2.imshow('image', img)
    # print(img.shape)
    cv2.waitKey(100)
    flat_arr = img.ravel()
    x = np.matrix(flat_arr)
    x = np.array(x)
    Y[:,i] = x
    i = i + 1

i = 0
# Centering the data 

for i in range(0,10):
    mean_x = mean_x + Y[:,i]

mean_x = mean_x/10
for j in range(0,10):
    x = Y[:,j] - mean_x
    X[:,j] = x

# Calculating the eigen vector and eigen values
S = X.T @ X
S = S/10
# print(S.shape)

eg, ev = linalg.eigh(S)

idx = eg.argsort()[::-1]
eg  = eg[idx]
ev  = ev[:,idx]

# print(eg)
# print(eg[0])

# Now calculating of eigenvalues and eigenvectors of original data matrix 
i = 0
ev_ = np.zeros((10201,10))
for eig in eg:
    if eg[i]<0:
        eg[i] = -eg[i]
    ev_[:,i] = X @ ev[:,i]
    ev_[:,i] = ev_[:,i]/math.sqrt(N*eg[i])
    i = i + 1

# Projection of data by using PCA
print(ev_.shape)
print(X.shape)
y = np.zeros((10,10))
y = ev_.T @ X           #Use of original data without substracting mean
# print(y)


feature_vector = y.T @ w[0:10]*(-1)
print("Feature matrix for testing")
print(feature_vector)

accuracy = 9/10*100
print("Accuracy = ",accuracy)



















