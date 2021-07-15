import numpy as np
from numpy import *

for i in [1,23,3]:
    print(i)

x = np.zeros((3,3))
y = [1,2,5]
x[:,0]=[1,2,3]
x[:,1]=[4,5,6]
x[:,2]=[7,8,9]
print(x)
a = [x[:,0]]
a = np.array(a)
print(a.T)
z = a.T @ a
print(z)