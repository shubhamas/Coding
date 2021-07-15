import torch 
import matplotlib.pyplot as plt
import numpy as np

# a = np.array([1,2,3])
# print(type(a),a.shape,a[0],a[1],a[2])

# b = np.zeros((2,2))

# c = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

# print(c)

# print(c[:,1:4]) #first two rows 
# #  print(:)

# print(c.T)

x = np.array([0,1,2,3,4])
y = np.array([0,2,3,6,8])

plt.figure(figsize=(8,5),dpi=100)
plt.plot(x,y, 'b^--',label='2x')
plt.show()


x = torch.rand(5, 3)
print(x)