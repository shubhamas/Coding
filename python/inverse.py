import numpy as np 
lst = [12,3,4,5,6,6,7,8]

# Calculating Inverse of Matrix
a = np.array([[6,1,1],[3,4,5],[6,7,8]])
print(a)
b = np.linalg.inv(a)
print(b)

# Function used to calculate SVD
u , s, v = np.linalg.svd(a)
# print(np.linalg.svd(a))
print(u)
print(s)
print(v)

#Function to calculate LU decomposition 
# l,u = np.linalg.

