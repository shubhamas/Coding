import torch
import torch.nn as nn 

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7]) 


# matrix multiplications
x1 = torch.rand((2,3))
x2 = torch.rand((3,5))
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

# matrix exponentiation
matrix_exp = torch.rand(5,5)
print(matrix_exp.matrix_power(5))

# element wise mult
z = x + y 

# dot product 
print(z)

# batch matir multiplication 

batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)

# example of brodcastimg 
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))




