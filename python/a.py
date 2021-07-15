import numpy as np

class Node:
    """A node in a computation graph."""
    def __init__(self, value, fun, parents):
        self.parents = parents
        self.value = value
        self.fun = fun 


val_z = 1.5 
z = Node(val_z, None, [])
val_t1 = np.negative(val_z)
t1 = Node(val_t1,np.negative, [z])
val_t2 = np.exp(val_t1)
t2 = Node(val_t2, np.exp, [t1])
val_t3 = np.add(val_t2, 1)
t3 = Node(val_t3, np.add, [t2])
val_y = np.reciprocal(val_t3)
y = Node(val_y, np.reciprocal, [t3])
print(round(y.value,3))

print(id(t1.parents))
print(id(z))