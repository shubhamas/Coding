
# # This a starting of Classes 

class computers() :

    # This is same as constructor in C++
    # __init__ executes as many times object called 
    def __init__(self,cpu,ram):
        # print("In Init")
        self.cpu = cpu
        self.ram = ram

    def config(self):
        print("config is :",self.cpu,self.ram)
        return 0

# a = '8'
# print(type(a))

# creating a object 

com1 = computers('i5','16')
com2 = computers('i3','8')

# Another method to call object 
# computers.config(com1)
x = com1.config()
com2.config()
print(x)
# ****************************************************************************************************************************

# class computers:

#     def __init__(self):
#         self.name = 'Shubham'
#         self.age = 23

# c1 = computers()
# c2 = computers()

# c1.name = 'Kirti'

# print(c1.name)
# print(c2.name)