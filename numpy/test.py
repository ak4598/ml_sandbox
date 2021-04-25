import numpy as np

x = np.linspace(1,10,100)
print(x)

x = np.ones((5,5,5))
print(x)
print(x.ravel())
print(x.reshape(25,5))
print(x.reshape(5,25))