import numpy as np

x = np.array([1,2,3,4,5,6])

x = x.reshape(3,2)

print(np.average(x, axis=1))