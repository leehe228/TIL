import numpy as np

a = [0.0 for k in range(19)]
a[4] = 1.0

print(np.argmax(a))
