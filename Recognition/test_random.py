import random
import numpy as np
from matplotlib import pyplot as plt
random_nums_uniform = []
random_nums_radn = []
for i in range(10000):
    random_nums_uniform.append(np.random.normal(1,100))
    random_nums_radn.append(random.randint(1, 100))

plt.figure()
plt.hist(random_nums_uniform)
plt.figure()
plt.hist(random_nums_radn)
plt.show()