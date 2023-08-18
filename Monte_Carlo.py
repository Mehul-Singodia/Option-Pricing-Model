import math
from numpy import *
from time import time


random.seed(10000)
t0 = time()
# Parameters
S0 = 200.; K = 105.; T = 1.0; r = 0.05; sigma = 0.2
M = 25; dt = T / M; I = 250000

# Simulate I paths with M time steps
S = S0 * exp(cumsum((r - 0.5 * sigma ** 2) * dt
+ sigma * math.sqrt(dt)
* random.standard_normal((M + 1, I)), axis=0))

# sum instead of cumsum would also do

S[0] = S0
# Calculating the Monte Carlo estimator
C0 = math.exp(-r * T) * sum(maximum(S[-1] - K, 0)) / I
# Results output
tnp2 = time() - t0

print('The European Option Value is: ', C0)  
print('The Execution Time is: ',tnp2) 

#GRID
import matplotlib.pyplot as plt
plt.plot(S[:, :10])
plt.grid(True)
plt.xlabel('Step')
plt.ylabel('Index')
plt.show()

# Histogram
plt.rcParams["figure.figsize"] = (12,6)
plt.hist(S[-1], bins=100)
plt.grid(True)
plt.xlabel('Index')
plt.ylabel('Frequency')

import numpy as np
plt.rcParams["figure.figsize"] = (12,6)
plt.hist(np.maximum(S[-1] - K, 0), bins=100)
plt.grid(True)
plt.xlabel('option inner value')
plt.ylabel('frequency')
plt.ylim(0, 10000)

