#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

ca = lambda x:(2 - 0.5*x) / (2. + 0.5*x)
cb = lambda x:(0.5*x) / (2. + 0.5*x)

print '1', ca(1), cb(1)
print '10', ca(10), cb(10)

x = np.linspace(0,1000,10000)

plt.ion()
plt.plot(x, ca(x), 'r--')
plt.plot(x, cb(x), 'b-')
plt.show()
