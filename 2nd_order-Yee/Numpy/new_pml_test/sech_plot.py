#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10000)

plt.plot(np.exp(-x)/np.cosh(x))
#plt.plot(np.tanh(x))
plt.show()
