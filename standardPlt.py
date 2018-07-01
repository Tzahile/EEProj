import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math

x = np.linspace(0, 20, 100000)
signal = np.sin(13 * x)
noise = np.random.normal(0, 0.01, signal.shape)
harmony1 = np.sin(6.75 * x)
harmony2 = np.sin(10.439 * x)
plt.plot(x, harmony1+harmony2)

plt.plot(x, signal+harmony1+harmony2+noise)

plt.show()
