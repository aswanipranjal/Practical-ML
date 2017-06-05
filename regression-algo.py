from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# dtype specifies the datatype. The default probably is float64 but we are being explicit
xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# plt.scatter(xs, ys)
# plt.show()

def regress(xs, ys):
	m = (((mean(xs)*mean(ys)) - (mean(xs*ys))) / (((mean(xs))**2) - mean(xs**2)))
	b = mean(ys) - m*mean(xs)
	return m, b

m, b = regress(xs, ys)
print(m, b)