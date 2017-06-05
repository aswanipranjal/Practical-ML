from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# dtype specifies the datatype. The default probably is float64 but we are being explicit
# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# plt.scatter(xs, ys)
# plt.show()
def create_dataset(size, variance, step=2, correlation=False):
	val = 1
	ys = []
	for i in range(size):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step
	xs = [i for i in range(len(ys))]
	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def regress(xs, ys):
	m = (((mean(xs)*mean(ys)) - (mean(xs*ys))) / (((mean(xs))**2) - mean(xs**2)))
	b = mean(ys) - m*mean(xs)
	return m, b

def squared_error(ys_original, ys_line):
	return sum((ys_line - ys_original)**2)

def coefficient_of_determination(ys_original, ys_line):
	y_mean_line = [mean(ys_original) for y in ys_original]
	squared_error_regression_line = squared_error(ys_original, ys_line)
	squared_error_mean_line = squared_error(ys_original, y_mean_line)
	return 1 - (squared_error_regression_line/squared_error_mean_line)

xs, ys = create_dataset(40, 1, 2, 'pos')

m, b = regress(xs, ys)
print(m, b)

regression_line = [(m*x)+b for x in xs]

predict_x = 8
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show();
# to find how accurate the best fit line is
# Squared error : SE
# Coefficient of determination R^2 = 1 - (SE of y-hat)/(SE of mean)
# Basically 1 minus the ratio of how accurate the best fit line is compared to the mean line
