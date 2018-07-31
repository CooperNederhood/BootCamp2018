import sympy as sy 
import numpy as np 
import matplotlib.pyplot as plt
import autograd.numpy as anp 
import timeit 
from autograd import grad 
from autograd import elementwise_grad
import numpy.linalg as la 

# Exercise 9.6
def quad_opt(x0, b, Q, tol):
	'''
	Optimize quad function of form:
		f(x) = (1/2)xQx - bx + c
	'''
	n = x0.shape[0]
	Df = Q@x0 - b 
	i = 0

	while (la.norm(Df) > tol):

		alpha = (Df.T @ Df) / (Df.T @ Q @ Df)
		x0 = x0 - alpha * Df 
		Df = Q@x0 - b
		i += 1

	return x0 

Q = np.identity(4)
x0 = np.full((4,1), 0)
b = np.full((4,1), 1)
tol = .001

x1 = quad_opt(x0, b, Q, tol)
assert np.allclose(Q@x1, b)
print("Quad opt passes test function!")
print()

# Exercise 9.7 
def calc_Df(fn, x, Rerr):
	'''
	Calculates Df at point x. 
	Assumes x is 1-D np array of len (n) and 
		fn accepts np array as input
	'''

	h = 2*np.sqrt(Rerr)

	Df = np.empty(x.size)
	I = np.identity(x.size)

	for i in range(x.size):
		e_i = I[:,i]
		Df[i] = (fn(x+e_i*h) - fn(x)) / (h)

	return Df 

# Show calc_Df function works
def ex_f(x):
	x_0 = x[0]
	x_1 = x[1]
	x_2 = x[2]

	f = lambda x_0, x_1, x_2: 2*x_0**2 + x_1**2 + x_2**4

	return f(x_0, x_1, x_2)

test_x = np.array([1,1,1])
print("Deriv estimate at {} is:\n{}".format(test_x, calc_Df(ex_f, test_x, 10e-6)))

def secant(x0, x1, f, tol=1e-8, max_iters=100, iters = 0):
    x2 = x1 - f(x1)*((x1 - x0)/(f(x1)-f(x0)))
    h = np.abs(x2-x1)
    if h < tol or iters > max_iters:
        return x2
    else:
        return secant(x1, x2, f, tol, max_iters, iters+1)

# Exercise 9.8
def steepest_descent(fn, x, epsilon, Rerr, max_iter_check=100):
	'''
	I was having issues finding alpha through the secant method
	so I am using the alpha/2 iteration :(
	'''

	Df = calc_Df(fn, x, Rerr)
	min_x = np.copy(x)

	alpha_tol = 10e-6
	i = 0

	while (la.norm(Df) > epsilon) and i < max_iter_check:
		print("At iteration {}".format(i))
		print("At point {}".format(min_x))
		print("Df is {}".format(Df))
		print("Df norm = {}".format(la.norm(Df)))

		alpha = 1
		alpha_count = 0
		while fn(min_x - alpha*Df) > fn(min_x) and alpha_count < 100:
			alpha = alpha / 2
			alpha_count += 1

		print("Alpha = ", alpha)
		print()

		min_x = min_x - alpha*Df 
		Df = calc_Df(fn, min_x, Rerr)
		i += 1


	return min_x

check_min = steepest_descent(ex_f, np.array([1,1,1]), epsilon=10e-4, Rerr=10e-6)
print("estimate of min is at: \n", check_min)
print()

# Exercise 9.9

def rosenbrock(x_array):

	x = x_array[0]
	y = x_array[1]

	return 100*(y-x**2)**2 + (1-x)**2

min_rosen = steepest_descent(rosenbrock, np.array([-2, 2]), epsilon=10e-4, Rerr=10e-2)
print("Estimate min of Rosenbrock function: ", min_rosen)


