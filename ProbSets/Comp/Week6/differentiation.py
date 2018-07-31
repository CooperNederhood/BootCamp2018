import sympy as sy 
import numpy as np 
import matplotlib.pyplot as plt
import autograd.numpy as anp 
import timeit 
from autograd import grad 
from autograd import elementwise_grad

#############
# Differentiation - QUESTION 1:
#############
plt.clf()
section = "Differentiation"
print("########\n{} - QUESTION 1\n########\n".format(section))

def sym_diff():

	x = sy.symbols('x')
	g = sy.diff( (sy.sin(x) + 1)**(sy.sin(sy.cos(x))), x)

	return sy.lambdify(x, g)

x_vals = np.linspace(-np.pi, np.pi, 200)
f = lambda x: (np.sin(x) + 1)**(np.sin(np.cos(x)))
f_prime = sym_diff()

plt.plot(x_vals, f(x_vals), label = 'fn')
plt.plot(x_vals, f_prime(x_vals), label = 'fn_deriv')
plt.title('Fn with deriv')
plt.legend()
ax = plt.gca()
ax.spines["bottom"].set_position("zero")
plt.savefig('Differentiation_q1.png')


#############
# Differentiation - QUESTION 2:
#############
section = "Differentiation"
print("########\n{} - QUESTION 2\n########\n".format(section))

def forward_1(fn, x_vals, h):
	return (fn(x_vals+h) -fn(x_vals))/h

def forward_2(fn, x_vals, h):
	return (-3*fn(x_vals) + 4*fn(x_vals+h)-fn(x_vals+2*h))/(2*h)

def backward_1(fn, x_vals, h):
	return (fn(x_vals) - fn(x_vals-h))/h 

def backward_2(fn, x_vals, h):
	return (3*fn(x_vals) - 4*fn(x_vals+h)+fn(x_vals+2*h))/(2*h)

def centered_1(fn, x_vals, h):
	return (fn(x_vals+h) - fn(x_vals-h))/(2*h)

def centered_2(fn, x_vals, h):
	return (fn(x_vals-2*h) - 8*fn(x_vals-h) + 8*fn(x_vals+h) - fn(x_vals+2*h))/(12*h)

plt.plot(x_vals, f(x_vals), label = 'fn')
plt.plot(x_vals, f_prime(x_vals), label = 'fn_deriv')
plt.title('Fn with deriv')
ax = plt.gca()
ax.spines["bottom"].set_position("zero")

plt.plot(x_vals, forward_1(f, x_vals, 0.00000001), label='forward_1')
plt.plot(x_vals, forward_2(f, x_vals, 0.00000001), label='forward_2')
plt.plot(x_vals, backward_1(f, x_vals, 0.00000001), label='backward_1')
plt.plot(x_vals, backward_2(f, x_vals, 0.00000001), label='backward_2')
plt.plot(x_vals, centered_1(f, x_vals, 0.00000001), label='centered_1')
plt.plot(x_vals, centered_2(f, x_vals, 0.00000001), label='centered_2')
plt.legend()
plt.savefig('Differentiation_q2.png')


#############
# Differentiation - QUESTION 3:
#############
section = "Differentiation"
print("########\n{} - QUESTION 3\n########\n".format(section))

plt.clf()

def build(diff_fn, fn, x_vals, h):

	return diff_fn(fn,x_vals, h)	

def plot_convergence(pt):

	f = lambda x: (np.sin(x) + 1)**(np.sin(np.cos(x)))

	h_vals = np.logspace(-3, 0, 10)
	y_vals = np.empty((10,6))

	labels = ['1 forward', '2 forward', '1 backward', '2 backward', '1 centered', '2 centered']

	for i in range(10):
		t = build(forward_1, f, pt, h_vals[i] )
		print(t)
		y_vals[i][0] = build(forward_1, f, pt, h_vals[i] )
		y_vals[i][1] = build(forward_2, f, pt, h_vals[i] )
		y_vals[i][2] = build(backward_1, f, pt, h_vals[i] )
		y_vals[i][3] = build(backward_2, f, pt, h_vals[i] )
		y_vals[i][4] = build(centered_1, f, pt, h_vals[i] )
		y_vals[i][5] = build(centered_2, f, pt, h_vals[i] )

	for i in range(6):
		l = "Order " + labels[i]
		plt.loglog(h_vals, y_vals[:,i], label=l, marker="o")


	plt.legend()
	plt.ylabel("Absolute error")
	plt.xlabel("h")
	plt.title("Convergence through time")
	plt.savefig("Differentiation_q3.png")

plot_convergence(1)

#############
# Differentiation - QUESTION 4:
#############
section = "Differentiation"
print("########\n{} - QUESTION 4\n########\n".format(section))

plt.clf()
d = np.load('plane.npy')

ALPHA = 1
BETA = 2
d_rad = np.copy(d)
d_rad[:,ALPHA] = np.deg2rad(d[:,ALPHA])
d_rad[:,BETA] = np.deg2rad(d[:,BETA])
x_t = 500 * np.tan(d_rad[:,BETA]) / (np.tan(d_rad[:,BETA]) - np.tan(d_rad[:,ALPHA]))
y_t = 500 * np.tan(d_rad[:,BETA])*np.tan(d_rad[:,ALPHA]) / (np.tan(d_rad[:,BETA]) - np.tan(d_rad[:,ALPHA]))

x_t_p1 = np.append(x_t[1:], [0])
x_t_m1 = np.append([0], x_t[:-1])
y_t_p1 = np.append(y_t[1:], [0])
y_t_m1 = np.append([0], y_t[:-1])

x_prime = np.empty((d.shape[0]))
x_prime[1:-1] = (1/2)*(x_t_p1[1:-1] - x_t_m1[1:-1])
x_prime[0] = (1/2)*(x_t_p1[0] - x_t[0])
x_prime[-1] = (1/2)*(-x_t_m1[-1] + x_t[-1])

y_prime = np.empty((d.shape[0]))
y_prime[1:-1] = (1/2)*(y_t_p1[1:-1] - y_t_m1[1:-1])
y_prime[0] = (1/2)*(y_t_p1[0] - y_t[0])
y_prime[-1] = (1/2)*(-y_t_m1[-1] + y_t[-1])

speed = np.sqrt(y_prime**2 + x_prime**2)
print("Speed vector is:\n", speed)


#############
# Differentiation - QUESTION 5:
#############
section = "Differentiation"
print("########\n{} - QUESTION 5\n########\n".format(section))

def numerical_jacobian(f, pt, h):

	''' Assumes function f is passed in as some list of functions
	of length m'''
	
	m = len(f)
	n = pt.shape[0]
	I = np.identity(n)
	J = np.empty((m, n))
	for i in range(m):
		fn = f[i]
		for j in range(n):
			J[i,j] = (fn(pt + h*I[:,j]) - fn(pt - h*I[:,j])) / (2*h)

	return J 

fn1 = lambda x, y: x**2
fn2 = lambda x, y: x**3 - y 
fn = [fn1, fn2]
pt = np.array([1,1])

# fn_J = numerical_jacobian(fn, pt, .00000001)
# print(fn_J)



#############
# Differentiation - QUESTION 6:
#############
section = "Differentiation"
print("########\n{} - QUESTION 6\n########\n".format(section))

def cheb_poly(x_array, n):

	if n == 0:
		return anp.ones_like(x_array)

	if n == 1:
		return x_array 

	else:
		return 2*x_array*cheb_poly(x_array, n-1) - cheb_poly(x_array, n-2)

cheb_poly_deriv = elementwise_grad(cheb_poly)
x_vals = anp.linspace(-1, 1, 1000)
plt.clf()
for n in range(5):
	plt.plot(x_vals, cheb_poly(x_vals, n), label="n={} Cheb".format(n))
	plt.plot(x_vals, cheb_poly_deriv(x_vals, n), label="n={} Cheb deriv".format(n))

plt.legend()
plt.title("Cheb poly's with derivs")
plt.savefig("Differentiation_q6.png")


#############
# Differentiation - QUESTION 7:
#############
section = "Differentiation"
print("########\n{} - QUESTION 7\n########\n".format(section))
plt.clf()

def experiment(N):

	f = lambda x: (anp.sin(x) + 1)**(anp.sin(anp.cos(x)))
	x_vals = np.random.uniform(0, 2*np.pi, N)

	data_2 = np.empty((N,2))
	data_3 = np.empty((N,2))
	data_4 = np.empty((N,2))

	for i in range(N):
		x_0 = x_vals[i]

		# (2):
		start = timeit.default_timer()
		f_prime = sym_diff()
		f_prime_2 = f_prime(x_0)
		error_2 = 1e-18
		time_2 = timeit.default_timer() - start
		data_2[i,0] = error_2
		data_2[i,1] = time_2

		# (3):
		start = timeit.default_timer()
		f_prime_3 = centered_2(f, x_0, 0.00000001)
		error_3 = np.abs(f_prime_2 - f_prime_3)
		time_3 = timeit.default_timer() - start
		data_3[i,0] = error_3
		data_3[i,1] = time_3

		# (4):
		start = timeit.default_timer()
		f_grad = grad(f)
		f_prime_4 = f_grad(x_0)
		error_4 = np.abs(f_prime_4 - f_prime_2)
		time_4 = timeit.default_timer() - start
		data_4[i,0] = error_4
		data_4[i,1] = time_4

	return data_2, data_3, data_4 

data_2, data_3, data_4  = experiment(200)
plt.loglog(data_2[:,1], data_2[:,0], linestyle="None", alpha=0.4, label='SymPy', marker='o' )
plt.loglog(data_3[:,1], data_3[:,0], linestyle="None", alpha=0.4, label='Diff quot', marker='o' )
plt.loglog(data_4[:,1], data_4[:,0], linestyle="None", alpha=0.4, label='Auto', marker='o' )
plt.legend()
plt.xlabel('Computation time (seconds)')
plt.ylabel('Absolute error')
plt.savefig("Differentiation_q7.png")

