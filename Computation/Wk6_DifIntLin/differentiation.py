import sympy as sy 
import numpy as np 
import matplotlib.pyplot as plt

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

def numerical_jacobian(f, h):

	











