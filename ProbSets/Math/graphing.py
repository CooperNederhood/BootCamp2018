import numpy as np 
import matplotlib.pyplot as plt 

def plot(fn, label):
	'''
	help function to quickly plot
	'''

	x_range = np.linspace(0, 6, 10000)

	if 'b=' not in label:
		plt.plot(x_range, fn(x_range), label=label, color='black')
	else:
		plt.plot(x_range, fn(x_range), label=label)


	
	plt.legend()

def plot_problem(constraints, file_name, name):

	plt.clf()
	for fn, label in constraints:
		plot(fn, label)

	plt.xlim((0,10))
	plt.ylim((0,10))
	t = "Feasible region for {}".format(name)
	plt.title(t)
	plt.savefig(file_name)


# QUESTION 8.1:
fn1 = lambda x: (2/3)*x + 4/3
fn2 = lambda x: x/6 - 1/6
fn3 = lambda x: 6 - x 

constraints = [(fn1, 'eq1'), (fn2, 'eq2'), (fn3, 'eq3')]
plot_problem(constraints, 'exercise8.1.png', '8.1')

# QUESTION 8.2 (i):
fn1 = lambda x: 5 - (1/3)* x
fn2 = lambda x: 6 - (2/3)* x 
fn3 = lambda x: x - 4

constraints = [(fn1, 'eq1'), (fn2, 'eq2'), (fn3, 'eq3')]
plot_problem(constraints, 'exercise8.2a.png', '8.2(i)')

# QUESTION 8.2 (ii):
fn1 = lambda x: 11 + x
fn2 = lambda x: 27 - x  
fn3 = lambda x: 14 - (2/5) * x 

constraints = [(fn1, 'eq1'), (fn2, 'eq2'), (fn3, 'eq3')]
plot_problem(constraints, 'exercise8.2b.png', '8.2(ii)')