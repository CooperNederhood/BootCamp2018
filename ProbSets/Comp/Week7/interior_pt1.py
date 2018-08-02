import sympy as sy 
import numpy as np 
import matplotlib.pyplot as plt
import autograd.numpy as anp 
import timeit 
from autograd import grad 
from autograd import elementwise_grad
import scipy.linalg as la 

#############
# Interior pt 1 - QUESTION 1:
#############
plt.clf()
section = "Interior pt 1"
print("########\n{} - QUESTION 1\n########\n".format(section))


def build_F(x, lamb, mu, A, b, c):

	n = x.size 
	m = mu.size 

	F = np.empty(2*n+m)
	F[0:m] = A.T@lamb + mu - c 
	F[m:m+n] = A@x - b 
	F[m+n:] = np.diag(mu)@x 

	return F 

test_x = np.array([1,2,3])
test_c = np.array([1,1,1])
test_A = np.identity(3)
test_b = np.array([3,3,3])
test_lamb = np.array([1,1,1])
test_mu = np.array([10,10,10])

F = build_F(test_x, test_lamb, test_mu, test_A, test_b, test_c)

	

#############
# Interior pt 1 - QUESTION 2:
#############
plt.clf()
section = "Interior pt 1"
print("########\n{} - QUESTION 2\n########\n".format(section))

def search_direction(F, x, lamb, mu, A, b, c, sigma=.1):

	n = x.size
	m = lamb.size 
	v = ( x.T @ mu ) / n 

	X = np.diag(x)
	M = np.diag(mu)

	Df_1 = np.hstack( (np.zeros((n,n)), A.T, np.identity(n)) )
	Df_2 = np.hstack( (A, np.zeros((m,m)), np.zeros((m,n))) )
	Df_3 = np.hstack( (M, np.zeros((n,m)), X))
	Df = np.vstack((Df_1, Df_2, Df_3))

	v_vec = np.concatenate((np.zeros(m+n), np.ones(n)*sigma*v))
	rhs = -F + v_vec
	L_U = la.lu_factor(Df)

	x_dir = la.lu_solve(L_U, rhs)

	return x_dir 

x_dir = search_direction(F, test_x, test_lamb, test_mu, test_A, test_b, test_c, .1)

#############
# Interior pt 1 - QUESTION 3:
#############
plt.clf()
section = "Interior pt 1"
print("########\n{} - QUESTION 3\n########\n".format(section))

def step_size(search_dirs, mu, x, lamb):

	n = x.size
	m = lamb.size 

	x_dir = search_dirs[0:n]
	lamb_dir = search_dirs[n:n+m]
	mu_dir = search_dirs[n+m:]

	assert x.size == x_dir.size 
	assert lamb_dir.size == lamb.size 

	alpha_max = np.min(1, np.min(-mu/mu_dir))
	delta_max = np.min(1, np.min(-x /x_dir ))

	alpha = np.min(1, 0.95*alpha_max)
	delta = np.min(1, 0.95*delta_max)

	return alpha, delta 

#############
# Interior pt 1 - QUESTION 4:
#############
plt.clf()
section = "Interior pt 1"
print("########\n{} - QUESTION 4\n########\n".format(section))

def starting_point(A, b, c):
	"""Calculate an initial guess to the solution of the linear program
	min c\trp x, Ax = b, x>=0.
	Reference: Nocedal and Wright, p. 410.
	"""
	# Calculate x, lam, mu of minimal norm satisfying both
	# the primal and dual constraints.
	B = la.inv(A @ A.T)
	x = A.T @ B @ b
	lam = B @ A @ c
	mu = c - (A.T @ lam)

	# Perturb x and s so they are nonnegative.
	dx = max((-3./2)*x.min(), 0)
	dmu = max((-3./2)*mu.min(), 0)
	x += dx*np.ones_like(x)
	mu += dmu*np.ones_like(mu)

	# Perturb x and mu so they are not too small and not too dissimilar.
	dx = .5*(x*mu).sum()/mu.sum()
	dmu = .5*(x*mu).sum()/x.sum()
	x += dx*np.ones_like(x)

	mu += dmu*np.ones_like(mu)

	return x, lam, mu

def interior_point(A, b, c, niter=20, tol=1e-16):

	x, lamb, mu = starting_point(A, b, c)
	n = x.size 

	v = ( x.T @ mu ) / n 
	i = 0

	while (v > tol) and (i < niter):

		F = build_F(x, lamb, mu, A, b, c)

		search_dirs = search_direction(F, x, lamb, mu, A, b, c)
		x_dir = search_dirs[0:n]
		lamb_dir = search_dirs[n:n+m]
		mu_dir = search_dirs[n+m:]

		alpha, delta = step_size(search_dirs, mu, x, lamb)

		x = x + delta*x_dir 
		lamb = lamb + alpha*lamb_dir
		mu = mu + alpha*mu_dir

		v = ( x.T @ mu ) / n 
		i = 0

	return x, c.T@x 

def randomLP(m, n):
	"""Generate a linear program min c\trp x s.t. Ax = b, x>=0.
	First generate m feasible constraints, then add
	slack variables to convert it into the above form.
	Inputs:
	m (int >= n): number of desired constraints.
	n (int): dimension of space in which to optimize.
	Outputs:
	A ((m,n+m) ndarray): Constraint matrix.
	b ((m,) ndarray): Constraint vector.
	c ((n+m,), ndarray): Objective function with m trailing 0s.
	x ((n,) ndarray): The first 'n' terms of the solution to the LP.
	"""
	A = np.random.random((m,n))*20 - 10
	A[A[:,-1]<0] *= -1
	x = np.random.random(n)*10
	b = np.zeros(m)
	b[:n] = A[:n,:] @ x
	b[n:] = A[n:,:] @ x + np.random.random(m-n)*10
	c = np.zeros(n+m)
	c[:n] = A[:n,:].sum(axis=0)/n
	A = np.hstack((A, np.eye(m)))

	return A, b, -c, x

m, n = 7, 5
A, b, c, x = randomLP(m, n)
point, value = interior_point(A, b, c)
#print(np.allclose(x, point[:n]))

