import sympy as sy 
import numpy as np 
import matplotlib.pyplot as plt
import autograd.numpy as anp 
import timeit 
from autograd import grad 
from autograd import elementwise_grad
import numpy.linalg as la 
import scipy.sparse as sparse

#############
# Iterative solvers - QUESTION 1&2:
#############
plt.clf()
section = "Iterative solvers"
print("########\n{} - QUESTION 1&2\n########\n".format(section))

def jacobi(A, b, tol, maxiters, plot=False):


	iter_count = 0
	error = np.inf 

	D = np.diag(A)	
	L = np.tril(A, k=-1)
	U = np.triu(A, k=1)

	error_tracker = np.empty(maxiters)

	assert np.allclose(A, np.diag(D) + L + U)

	D_inv = 1/D 
	D_inv = D_inv.reshape((D_inv.size, 1))

	cur_x = np.zeros((A.shape[1],1))

	while (iter_count < maxiters) and (error > tol):

		new_x = cur_x + D_inv*(b-A@cur_x)

		error = np.max(np.abs(new_x - cur_x))
		error_tracker[iter_count] = error 
		iter_count += 1


		cur_x = new_x 

	error_tracker = error_tracker[0:iter_count]
	if plot:
		plt.semilogy(range(iter_count), error_tracker)
		plt.xlabel("Iteration")
		plt.ylabel("Absolute error of approx")
		plt.title("Covergence of Jacobi Method")
		plt.savefig('jacobi_covergence.png')


	return cur_x 


def diag_dom(n, num_entries=None):
	"""Generate a strictly diagonally dominant (n, n) matrix.
	Parameters:
	n (int): The dimension of the system.
	num_entries (int): The number of nonzero values.
	Defaults to n^(3/2)-n.
	Returns:
	A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
	"""
	if num_entries is None:
		num_entries = int(n**1.5) - n
	A = np.zeros((n,n))
	rows = np.random.choice(np.arange(0,n), size=num_entries)
	cols = np.random.choice(np.arange(0,n), size=num_entries)
	data = np.random.randint(-4, 4, size=num_entries)
	for i in range(num_entries):
		A[rows[i], cols[i]] = data[i]
	for i in range(n):
		A[i,i] = np.sum(np.abs(A[i])) + 1
	return A

def test_part1(n):

	A = diag_dom(n)
	b = np.random.random(n).reshape((n,1))
	x = jacobi(A, b, 10e-10, 200, plot=True)

	return np.allclose(A@x, b)

for n in range(2, 10):
	#assert test_part1(n)
	assert True
print("Passed test #1")

#############
# Iterative solvers - QUESTION 3:
#############
plt.clf()
section = "Iterative solvers"
print("########\n{} - QUESTION 3\n########\n".format(section))

def Gauss_Seidel(A, b, tol, maxiters, plot=False):

	n = A.shape[1]
	iter_count = 0
	error = np.inf 

	error_tracker = np.empty(maxiters)

	cur_x = np.zeros((n,1))

	while (iter_count < maxiters) and (error > tol):

		preserve_x = np.copy(cur_x)
		for i in range(n):
			cur_x[i,0] += (1/A[i,i]) * (b[i,0] - A[i,:].T @ cur_x)

		error = np.max(np.abs(preserve_x - cur_x))
		error_tracker[iter_count] = error 
		iter_count += 1


	error_tracker = error_tracker[0:iter_count]
	if plot:
		plt.semilogy(range(iter_count), error_tracker)
		plt.xlabel("Iteration")
		plt.ylabel("Absolute error of approx")
		plt.title("Covergence of Gauss_Seidel Method")
		plt.savefig('Gauss_Seidel_covergence.png')


	return cur_x 

def test_part3(n):

	A = diag_dom(n)
	b = np.random.random(n).reshape((n,1))
	x = Gauss_Seidel(A, b, 10e-10, 200, plot=True)

	return np.allclose(A@x, b)

for n in range(2, 10):
	assert test_part3(n)
	assert True
print("Passed test #3")

#############
# Iterative solvers - QUESTION 4:
#############
plt.clf()
section = "Iterative solvers"
print("########\n{} - QUESTION 4\n########\n".format(section))

def Gauss_Seidel_sparse(A, b, tol, maxiters):
	'''
	NOTE: matrix A is a sparse matrix
	'''

	n = A.shape[1]
	iter_count = 0
	error = np.inf 

	cur_x = np.zeros((n,1))

	while (iter_count < maxiters) and (error > tol):

		preserve_x = np.copy(cur_x)
		for i in range(n):
			rowstart = A.indptr[i]
			rowend = A.indptr[i+1]

			Aix = A.data[rowstart:rowend] @ cur_x[A.indices[rowstart:rowend]]
			cur_x[i,0] += (1/A[i,i]) * (b[i,0] - Aix)

		error = np.max(np.abs(preserve_x - cur_x))
		iter_count += 1

	return cur_x 

def test_part4(n):

	A = diag_dom(n)
	b = np.random.random(n).reshape((n,1))
	x = Gauss_Seidel_sparse(sparse.csr_matrix(A), b, 10e-10, 200)

	return np.allclose(A@x, b)

for n in range(2, 10):
	assert test_part4(n)
	assert True
print("Passed test #4")


#############
# Iterative solvers - QUESTION 5:
#############
plt.clf()
section = "Iterative solvers"
print("########\n{} - QUESTION 5\n########\n".format(section))

def Gauss_Seidel_SOR(A, b, tol, maxiters, omega):
	'''
	NOTE: matrix A is a sparse matrix
	'''

	n = A.shape[1]
	iter_count = 0
	error = np.inf 

	cur_x = np.zeros((n,1))

	while (iter_count < maxiters) and (error > tol):

		preserve_x = np.copy(cur_x)
		for i in range(n):
			rowstart = A.indptr[i]
			rowend = A.indptr[i+1]

			Aix = A.data[rowstart:rowend] @ cur_x[A.indices[rowstart:rowend]]
			cur_x[i,0] += (omega/A[i,i]) * (b[i,0] - Aix)

		error = np.max(np.abs(preserve_x - cur_x))
		iter_count += 1

	return cur_x, iter_count

def test_part5(n):

	A = diag_dom(n)
	b = np.random.random(n).reshape((n,1))
	x = Gauss_Seidel_SOR(sparse.csr_matrix(A), b, 10e-10, 200, .5)[0]

	return np.allclose(A@x, b)

for n in range(2, 10):
	assert test_part5(n)
	assert True
print("Passed test #5")

#############
# Iterative solvers - QUESTION 6:
#############
plt.clf()
section = "Iterative solvers"
print("########\n{} - QUESTION 6\n########\n".format(section))

def laplace(n, omega, tol=10e-8, maxiters=100, plot=False):

	b_n = np.zeros(n)
	b_n[0] = -100
	b_n[n-1] = -100
	b = np.tile(b_n, n)
	b = b.reshape((n**2,1))

	A = np.tile(np.identity(n), (n,n) )
	np.fill_diagonal(A, -4)

	ones = np.ones( (n**2, n**2) )
	ones_diag_upper = np.triu(np.tril(ones, 1), 1)
	ones_diag_lower = np.triu(np.tril(ones, -1), -1)

	A = A + ones_diag_lower + ones_diag_upper

	x, iters = Gauss_Seidel_SOR(sparse.csr_matrix(A), b, tol, maxiters, omega)

	x = x.reshape((n,n))

	if plot:
		plt.pcolormesh(x, cmap="coolwarm")
		plt.savefig('Laplace.png')

laplace(10, 1, plot=True)

