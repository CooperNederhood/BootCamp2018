import numpy as np 
import matplotlib.pyplot as plt 
import numpy.linalg as LA

#############
# QR decomposition - QUESTION 1:
#############
section = "QR decomposition"
print("########\n{} - QUESTION 1\n########\n".format(section))

def QR_decomp(A):
	'''
	Given m x n numpy matrix of rank A, 
	returns reduced QR decomp of A

	returns: tuple of Q, R matrices
	'''

	m, n = A.shape
	Q = np.copy(A)

	R = np.zeros( (n,n) )

	for i in range(n):
		R[i,i] = LA.norm(Q[:,i])
		Q[:,i] = Q[:,i] / R[i,i]

		for j in range(i+1, n):
			R[i,j] = Q[:,j] @ Q[:,i]
			Q[:,j] = Q[:,j] - R[i,j]*Q[:,i]

	return Q,R 

# Test function
A = np.random.random((6,4))
Q,R = QR_decomp(A)
print(A.shape, Q.shape, R.shape)

assert np.allclose(np.triu(R), R)
assert np.allclose(Q.T @ Q, np.identity(4))
assert np.allclose(Q @ R, A)


#############
# QR decomposition - QUESTION 2:
#############
section = "QR decomposition"
print("########\n{} - QUESTION 2\n########\n".format(section))

def det_A(A):
	'''
	Given n x n numpy matrix A,
	returns the determinant
	'''

	Q, R = QR_decomp(A)

	return np.prod(np.diag(R))

# Test function
A = np.random.random((6,6))
assert np.abs( np.abs(LA.det(A)) - det_A(A)) < 0.0001


#############
# QR decomposition - QUESTION 3:
#############
section = "QR decomposition"
print("########\n{} - QUESTION 3\n########\n".format(section))

def solve_linear(A, b):
	'''
	Given nxn matrix A and vector b, solves
	the system Ax = b
	'''

	Q, R = QR_decomp(A)

	n = R.shape[1]

	y = Q.T @ b 

	l = list(range(b.size))
	x = np.empty(b.size)

	for i in l[::-1]:
		print("Solving for x{}".format(i))

		x[i] = b[i]/A[i,i]

		#update, eliminate x[i] from other equations
		for j in range(0, i):
			b[j] -= A[j,i]*x[i]

	return x

# test function
A = np.array( [ [1,2], [0,1] ])
b = np.array([10, 8])

x = solve_linear(A, b)

#############
# QR decomposition - QUESTION 4:
#############
section = "QR decomposition"
print("########\n{} - QUESTION 4\n########\n".format(section))

def house_holder(A):
	'''
	Does the houseHolder QR decomp of 
	m x n matrix A
	'''

	m, n = A.shape
	R = np.copy(A)
	Q = np.identity(m)

	for k in range(n):
		u = np.copy(R[k:, k])
		print("k = {}".format(k))
		print("Shape u = {}".format(u.shape))
		print(R[k:, k:])

		u[0] = u[0] + np.sign(u[0]) * LA.norm(u)

		u = u / LA.norm(u)

		print(u.T@R[k:, k:])
		print()
		R[k:, k:] -= 2*np.outer(u, (u.T@R[k:, k:]))
		Q[k:, :] -= 2*np.outer(u, (u.T@Q[k:, :]))

	return Q.T, R 

# Check this QR decomp against our earlier decomp
# A = np.random.random((6,4))
# Q,R = house_holder(A)
# print(A.shape, Q.shape, R.shape)

# assert np.allclose(np.triu(R), R)
# assert np.allclose(Q.T @ Q, np.identity(4))
# assert np.allclose(Q @ R, A)

#############
# QR decomposition - QUESTION 5:
#############
section = "QR decomposition"
print("########\n{} - QUESTION 5\n########\n".format(section))

def hessenberg(A):
	'''
	Does the hessenberg algorithm
	'''

	m, n = A.shape 
	H = np.copy(A)
	Q = np.identity(m)
	for k in range(n-2):
		u = np.copy(H[k+1:, k])
		u[0] = u[0] + np.sign(u[0]) * LA.norm(u)

		u = u / LA.norm(u)

		H[k+1:, k:] = H[k+1:, k:]  - 2*np.outer(u, (u.T@H[k+1:, k:]))
		H[:, k+1:] = H[:, k+1:] - 2*(H[:,k+1:]@u)@u.T 
		Q[k+1:, :] = Q[k+1:, :] - 2*np.outer(u, u.T@Q[k+1:, :])

	return H, Q.T 

# A = np.random.random((8,8))
# H, Q = hessenberg(A)


#############
# LeastSquares_Eigenvalues - QUESTION 1:
#############
import scipy.linalg as la 

section = "LeastSquares_Eigenvalues"
print("########\n{} - QUESTION 1\n########\n".format(section))

def least_squares(A, b):

	Q, R = QR_decomp(A)
	y = Q.T@b
	x = la.solve_triangular(R, y)

	return x 



#############
# LeastSquares_Eigenvalues - QUESTION 2:
#############
section = "LeastSquares_Eigenvalues"
print("########\n{} - QUESTION 2\n########\n".format(section))

root = "../../../Computation/Wk3_Decomp/"
file = "housing.npy"

d = np.load(root+file)
ones = np.ones(d.shape[0])

A = np.column_stack((d[:,0], ones))
b = d[:,1]
ls_sol = least_squares(A, b)

fn = lambda x: ls_sol[1] + ls_sol[0]*x

plt.scatter(d[:,0], d[:,1])
x_vals = np.arange(d[:,0].min(), d[:,0].max())
plt.plot(x_vals, fn(x_vals), label='least squares line')
plt.legend()
plt.xlabel('Year from baseline')
plt.ylabel('House index')
plt.title('Housing index over time LS estimation')
plt.show()


