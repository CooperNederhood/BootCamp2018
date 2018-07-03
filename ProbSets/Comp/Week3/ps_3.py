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





