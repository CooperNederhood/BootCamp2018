import numpy as np 
import matplotlib.pyplot as plt 
import numpy.linalg as LA
from scipy import linalg as la 

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
#plt.show()
plt.clf()


#############
# LeastSquares_Eigenvalues - QUESTION 3:
#############
section = "LeastSquares_Eigenvalues"
print("########\n{} - QUESTION 3\n########\n".format(section))

def ls_poly(X, b, degree):
	'''
	Fit polynomials of degree

	Returns coefficients
	'''

	A = np.vander(X, N=degree)
	c = la.lstsq(A, b)[0]

	return np.poly1d(c) 

def ls_all_poly(X, b):
	'''
	Fits polynomials of degree 3, 6, 9
	'''

	degrees = [3, 6, 9]

	x_vals = np.linspace(np.min(X), np.min(X), 10000)
	plt.scatter(X, b)

	for d in degrees:
		fn = ls_poly(X, b, d)
		print(fn)
		y_vals = fn(x_vals)
		print(y_vals)
		print()

		plt.plot(x_vals, fn(x_vals), label='degree {}'.format(d))

	plt.legend()
	#plt.show()

ls_all_poly(d[:,0], d[:,1])


#############
# LeastSquares_Eigenvalues - QUESTION 5:
#############
section = "LeastSquares_Eigenvalues"
print("########\n{} - QUESTION 5\n########\n".format(section))


def power_method(A, N, tol):
	'''
	Power method on matrix A for max iters of N, tolerance
	'''

	m, n = A.shape 
	x = np.random.rand(n)
	x = x / la.norm(x)

	count = 0
	tol = np.inf

	while (count < N) and (tol > N):
		x = A @ x 

		x = x / la.norm(x)

	return x.T @ A @ x, x 


A = np.random.random( (10,10) )
eigs, vecs = la.eig(A) 

loc = np.argmax(eigs)
lamb, x = eigs[loc], vecs[:, loc]

assert np.allclose(A @ x, lamb * x)
print("Passes assertion check")



#############
# LeastSquares_Eigenvalues - QUESTION 6:
#############
section = "LeastSquares_Eigenvalues"
print("########\n{} - QUESTION 6\n########\n".format(section))

def qr_algor(A, N, tol):

	m, n = A.shape 
	S = la.hessenberg(A)

	for k in range(N):
		Q, R = la.qr(S)

		S = R @ Q 

	eigs = []
	i = 0 


#############
# SVD_ImageCompression - QUESTION 1:
#############
section = "SVD_ImageCompression"
print("########\n{} - QUESTION 1\n########\n".format(section))

def compact_svd(A, tol):
	'''
	Compact SVD of matrix A
	'''

	A_H = A.conj().T 
	eigs, vecs = la.eig(A_H @ A)
	sing_vals = np.sqrt(eigs)

	order = np.argsort(sing_vals)[::-1]
	sing_vals = sing_vals[order]
	vecs = vecs[:, order]

	# restrict to those larger than tol
	r = sing_vals[ sing_vals > tol ].size 
	sigma_1 = sing_vals[:r]
	v_1 = vecs[:r]
	u_1 = A @ v_1 / sigma_1 

	return u_1, sigma_1, v_1.conj().T 

test0 = np.random.random((4,3))
test0_result = la.svd(test0, full_matrices=False)
my_result = compact_svd(test0, 0.00001)
for i in range(3):
	assert np.allclose(np.abs(test0_result[i]), np.abs(my_result[i]))
print("Passes SVD test")


#############
# SVD_ImageCompression - QUESTION 2:
#############
section = "SVD_ImageCompression"
print("########\n{} - QUESTION 2\n########\n".format(section))
plt.clf()

def vis_svd(A):
	'''
	Visualize SVD
	'''

	theta = np.random.uniform(low=0, high=2*np.pi, size=200)
	x = np.cos(theta)
	y = np.sin(theta)
	S = np.vstack((x,y))
	E = np.array([[1,0,0], [0, 0, 1]])

	U, sigma, V_H = la.svd(A, full_matrices=False)
	sigma = np.diag(sigma)

	fig, axes = plt.subplots(2,2)

	print(S.shape)
	axes[0][0].scatter(S[0,:], S[1,:])
	axes[0][0].plot(E[0,:], E[1,:])
	
	S = V_H @ S 
	E = V_H @ E 

	axes[0][1].scatter(S[0,:], S[1,:])
	axes[0][1].plot(E[0,:], E[1,:])

	S = sigma @ S 
	E = sigma @ E 

	print(S.shape)
	axes[1][0].scatter(S[0,:], S[1,:])
	axes[1][0].plot(E[0,:], E[1,:])

	S = U @ S 
	E = U @ E 

	axes[1][1].scatter(S[0,:], S[1,:])
	axes[1][1].plot(E[0,:], E[1,:])

	plt.show()

A = np.array([ [3, 1], [1, 3]])
#vis_svd(A)

#############
# SVD_ImageCompression - QUESTION 3:
#############
section = "SVD_ImageCompression"
print("########\n{} - QUESTION 3\n########\n".format(section))
plt.clf()

def trun_svd(A, s):
	'''
	Computes the truncated SVD to size s
	'''

	U, sigma, V_H = la.svd(A, full_matrices=False)
	order = np.argsort(sigma)[::-1]

	U_hat = U[:, order][:, 0:s]
	sigma_hat = sigma[order][0:s]
	V_H_hat = V_H[order, :][0:s, :]

	storage = U_hat.size + sigma_hat.size + V_H_hat.size 

	return U_hat, sigma_hat, V_H_hat, storage 

A = np.random.random( (10,8) )
s = 4

x = trun_svd(A, s)
for i in x:
	print(i)


#############
# SVD_ImageCompression - QUESTION 4:
#############
section = "SVD_ImageCompression"
print("########\n{} - QUESTION 4\n########\n".format(section))
plt.clf()

def trun_svd_eps(A, eps):
	'''
	Computes the truncated SVD to size s
	'''

	U, sigma, V_H = la.svd(A, full_matrices=False)
	order = np.argsort(sigma)[::-1]

	# just construct s as the index of the first sing_val NOT above eps threshold
	sigma_temp = sigma[ order ]
	s = sigma[sigma_temp > eps].size 

	U_hat = U[:, order][:, 0:s]
	sigma_hat = sigma[order][0:s]
	V_H_hat = V_H[order, :][0:s, :]

	storage = U_hat.size + sigma_hat.size + V_H_hat.size 

	return U_hat, sigma_hat, V_H_hat, storage 

A = np.random.random( (10,8) )
eps = 10

x = trun_svd(A, s)
for i in x:
	print(i)


#############
# SVD_ImageCompression - QUESTION 5:
#############
section = "SVD_ImageCompression"
print("########\n{} - QUESTION 5\n########\n".format(section))
plt.clf()

def process_image(file, s):
	'''
	Plots original file image and s-rank representation
	'''

	img = plt.imread(file) / 255
	print(img.shape)
	new_img = np.copy(img)

	dim = img.ndim
	if dim == 2:
		U_hat, sigma_hat, V_H_hat, size = trun_svd(img, s)
		new_img = U_hat @ np.diag(sigma_hat) @ V_H_hat

	else:
		assert dim == 3
		for h in range(new_img.shape[2]):
			U_hat, sigma_hat, V_H_hat, size = trun_svd(img[:,:,h], s)

			new_img[:,:,h] = np.clip(U_hat @ np.diag(sigma_hat) @ V_H_hat, 0, 1)


	fig, axes = plt.subplots(2)
	axes[0].imshow(img)
	axes[1].imshow(new_img)

	plt.show()


h_file = "hubble.jpg"

process_image(root+h_file, 4)

