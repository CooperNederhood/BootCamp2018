import sympy as sy 
import numpy as np 
import matplotlib.pyplot as plt
import autograd.numpy as anp 
import timeit 
from autograd import grad 
from autograd import elementwise_grad
import scipy.linalg as la 
import scipy.optimize as opt 

#############
# Newton & Quasi Newton - QUESTION 1:
#############
plt.clf()
section = "Newton & Quasi Newton"
print("########\n{} - QUESTION 1\n########\n".format(section))


def  newton_method(Df, D2f, x0, maxiter=100, tol=10e-3):

	error = np.max(np.abs(Df(x0)))
	i = 0
	x = x0

	while (error > tol) and (i < maxiter):

		Df_x = Df(x)
		D2f_x = D2f(x) 

		z_k = la.solve(D2f_x, Df_x.T)

		x = x - z_k 

		error = np.max(np.abs(Df(x)))
		i += 1

	return x

f = opt.rosen 
df = opt.rosen_der 
d2f = opt.rosen_hess 
opt.fmin_bfgs(f=f, x0=[-2,2], fprime=df, maxiter=50)

solution = newton_method(df, d2f, np.array([-2.,2.]))
print(solution)



