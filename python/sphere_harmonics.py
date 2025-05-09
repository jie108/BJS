"""
 Purpose: Generate the matrices to construct design matrix. (X = SH * R)
     
 2018: Created (By Jilei Yang)
 30th Jan 2020: Added the description of functions (By Seungyong Hwang)
"""
## The built-in pakcages
import numpy as np
from scipy.special import lpmn
from math import factorial

## Defined functions for the tasks with Spherical Harmonic basis.
from sphere_mesh import spmesh


# Description: Evaluate spherical harmonic Phi_{lm} at location (theta, phi), where l is the level and m is the phase
# Input: (l: Level, m: Phase, (theta, phi): spherical coordinates)
# Output: a complex value of evaluation
def spharmonic_eval(l, m, theta, phi):

	sign_m = np.sign(m)
	m = np.abs(m)

	C = np.sqrt((2*l+1)/(4*np.pi)*factorial(l-m)/factorial(l+m))
	P = lpmn(m, l, np.cos(theta))[0][m, l]
	Y = C*P*np.exp(1j*m*phi)

	if sign_m < 0:
		Y = (-1)**m*np.conjugate(Y)

	return Y


# Description: evaluate real symmetrized spherical harmonics up to lmax on the grid specified by spherical coordinates (theta, phi)
# \tilde{Phi}_{lm}=(-1)^m\sqrt{2}Re(Phi_{lm}) if m<0, Phi_{l0} if m=0, (-1)^m\sqrt{2}Img(Phi_{lm}) if m>0
# Input: (theta, phi): a set of spherical coordinates (can be generated from the function "spmesh" - equi_angle grid
#        lmax : the maximum level of basis. (even number)
# Output: Spherical harmonic design matrix (size: number of grid points * number of symmetrized SH basis)
def spharmonic(theta, phi, lmax):

	L = int((lmax+1)*(lmax+2)/2)  # number of symmetrized SH basis
	SH_matrix = np.zeros((len(theta), L))

	for i in range(len(theta)):  # vertex
		for l in range(0, lmax+1, 2):  # even level SH
			for m in range(-l, l+1):  # SH phase
				SH_index = int(l*(l+1)/2+m)
				Y_lm = spharmonic_eval(l, m, theta[i], phi[i])
				if m < 0:
					SH_matrix[i, SH_index] = (-1)**m*np.sqrt(2)*Y_lm.real
				elif m == 0:
					SH_matrix[i, SH_index] = Y_lm.real
				else:
					SH_matrix[i, SH_index] = (-1)**m*np.sqrt(2)*Y_lm.imag

	return SH_matrix


# Description: generate response function for a single fiber. response function is along (theta0, phi0) and is evaluated at (theta, phi)
# Input: b: b-value (1 = 1,000mm^2/s, 3 = 3,000mm^2/s)
#       ratio: the ratio between the 1st eigenvalue and the average of the 2nd and 3rd eigenvalues.
#       (theta0, phi0): a direction of a single fiber
#       (theta, phi): a set of spherical harmonic coordiates
# Output: a value of response function
#def myresponse(b, ratio, theta0, phi0, theta, phi):

#	D = np.diag(np.array([1.0/ratio, 1.0/ratio, 1]))
#	u = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

	# rotation matrix around y-axis
#	T_theta = np.array([[np.cos(theta0), 0, -np.sin(theta0)], [0, 1, 0], [np.sin(theta0), 0, np.cos(theta0)]])
	# rotation matrix around z-axis
#	T_phi = np.array([[np.cos(phi0), np.sin(phi0), 0], [-np.sin(phi0), np.cos(phi0), 0], [0, 0, 1]])
	# rotation matrix T0 satisfies T0*[sin(theta0)*cos(phi0), sin(theta0)*sin(phi0), cos(theta0)]=[0, 0, 1]
#	T0 = T_theta.dot(T_phi)

	# plug in T0*u into single tensor model along z-axis (i.e., D is a diagonal matrix)
#	y = np.exp(-b*u.T.dot(T0.T).dot(D).dot(T0).dot(u))

#	return y


# Description: generate response function for a single fiber. response function is along (theta0, phi0) and is evaluated at (theta, phi)
# Input: b: b-value (1,000mm^2/s ~ 3,000mm^2/s)
#		lambda1 : first eigenvalue (default 1,000^-3)
#       ratio: the ratio between the 1st eigenvalue and the average of the 2nd and 3rd eigenvalues.
#       (theta0, phi0): a direction of a single fiber
#       (theta, phi): a set of spherical harmonic coordiates
# Output: a value of response function
def single_tensor(b, ratio, theta0, phi0, theta, phi, lambda1 = 0.001):

	D = np.diag(np.array([lambda1/ratio, lambda1/ratio, lambda1]))
	u = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

	# rotation matrix around y-axis
	T_theta = np.array([[np.cos(theta0), 0, -np.sin(theta0)], [0, 1, 0], [np.sin(theta0), 0, np.cos(theta0)]])
	# rotation matrix around z-axis
	T_phi = np.array([[np.cos(phi0), np.sin(phi0), 0], [-np.sin(phi0), np.cos(phi0), 0], [0, 0, 1]])
	# rotation matrix T0 satisfies T0*[sin(theta0)*cos(phi0), sin(theta0)*sin(phi0), cos(theta0)]=[0, 0, 1]
	T0 = T_theta.dot(T_phi)

	# plug in T0*u into single tensor model along z-axis (i.e., D is a diagonal matrix)
	y = np.exp(-b*u.T.dot(T0.T).dot(D).dot(T0).dot(u))

	return y


# Description: generate R matrix based on response function (myresponse)
# \sqrt{4pi/(2l+1)}<R, Phi_{l0}> in blocks of length 2l+1
# since R(theta)=\sum_{l,m}r_{lm}\Phi_{lm}(theta,phi), we take N samples on a dense grid of sphere, then
# R_{N*1}=Phi_{N*L}*r_{L*1}. <R, Phi_{l0}>=r_{l0} can be extracted from r=(Phi'*Phi)^{-1}Phi'*R
# Input: b: b-value (1 = 1,000mm^2/s, 3 = 3,000mm^2/s)
#       ratio: the ratio between the 1st eigenvalue and the average of the 2nd and 3rd eigenvalues.
#       lmax : the maximum level of basis. (number of symmetrized SH basis  = (lmax+1)*(lmax+2)/2)
# Output: R_matrix: number of symmetrized SH basis * number of symmetrized SH basis, a diagonal matrix with diagonal elements

def Rmatrix(b, ratio, lmax, J = 5, lambda1=0.001):

	pos, theta, phi = spmesh(J, half = 0)

	# evaluate response function on grid specified by (theta, phi)
	#R = np.array([myresponse(b, ratio, 0, 0, theta[i], phi[i]) for i in range(len(theta))])
	R = np.array([single_tensor(b, ratio, 0, 0, theta[i], phi[i], lambda1=lambda1) for i in range(len(theta))])
	SH_matrix = spharmonic(theta, phi, lmax)
	# inner product of response function and all SH basis up to lmax
	r = np.linalg.solve(SH_matrix.T.dot(SH_matrix), SH_matrix.T.dot(R))

	l = np.arange(0, lmax+1, 2)
	# inner product of response function and all SH basis up to lmax with m=0
	r_l = r[(l*(l+1)/2).astype("int")] * np.sqrt(4*np.pi/(2*l+1))
	R_matrix = np.diag(np.repeat(r_l, (2*l+1).astype("int")))

	return R_matrix
