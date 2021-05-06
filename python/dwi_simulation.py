"""
 Purpose: Functions for numerical simulation (Generate Data, and Evaluation)
 30th Jan 2020: Created (By Seungyong Hwang)
"""
#Built-in package
import numpy as np

# Defined functions for the tasks with Spherical Harmonic basis.
from sphere_harmonics import single_tensor, spharmonic, spharmonic_eval #myresponse
from sphere_mesh import spmesh
from fod_estimation import fod_stand

#%% Description: Calcaulte SH coefficient for Dirac delta function (theta0, phi0) on sphere via SH basis with lmax
def dirac_sh_coeff(lmax, theta0, phi0):
    
    L = int((lmax+1)*(lmax+2)/2)  # number of symmetrized SH basis
    coeff = np.zeros((L,1))
    
    for l in range(0, lmax+1, 2): #SH level
        
        for m in range(-l, l+1):  # SH phase
            
            coeff_index = int( (l+1) * (l+2) / 2 - (l - m) - 1)
            Y_lm1 = spharmonic_eval(l, m, theta0, phi0)
            Y_lm2 = spharmonic_eval(l, -m, theta0, phi0)
            if m < 0 and m >= (-l):
                coeff[coeff_index] = (((-1)**m * Y_lm1 + Y_lm2)/np.sqrt(2)).real
            elif m == 0:
                coeff[coeff_index] = Y_lm1.real
            else:
                coeff[coeff_index] = (1j*( (-1)**(m+1) * Y_lm1 + Y_lm2)/np.sqrt(2)).real

    return coeff

#%% Description: Generate True FOD based on the specified (theta0, phi0)
# each fiber is along (theta0, phi0) and is evaluated with SH with l_max 
def true_fod_crossing(weight, theta0, phi0, lmax = 20):
    pos, theta, phi = spmesh(J = 5, half = False) 
    SH = spharmonic(theta, phi, lmax)  
    
    if len(weight) > 1:
        coeff_fod = weight[0] * dirac_sh_coeff(lmax, theta0[0], phi0[0])
        for i in range(1, len(weight)):
            coeff_fod =  coeff_fod + weight[i] *  dirac_sh_coeff(lmax, theta0[i], phi0[i])
    
    fod_temp = SH.dot(coeff_fod)
    fod = fod_temp * (fod_temp > 0.1 * max(fod_temp))
    fod  = fod_stand(fod)
    
    return fod
#%% Description: generate noiseless DWI for a crossing fiber - Seungyong
# each response function is along (theta0, phi0) and is evaluated at (theta, phi)
def tensor_crossing(b, ratio, weight, theta0, phi0, theta, phi, lambda1 = 0.001):
    
    dwi_noiseless = weight[0] * np.array([single_tensor(b, ratio, theta0[0], phi0[0], theta = theta[at], phi = phi[at], lambda1 = lambda1) for at in range(len(theta))])
    
    if len(weight) > 1:
        for i in range(1, len(weight)):
            dwi_noiseless =  dwi_noiseless + weight[i] * np.array([single_tensor(b, ratio, theta0[i], phi0[i], theta = theta[at], phi = phi[at], lambda1 = lambda1) for at in range(len(theta))])
        
    
    return(dwi_noiseless)

#%% Description: add Rician noise on noiseless DWI
# if matlab_randn=True, use pre-stored normal-distributed random values generated in matlab
# if matlab_randn=False, generate normal-distributed random values in python
def Rician_noise(dwi_noiseless, sigma, seed = 0):
    
    np.random.seed(seed)
    error = sigma*np.random.randn(*((2,)+dwi_noiseless.shape))
    
    dwi = dwi_noiseless.copy()
    dwi = np.sqrt((dwi + error[0])**2 + error[1]**2)
    
    return dwi
#%% Description: Transform the spherical coordinate sysmtem to cartesian coordinate system
def sph2cart(theta, phi, r = 1):
    return r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)
#%%
def cart2sph(x, y, z):
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    phi += 2*np.pi*(phi<0)
    return theta, phi
#%% Description: Metric for distribution comparison (hellinger distance)
def hellinger_distance(f1, f2, const):
    f1 = fod_stand(f1)
    f2 = fod_stand(f2)    
    return max(np.sqrt(0.5 * np.sum((np.sqrt(f1) - np.sqrt(f2))**2)), 1e-4)

#%% Description: Based on the true fiber location and estimated fiber location, calcaulte separation angle and peaks error.
def fib2_sep_angle(true_pos1, true_pos2, est_pos1, est_pos2):
    
    fod_temp = [true_pos1, true_pos2]
    
    ang11 = np.arctan2(np.linalg.norm(np.cross(true_pos1, est_pos1)), np.dot(true_pos1, est_pos1))
    ang12 = abs(np.pi - ang11)
    ang111 = min(ang11, ang12)
    
    ang21 = np.arctan2(np.linalg.norm(np.cross(true_pos2, est_pos1)), np.dot(true_pos2, est_pos1))
    ang22 = abs(np.pi - ang21)
    ang222 = min(ang21, ang22)

    angles = [ang111, ang222]

    peak1_error = min(angles)
    
    index_temp = angles.index(peak1_error)

    if(np.dot(fod_temp[index_temp],est_pos1) < 0):
        est_pos1 = - est_pos1

    del fod_temp[index_temp]
    
    ang11 = np.arctan2(np.linalg.norm(np.cross(fod_temp[0],est_pos2)), np.dot(fod_temp[0],est_pos2))
    ang12 = abs(np.pi - ang11)
    peak2_error = min(ang11, ang12)
    
    ang11 = np.arctan2(np.linalg.norm(np.cross(est_pos1, est_pos2)), np.dot(est_pos1, est_pos2))
    ang12 = abs(np.pi - ang11)
    sep_angle = min(ang11, ang12)
    
    peak1_error = peak1_error * 180 / np.pi
    peak2_error = peak2_error * 180 / np.pi
    sep_angle = sep_angle * 180 / np.pi
    
    return peak1_error, peak2_error, sep_angle

#%%
def fib3_sep_angle(true_pos1, true_pos2, true_pos3, est_pos1, est_pos2, est_pos3):

    fod_temp=[true_pos1, true_pos2, true_pos3]
    
    fod_loc=[0,1,2]

    #Separation Angle
    ang11 = np.arctan2(np.linalg.norm(np.cross(est_pos1,est_pos2)), np.dot(est_pos1,est_pos2))
    ang12 = abs(np.pi - ang11)
    sep_angle1 = min(ang11, ang12)

    ang11 = np.arctan2(np.linalg.norm(np.cross(est_pos1,est_pos3)), np.dot(est_pos1,est_pos3))
    ang12 = abs(np.pi - ang11)
    sep_angle2 = min(ang11, ang12)
    
    ang11 = np.arctan2(np.linalg.norm(np.cross(est_pos2,est_pos3)), np.dot(est_pos2,est_pos3))
    ang12 = abs(np.pi - ang11)
    sep_angle3 = min(ang11, ang12)

    #Peak Error
    
    ang11 = np.arctan2(np.linalg.norm(np.cross(fod_temp[0],est_pos1)), np.dot(fod_temp[0],est_pos1))
    ang12 = abs(np.pi - ang11)
    ang111 = min(ang11, ang12)
    
    ang21 = np.arctan2(np.linalg.norm(np.cross(fod_temp[1],est_pos1)), np.dot(fod_temp[1],est_pos1))
    ang22 = abs(np.pi - ang21)
    ang222 = min(ang21, ang22)
    
    ang31 = np.arctan2(np.linalg.norm(np.cross(fod_temp[2],est_pos1)), np.dot(fod_temp[2],est_pos1))
    ang32 = abs(np.pi - ang31)
    ang333 = min(ang31, ang32)
    
    angles=[ang111,ang222,ang333]

    index_temp = angles.index(np.min(angles))
    
    if(np.dot(fod_temp[fod_loc[index_temp]],est_pos1)<0):
        est_pos1 = -est_pos1
    
    theta_temp, phi_temp = cart2sph(est_pos1[0], est_pos1[1], est_pos1[2])
    
    if(fod_loc[index_temp] == 0):# phi1 phi2 theta1 theta2 store
        phi1 = phi_temp
        theta1 = theta_temp
        peak1_error = np.min(angles) # peak 1 error
        sep1_angle = sep_angle1
    elif(fod_loc[index_temp] == 1):
        phi2 = phi_temp
        theta2 = theta_temp
        peak2_error = np.min(angles) # peak 2 error
        sep2_angle = sep_angle2
    else:
        phi3 = phi_temp
        theta3 = theta_temp
        peak3_error = np.min(angles) # peak 3 error
        sep3_angle = sep_angle3            

    del fod_loc[index_temp]

    ang11 = np.arctan2(np.linalg.norm(np.cross(fod_temp[fod_loc[0]],est_pos2)), np.dot(fod_temp[fod_loc[0]],est_pos2))
    ang12 = abs(np.pi - ang11)
    ang111 = min(ang11, ang12)

    ang21 = np.arctan2(np.linalg.norm(np.cross(fod_temp[fod_loc[1]],est_pos2)), np.dot(fod_temp[fod_loc[1]],est_pos2))
    ang22 = abs(np.pi - ang21)
    ang222 = min(ang21, ang22)    

    angles=[ang111,ang222]

    index_temp= angles.index(np.min(angles))
    
    if(np.dot(fod_temp[fod_loc[index_temp]],est_pos2)<0):
        est_pos2 = -est_pos2
    
    theta_temp, phi_temp = cart2sph(est_pos2[0], est_pos2[1], est_pos2[2])
    
    if(fod_loc[index_temp] == 0):# phi1 phi2 theta1 theta2 store
        phi1 = phi_temp
        theta1 = theta_temp
        peak1_error = np.min(angles) # peak 1 error
        sep1_angle = sep_angle1
    elif(fod_loc[index_temp] == 1):
        phi2 = phi_temp
        theta2 = theta_temp
        peak2_error = np.min(angles) # peak 2 error
        sep2_angle = sep_angle2
    else:
        phi3 = phi_temp
        theta3 = theta_temp
        peak3_error = np.min(angles) # peak 3 error 
        sep3_angle = sep_angle3              

    del fod_loc[index_temp]

    ang11 = np.arctan2(np.linalg.norm(np.cross(fod_temp[fod_loc[0]],est_pos3)), np.dot(fod_temp[fod_loc[0]],est_pos3))
    ang12 = abs(np.pi - ang11)
    ang111 = min(ang11, ang12)
    
    if(np.dot(fod_temp[fod_loc[0]],est_pos3)<0):
        est_pos3 = -est_pos3

    theta_temp, phi_temp = cart2sph(est_pos3[0], est_pos3[1], est_pos3[2])
    
    if(fod_loc[0] == 0):# phi1 phi2 theta1 theta2 store
        phi1 = phi_temp
        theta1 = theta_temp
        peak1_error = np.min(angles) # peak 1 error
        sep1_angle = sep_angle1
    elif(fod_loc[0] == 1):
        phi2 = phi_temp
        theta2 = theta_temp
        peak2_error = np.min(angles) # peak 2 error
        sep2_angle = sep_angle2
    else:
        phi3 = phi_temp
        theta3 = theta_temp
        peak3_error = np.min(angles) # peak 3 error
        sep3_angle = sep_angle3             

    peak1_error = peak1_error * 180 / np.pi
    peak2_error = peak2_error * 180 / np.pi
    peak3_error = peak3_error * 180 / np.pi

    sep1_angle = sep1_angle * 180 / np.pi
    sep2_angle = sep2_angle * 180 / np.pi
    sep3_angle = sep3_angle * 180 / np.pi

    return sep1_angle, sep2_angle, sep3_angle, peak1_error, peak2_error, peak3_error


#%% From NARM project
def region3D(n, theta, phi, b, ratio, lmax, r = 0.6, sigma = 0.05, seed = 0):

    r_mesh = np.linspace(1./(2*n), 1-1./(2*n), n)
    r_y, r_x, r_z = np.meshgrid(r_mesh, r_mesh, r_mesh)

    # fiber generation schema 1
    # find plain perpendicular to xy-plain and has pi/4 angle towards x-axis conter-clockwisely, 
    # and point (x,y,z) is on it (this plain has width sqrt(2)*(1-|x-y|) and height 1)
    # 2D coordinates of point (x,y,z) on this plain is (sqrt(2)*min(x,y), z)
    # extend this plain to have width 1 and height 1, then 2D coordinates become (min(x,y)/(1-|x-y|),z) = (tx,ty)
    # in this extended plain, set bottom right vertex as center, and draw 1/4 circle with radius 1 and r
    # points being selected in this extended plain should satisfy:
    # lower edge of its square should be lower than intersection between radius-1 circle and right edge of its square
    # i.e., ty-0.05<sqrt(1^2-(1-(tx+0.05))^2)
    # upper edge of its square should be higher than intersection between radius-r circle and left edge of its square
    # i.e., ty+0.05>sqrt(r^2-(1-(tx-0.05))^2)
    r_xy_1 = np.minimum(r_x, r_y)/(1-np.abs(r_x-r_y))
    fib_indi_1 = ((1-(r_xy_1+0.05))**2+(r_z-0.05)**2 < 1) & ((1-(r_xy_1-0.05))**2+(r_z+0.05)**2 > r**2)
    # fiber generation schema 2
    # find plain perpendicular to xy-plain and has pi*3/4 angle towards x-axis conter-clockwisely, 
    # and point (x,y,z) is on it (this plain has width sqrt(2)*min(2-x-y,x+y) and height 1)
    # 2D coordinates of point (x,y,z) on this plain is (sqrt(2)*min(x,1-y), z)
    # extend this plain to have width 1 and height 1, then 2D coordinates become (min(x,1-y)/min(2-x-y,x+y),z) = (tx,ty)
    # in this extended plain, set bottom left vertex as center, and draw 1/4 circle with radius 1 and r
    # points being selected in this extended plain should satisfy:
    # lower edge of its square should be lower than intersection between radius-1 circle and left edge of its square
    # i.e., ty-0.05<sqrt(1^2-(tx-0.05)^2)
    # upper edge of its square should be higher than intersection between radius-r circle and right edge of its square
    # i.e., ty+0.05>sqrt(r^2-(tx+0.05)^2)
    r_xy_2 = np.minimum(r_x, 1-r_y)/np.minimum(2-r_x-r_y, r_x+r_y)
    fib_indi_2 = ((r_xy_2-0.05)**2+(r_z-0.05)**2 < 1) & ((r_xy_2+0.05)**2+(r_z+0.05)**2 > r**2)

    # fib_indi[i,j]=0: voxel_ij contains no fibers
    # fib_indi[i,j]=1: voxel_ij contains fiber generated from schema 1
    # fib_indi[i,j]=2: voxel_ij contains fiber generated from schema 2
    # fib_indi[i,j]=3: voxel_ij contains fibers generated from schema 1 and schema 2 (crossing fibers)
    # simulation region plot can be obtained by rotating fib_indi 90 degree conter-clockwisely
    fib_indi = fib_indi_1 + 2*fib_indi_2

    # spherical coordinates of fiber directions in each voxel
    # coordinate system here is within each voxel, with x-axis perpendicular to the screen, 
    # and is different from coordinate system for generating fibers voxelwisely in simulation region
    theta_fib_1 = np.arctan2(r_xy_1, 1-r_z)  # polar angle of fiber 1
    theta_fib_2 = np.arctan2(r_xy_2, r_z)+np.pi/2 # polar angle of fiber 2
    theta_fib = np.stack((theta_fib_1, theta_fib_2), axis=-1)
    phi_fib_1 = np.ones((n, n, n))*0.75*np.pi  # azimuthal angle of fiber 1
    phi_fib_2 = np.ones((n, n, n))*0.25*np.pi  # azimuthal angle of fiber 2
    phi_fib = np.stack((phi_fib_1, phi_fib_2), axis=-1)

    # calculate noiseless DWI, DWI (with Rician noise) and SH coefficients of dirac functions
    DWI_noiseless = np.zeros((n, n, n, len(theta)))
    SH_coef = np.zeros((n, n, n, int((lmax+1)*(lmax+2)/2)))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if fib_indi[i, j, k] == 0:
                    DWI_noiseless[i, j, k] = np.exp(-b)  # D becomes identity matrix in single tensor model
                    SH_coef[i, j, k, 0] = spharmonic_eval(0, 0, 0, 0).real
                elif fib_indi[i, j, k] == 1:
                    for l in range(len(theta)):
                        DWI_1 = single_tensor(b, ratio, theta_fib[i, j, k, 0], phi_fib[i, j, k, 0], theta[l], phi[l])
                        DWI_noiseless[i, j, k, l] = DWI_1
                    SH_coef[i, j, k] = spharmonic(theta_fib[i, j, k, :1], phi_fib[i, j, k, :1], lmax)
                elif fib_indi[i, j, k] == 2:
                    for l in range(len(theta)):
                        DWI_2 = single_tensor(b, ratio, theta_fib[i, j, k, 1], phi_fib[i, j, k, 1], theta[l], phi[l])
                        DWI_noiseless[i, j, k, l] = DWI_2
                    SH_coef[i, j, k] = spharmonic(theta_fib[i, j, k, 1:], phi_fib[i, j, k, 1:], lmax)
                else:  # crossing-fiber case
                    for l in range(len(theta)):
                        DWI_1 = single_tensor(b, ratio, theta_fib[i, j, k, 0], phi_fib[i, j, k, 0], theta[l], phi[l])
                        DWI_2 = single_tensor(b, ratio, theta_fib[i, j, k, 1], phi_fib[i, j, k, 1], theta[l], phi[l])
                        DWI_noiseless[i, j, k, l] = (DWI_1+DWI_2)/2  # volumn fractions for two fibers are both 0.5
                    SH_coef[i, j, k] = (spharmonic(theta_fib[i, j, k, :1], phi_fib[i, j, k, :1], lmax) 
                        + spharmonic(theta_fib[i, j, k, 1:], phi_fib[i, j, k, 1:], lmax))/2
    DWI = Rician_noise(DWI_noiseless, sigma, seed)
    # take upper half of simulation region
    DWI_noiseless, DWI, SH_coef = DWI_noiseless[:, :, int(n/2):], DWI[:, :, int(n/2):], SH_coef[:, :, int(n/2):]
    theta_fib, phi_fib, fib_indi = theta_fib[:, :, int(n/2):], phi_fib[:, :, int(n/2):], fib_indi[:, :, int(n/2):]

    return DWI_noiseless, DWI, SH_coef, theta_fib, phi_fib, fib_indi