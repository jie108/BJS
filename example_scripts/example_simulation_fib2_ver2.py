"""
Purpose: Example for Simulation (2 fiber case)
@author: Seungyong (30 Jan 2020)
"""

#%% Setup 

#%% Set working directory
import os

#path = 'python script path'
#path= 'C:/Users/Jie/Dropbox/Projects/CRCNS_projects_2015_/SeungyongHwang_FDD_estimation/codes/BJS/python'
path = '/Users/seungyong/Dropbox/FDD_estimation/codes/BJS/python'
os.chdir(path)

# import built in Packages
import numpy as np
import tqdm
# import functions: generate Spherical Harmonic basis, spherical sampling, peak detection, and FOD estimation, generating DWI data.
#from sphere_harmonics import spharmonic_eval, spharmonic, myresponse, Rmatrix
from sphere_harmonics import *
from sphere_mesh import spmesh
from FOD_peak import FOD_Peak
from fod_estimation import *
from dwi_simulation import *

#%% Simulation 
#%% Setup simulation parameters 
# True FOD setting: 2-fiber
weight = [0.5, 0.5] #weight for each fiber 
test_degree = 45 # separation angle 
ratio = 10 #response function: ratio between the first eigenvalue and the avarage of the second and the third eigenvalues in the tensor. 

# Parameters for DWI generation and level of  SH basis used in the methods
J = 3 #Tessellation order: control the sampling scheme of the gradient directions on the sphere. E.g., J = 2.5 (n = 41), J = 3 (n = 91), J = 4(n = 321)     
lmax = 10 #the maximum level of real symmetrized spherical harmonic basis used in (BJS, superCSD, SHridge); for n=41 (J=2.5), use lmax=6; for n=91 (J=3), use lmax=10, for n=321, lmax=12
b = 3000 #b-value (1000 = 1,000s/mm^2, 3000 = 3,000s/mm^2)
sigma = 0.02 #SNR = 1/sigma (sigma=0.02 (SNR = 50), sigma=0.05 (SNR = 20))

# level of SH basis used for the super-resolution updating
lmax_update = 10 #the maximum level of real symmetrized spherical harmonic basis used for the super-resolution updating in (BJS, superCSD)

# the number of replication
rep=100 

#% Generate Design Matrix and DWI data 
#sample the gradient directions
pos, theta, phi, sampling_index = spmesh(J = J, half = True) # cartesian coordinates and spherical coordinates)on the sphere (equi_angle grid)

#generate design matrix: used for the methods (Notes: takes time to generate matrice R and SH)
SH = spharmonic(theta, phi, lmax_update) #  Phi
R = Rmatrix(b, ratio, lmax_update) # response matrix R; design matrix = Phi *R

# Generate True FOD: for data generation, and evaluation/plotting purpose 
fib2_theta, fib2_phi = [(180 - test_degree) * np.pi / 180, np.pi], [np.pi/2, np.pi/2] ##directions of the true fibers 
fib2_pos1 = sph2cart(fib2_theta[0], fib2_phi[0]) #cartesian coordinates of the true fiber directions
fib2_pos2 = sph2cart(fib2_theta[1], fib2_phi[1])
true_fod = true_fod_crossing(weight, fib2_theta, fib2_phi, lmax = 20) # projection of the FOD on a high order SH basis 

# Generate DWI observations: 
dwi_noiseless = tensor_crossing(b, ratio, weight, fib2_theta, fib2_phi, theta, phi) #noisless signals
dwi = np.zeros((len(theta),rep))
for i in range(rep):
    dwi[:,i] = Rician_noise(dwi_noiseless, sigma, seed=i) ##add Rician noise to noiseless DWI signals

#% Generate SH evaluation matrices and True FOD: take a bit of time
#generate SH evaluation matrix used in the super-resolution updating step
pos_dense_half, theta_dense_half, phi_dense_half, sampling_index_dense_half = spmesh(J = 5, half = True) # generate the evaluation grid points on the sphere (equi_angle grid)
SHD = spharmonic(theta_dense_half, phi_dense_half, lmax_update) # design matrix

#generate SH evaluation matrix on the whole sphere on a dense grid for plotting/evaluation purpose 
pos_dense, theta_dense, phi_dense = spmesh(J = 5, half = False) # generate the location of grid point on the sphere (equi_angle grid)
SHP = spharmonic(theta_dense, phi_dense, lmax_update) #
#% Parameters for methods 
# Tuning parameter grid for SHRidge: equally spaced on log scale on the coarseer grid, linearly equally spaced within the sub-intervals
lam = np.array([np.linspace(1e-06,1e-05,21)[:20],
            np.linspace(1e-05,1e-04,21)[:20],
            np.linspace(1e-04,1e-03,21)[:20],
            np.linspace(1e-03,1e-02,21)[:20],
            np.linspace(1e-02,1e-01,21)[:20],
            np.linspace(1e-01,1,21)[:20],
            np.linspace(1,10,21)[:20]])
lam = lam.reshape(1,-1)[0]


# Parameters for Peak detection
nbhd = 40  #neighborhood size
thresh = 0.4 #peak thresholding: ingore any peak < thresh * max
degree = 5 #clustering peaks within "degree" as one 
peak_cut = 4 # maximum number of peaks: only return the top "peak_cut" peaks  


#% Auxiliary components of the methods:
# BJS: generate the associated eigenvalues of each block of the covariance matrix used in BJS estimator definition
L = int((lmax+1)*(lmax+2)/2) #the number of SH basis used by the methods 
SH_init = SH[:,:L]
R_init = R[:L,:L]
mu1, mu2, muinf = cal_mu(SH_init, R_init, lmax)

#SHRidge: generate the LB penalty matrix
P=penalty_mat(lmax)

#Peak detection: 
dis = pos_dense.T.dot(pos_dense) # pairwise distance between the grid points on the dense evaluation grid
idx = np.zeros(dis.shape, dtype=int) # for each grid point, sort other grid points in increasing distance to it; this is used as input for the peak detection  algorithm and generated once to avoid repeated calculation 
for i in range(dis.shape[0]):
    idx[i, :] = np.argsort(-dis[i, :])
#% Apply the methods to the simulated DWI data
# arrayes to store results 
#store_est_fod_bjs1 = np.zeros((SHP.shape[0],rep)) #For FOD estimation
#store_est_fod_bjs15 = np.zeros((SHP.shape[0],rep)) #For FOD estimation
store_est_fod_bjs2 = np.zeros((SHP.shape[0],rep)) #For FOD estimation
#store_est_fod_bjs25 = np.zeros((SHP.shape[0],rep)) #For FOD estimation
#store_est_fod_bjs3 = np.zeros((SHP.shape[0],rep)) #For FOD estimation

store_est_fod_shridge = np.zeros((SHP.shape[0],rep))
store_est_fod_superCSD = np.zeros((SHP.shape[0],rep))

store_est_fod_coeff_shridge = np.zeros((L, rep))

#store_num_fib_bjs1 = np.zeros(rep) #For peak detection
#store_num_fib_bjs15 = np.zeros(rep)
store_num_fib_bjs2 = np.zeros(rep)
#store_num_fib_bjs25 = np.zeros(rep)
#store_num_fib_bjs3 = np.zeros(rep)

store_num_fib_shridge = np.zeros(rep)
store_num_fib_superCSD = np.zeros(rep)

#store_est_sep_bjs1 = [] #For separation angle 
#store_est_sep_bjs15 = []
store_est_sep_bjs2 = []
#store_est_sep_bjs25 = []
#store_est_sep_bjs3 = []

store_est_sep_shridge = []
store_est_sep_superCSD = []

#store_peak1_err_bjs1 = []
#store_peak2_err_bjs1 = []
#store_peak1_err_bjs15 = []
#store_peak2_err_bjs15 = []
store_peak1_err_bjs2 = []
store_peak2_err_bjs2 = []
#store_peak1_err_bjs25 = []
#store_peak2_err_bjs25 = []
#store_peak1_err_bjs3 = []
#store_peak2_err_bjs3 = []

store_peak1_err_superCSD = []
store_peak2_err_superCSD = []

store_peak1_err_shridge = []
store_peak2_err_shridge = []
#% perform estimation 
for i in tqdm.tqdm(range(rep), desc="FOD estimation(BJS)"): # ith replicate 
    #BJS
    #store_est_fod_bjs1[:,i] = BJS(dwi[:,i], SH, SHD, SHP, R, mu1, mu2, muinf, lmax, tau = 0, t=2/4)
    #store_est_fod_bjs15[:,i] = BJS(dwi[:,i], SH, SHD, SHP, R, mu1, mu2, muinf, lmax, tau = 0, t=3/4)
    store_est_fod_bjs2[:,i] = BJS(dwi[:,i], SH, SHD, SHP, R, mu1, mu2, muinf, lmax, tau = 0,t=4/4)
    #store_est_fod_bjs25[:,i] = BJS(dwi[:,i], SH, SHD, SHP, R, mu1, mu2, muinf, lmax, tau = 0,t=5/4)
    #store_est_fod_bjs3[:,i] = BJS(dwi[:,i], SH, SHD, SHP, R, mu1, mu2, muinf, lmax, tau = 0,t=6/4)

for i in tqdm.tqdm(range(rep), desc="FOD estimation(SHridge)"): # ith replicate 
    #SHRidge 
    store_est_fod_coeff_shridge[:,i], store_est_fod_shridge[:,i] = SH_ridge(dwi[:,i], SH, SHP, R, P, lmax, lam)    


for i in tqdm.tqdm(range(rep), desc="FOD estimation(superCSD)"): # ith replicate 
    #SCSD: take the SHRidge estimator as initial 
    store_est_fod_superCSD[:,i] = superCSD(dwi[:,i], SH, SHD, SHP, R, store_est_fod_coeff_shridge[:15,i])
    
for i in tqdm.tqdm(range(rep), desc="Peak Detection:"): 
    #peak detection 
    num_fib_shridge, fib_loc_shridge = FOD_Peak(store_est_fod_shridge[:,i], idx, nbhd, thresh, degree, pos_dense, sampling_index_dense_half, True, peak_cut)
    num_fib_superCSD, fib_loc_superCSD = FOD_Peak(store_est_fod_superCSD[:,i], idx, nbhd, thresh, degree, pos_dense, sampling_index_dense_half, True, peak_cut)
    #num_fib_bjs1, fib_loc_bjs1 = FOD_Peak(store_est_fod_bjs1[:,i], idx, nbhd, thresh, degree, pos_dense, sampling_index_dense_half, True, peak_cut)
    #num_fib_bjs15, fib_loc_bjs15 = FOD_Peak(store_est_fod_bjs15[:,i], idx, nbhd, thresh, degree, pos_dense, sampling_index_dense_half, True, peak_cut)
    num_fib_bjs2, fib_loc_bjs2 = FOD_Peak(store_est_fod_bjs2[:,i], idx, nbhd, thresh, degree, pos_dense, sampling_index_dense_half, True, peak_cut)
    #num_fib_bjs25, fib_loc_bjs25 = FOD_Peak(store_est_fod_bjs25[:,i], idx, nbhd, thresh, degree, pos_dense, sampling_index_dense_half, True, peak_cut)
    #num_fib_bjs3, fib_loc_bjs3 = FOD_Peak(store_est_fod_bjs3[:,i], idx, nbhd, thresh, degree, pos_dense, sampling_index_dense_half, True, peak_cut)
    

    #record peak detection results 
    store_num_fib_shridge[i] = num_fib_shridge
    store_num_fib_superCSD[i] = num_fib_superCSD
    #store_num_fib_bjs1[i] = num_fib_bjs1
    #store_num_fib_bjs15[i] = num_fib_bjs15
    store_num_fib_bjs2[i] = num_fib_bjs2
    #store_num_fib_bjs25[i] = num_fib_bjs25
    #store_num_fib_bjs3[i] = num_fib_bjs3
    

    #get and record separation angle estimation  
    if (num_fib_shridge==2):
        peak1_error_shridge, peak2_error_shridge, est_sep_shridge = fib2_sep_angle(fib2_pos1, fib2_pos2, fib_loc_shridge[:,0], fib_loc_shridge[:,1])
        store_est_sep_shridge.append(est_sep_shridge)
        store_peak1_err_shridge.append(peak1_error_shridge)
        store_peak2_err_shridge.append(peak2_error_shridge)

    if (num_fib_superCSD==2):
        peak1_error_scsd, peak2_error_scsd, est_sep_superCSD = fib2_sep_angle(fib2_pos1, fib2_pos2, fib_loc_superCSD[:,0], fib_loc_superCSD[:,1])
        store_est_sep_superCSD.append(est_sep_superCSD)
        store_peak1_err_superCSD.append(peak1_error_scsd)
        store_peak2_err_superCSD.append(peak2_error_scsd)
       
    #if (num_fib_bjs1==2):
    #    peak1_error_bjs1, peak2_error_bjs1, est_sep_bjs1 = fib2_sep_angle(fib2_pos1, fib2_pos2, fib_loc_bjs1[:,0], fib_loc_bjs1[:,1])
    #    store_est_sep_bjs1.append(est_sep_bjs1)
    #    store_peak1_err_bjs1.append(peak1_error_bjs1)
    #    store_peak2_err_bjs1.append(peak2_error_bjs1)

    #if (num_fib_bjs15==2):
    #    peak1_error_bjs15, peak2_error_bjs15, est_sep_bjs15 = fib2_sep_angle(fib2_pos1, fib2_pos2, fib_loc_bjs15[:,0], fib_loc_bjs15[:,1])
    #    store_est_sep_bjs15.append(est_sep_bjs15)
    #    store_peak1_err_bjs15.append(peak1_error_bjs15)
    #    store_peak2_err_bjs15.append(peak2_error_bjs15)
        
    if (num_fib_bjs2==2):
        peak1_error_bjs2, peak2_error_bjs2, est_sep_bjs2 = fib2_sep_angle(fib2_pos1, fib2_pos2, fib_loc_bjs2[:,0], fib_loc_bjs2[:,1])
        store_est_sep_bjs2.append(est_sep_bjs2)
        store_peak1_err_bjs2.append(peak1_error_bjs2)
        store_peak2_err_bjs2.append(peak2_error_bjs2)

    #if (num_fib_bjs25==2):
    #    peak1_error_bjs25, peak2_error_bjs25, est_sep_bjs25 = fib2_sep_angle(fib2_pos1, fib2_pos2, fib_loc_bjs25[:,0], fib_loc_bjs25[:,1])
    #    store_est_sep_bjs25.append(est_sep_bjs25)
    #    store_peak1_err_bjs25.append(peak1_error_bjs25)
    #    store_peak2_err_bjs25.append(peak2_error_bjs25)

    #if (num_fib_bjs3==2):
    #    peak1_error_bjs3, peak2_error_bjs3, est_sep_bjs3 = fib2_sep_angle(fib2_pos1, fib2_pos2, fib_loc_bjs3[:,0], fib_loc_bjs3[:,1])
    #    store_est_sep_bjs3.append(est_sep_bjs3)
    #    store_peak1_err_bjs3.append(peak1_error_bjs3)
    #    store_peak2_err_bjs3.append(peak2_error_bjs3)        

#% Evaluation 
#BJS:         
#correct_bjs1 = len(np.where(store_num_fib_bjs1==2)[0]) #percentage of correct peak detection
#under_bjs1 = len(np.where(store_num_fib_bjs1<2)[0])
#over_bjs1 = len(np.where(store_num_fib_bjs1>2)[0])

#correct_bjs15 = len(np.where(store_num_fib_bjs15==2)[0]) #percentage of correct peak detection
#under_bjs15 = len(np.where(store_num_fib_bjs15<2)[0])
#over_bjs15 = len(np.where(store_num_fib_bjs15>2)[0])

correct_bjs2 = len(np.where(store_num_fib_bjs2==2)[0]) #percentage of correct peak detection
under_bjs2 = len(np.where(store_num_fib_bjs2<2)[0])
over_bjs2 = len(np.where(store_num_fib_bjs2>2)[0])

#correct_bjs25 = len(np.where(store_num_fib_bjs25==2)[0]) #percentage of correct peak detection
#under_bjs25 = len(np.where(store_num_fib_bjs25<2)[0])
#over_bjs25 = len(np.where(store_num_fib_bjs25>2)[0])

#correct_bjs3 = len(np.where(store_num_fib_bjs3==2)[0]) #percentage of correct peak detection
#under_bjs3 = len(np.where(store_num_fib_bjs3<2)[0])
#over_bjs3 = len(np.where(store_num_fib_bjs3>2)[0])

#SHRidge
correct_shridge = len(np.where(store_num_fib_shridge==2)[0])
under_shridge = len(np.where(store_num_fib_shridge<2)[0])
over_shridge = len(np.where(store_num_fib_shridge>2)[0])

#SCSD
correct_superCSD = len(np.where(store_num_fib_superCSD==2)[0])
under_superCSD = len(np.where(store_num_fib_superCSD<2)[0])
over_superCSD = len(np.where(store_num_fib_superCSD>2)[0])    


#show results on screen : add an if to deal with 0% success rate; print the setting as in the tables
print("J: {}, lmax: {}, lmax_update:{}, SNR:{}, b-value:{}s/mm^2 ".format(J, lmax, lmax_update, 1/sigma, b))

#if (correct_bjs1 != 0):
#    print("BJS1: D.R {:.2f}, bias1 {:.3f}, bias2 {:.3f}, s.e {:.3f}, RMSAE {:.3f}".format(
#            correct_bjs1/rep, 
#            np.mean(store_est_sep_bjs1)-test_degree, np.mean(np.array(store_est_sep_bjs1)-test_degree),np.mean(np.array(store_peak1_err_bjs1)**2 + np.array(store_peak2_err_bjs1)**2)**(0.5)))
#else:
#    print("BJS1: D.R 0, mean(s.e.) - (-)")


#if (correct_bjs15 != 0):
#    print("BJS15: D.R {:.2f}, mean.sep {:.3f} RMSAE {:.3f}".format(
#            correct_bjs15/rep, np.mean(store_est_sep_bjs15), np.mean(np.array(store_peak1_err_bjs15)**2 + np.array(store_peak2_err_bjs15)**2)**(0.5)))
#else:
#    print("BJS1: D.R 0, mean(s.e.) - (-)")

if (correct_bjs2 != 0):
    print("BJS2: D.R {:.2f}, bias1 {:.3f}, bias2 {:.3f}, se {:.3f}, RMSAE {:.3f}".format(
            correct_bjs2/rep, np.mean(store_est_sep_bjs2)-test_degree, np.mean(np.array(store_est_sep_bjs2)-test_degree),
            np.std(np.array(store_est_sep_bjs2)-test_degree)/np.sqrt(correct_bjs2),
            np.mean(np.array(store_peak1_err_bjs2)**2 + np.array(store_peak2_err_bjs2)**2)**(0.5)))
else:
    print("BJS1: D.R 0, mean(s.e.) - (-)")

#if (correct_bjs25 != 0):
#    print("BJS25: D.R {:.2f}, mean.sep {:.3f} RMSAE {:.3f}".format(
#            correct_bjs25/rep, np.mean(store_est_sep_bjs25), np.mean(np.array(store_peak1_err_bjs25)**2 + np.array(store_peak2_err_bjs25)**2)**(0.5)))
#else:
#    print("BJS1: D.R 0, mean(s.e.) - (-)")

#if (correct_bjs3 != 0):
#    print("BJS3: D.R {:.2f}, mean.sep {:.3f} RMSAE {:.3f}".format(
#            correct_bjs3/rep, np.mean(store_est_sep_bjs3), np.mean(np.array(store_peak1_err_bjs3)**2 + np.array(store_peak2_err_bjs3)**2)**(0.5)))
#else:
#    print("BJS1: D.R 0, mean(s.e.) - (-)")    

if (correct_superCSD != 0):
    print("superCSD: D.R {:.2f}, bias1 {:.3f}, bias2 {:.3f}, se {:.3f}, RMSAE {:.3f}".format(
            correct_superCSD/rep, np.mean(store_est_sep_superCSD)-test_degree, np.mean(np.array(store_est_sep_superCSD)-test_degree),
            np.std(np.array(store_est_sep_superCSD)-test_degree)/np.sqrt(correct_superCSD),
            np.mean(np.array(store_peak1_err_superCSD)**2 + np.array(store_peak2_err_superCSD)**2)**(0.5)))
else:
    print("superCSD: D.R 0, mean(s.e.) - (-)")
    
if (correct_shridge != 0):
    print("SHridge: D.R {:.2f}, bias1 {:.3f}, bias2 {:.3f}, se {:.3f}, RMSAE {:.3f}".format(
            correct_shridge/rep, np.mean(store_est_sep_shridge)-test_degree, np.mean(np.array(store_est_sep_shridge)-test_degree),
            np.std(np.array(store_est_sep_shridge)-test_degree)/np.sqrt(correct_shridge),
            np.mean(np.array(store_peak1_err_shridge)**2 + np.array(store_peak2_err_shridge)**2)**(0.5)))
else:
    print("SHridge: D.R 0, mean(s.e.) - (-)")


#%%
bjs_mean =  np.mean(store_est_fod_bjs, axis=1)    
scsd_mean =  np.mean(store_est_fod_superCSD, axis=1)   
shridge_mean =  np.mean(store_est_fod_shridge, axis=1)
   
bjs_std =  np.std(store_est_fod_bjs, axis=1)    
scsd_std =  np.std(store_est_fod_superCSD, axis=1)   
shridge_std =  np.std(store_est_fod_shridge, axis=1)   

bjs_m2sd = bjs_mean +  2*bjs_std
scsd_m2sd = scsd_mean + 2*scsd_std
shridge_m2sd = shridge_mean + 2*shridge_std

#%
import scipy.io
file_name = "estfib2degree"+str(test_degree)+"J"+str(int(J))+'l'+str(lmax)+'b'+str(b)+'snr'+str(int(1/sigma))+'.mat'
scipy.io.savemat(file_name,{'degree':test_degree, 'J':J, 'b':b, 'lmax':lmax, 'lmax_update':lmax_update, 'snr' : 1/sigma, 'theta':fib2_theta,'phi':fib2_phi,'true':true_fod,'BJS_mean':bjs_mean, "SCSD_mean":scsd_mean, 'SHridge_mean':shridge_mean, 'BJS_m2sd':bjs_m2sd, 'SCSD_m2sd':scsd_m2sd, 'SHridge_m2sd':shridge_m2sd})