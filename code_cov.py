

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import integrate
import pylab as py
#from scipy.optimize import minimize
from math import *
from lmfit import minimize, Parameters, report_errors, Minimizer
from iminuit import Minuit, describe, Struct
import numpy
import pyfits
import glob

from  scipy.linalg.lapack import dpotrf
from scipy.linalg.lapack import dtrtrs
import cosmolopy.distance as cd

# -*- coding: utf-8 -*-

#!/usr/bin/python

class Supernovae:

	def __init__(self,lignes):
		for i in range(0,len(lignes)):
			lignes[i]=lignes[i].split(" ")[1]
		
		self.name = lignes[0]
		self.z  = float(lignes[1])
		self.z_err  = float(lignes[2])
		self.maxdate = float(lignes[3])
		self.maxdate_err = float(lignes[4])
		self.bmax  = float(lignes[5])
		self.bmax_err  = float(lignes[6])
		self.x0  = float(lignes[7])
		self.x0_err  = float(lignes[8])
		self.x1  = float(lignes[9])
		self.x1_err  = float(lignes[10])
		self.colour  = float(lignes[11])
		self.colour_err  = float(lignes[12])


	def aff(self):
		print self.name,self.z

def openFichier(nom):
	fd = open(nom,'r')
	lignes_tmp = fd.readlines()
	fd.close()
	return lignes_tmp

def recupererData(chemin):
	tab_supernovae=[]
	lignes = openFichier(chemin)
	lignes = [x for x in lignes if not x.startswith('#')]
	for l in range(0,len(lignes),14):
		tab_supernovae.append(Supernovae(lignes[l:l+13]))
	return tab_supernovae


def integrale(zz,wm,wl):
	return ((1.0+zz)**2.0 * (1.0+wm*zz) - zz*(2.0+zz)*wl)**-0.5

def script_lumdist(wm, wl, zed):

###calcul de DL sans H0
####Modele domine par la matire et la constante cosmologique####
	H0= 0.7
	clum = 299792458.
	zed = np.array(zed)
	cosmobit=[]
	#print 1.-(wm+wl)

	if np.abs(1.0 - (wm +wl)) <=0.001:
		#courbure nulle univers plat
		#print 'on est dans le cas 1'
		curve = 1.0
		const = clum * (1+zed)/(H0*np.sqrt(np.abs(curve)) )
		for zz in zed:
			
			integral=integrate.quad(integrale, 0, zz, args=(wm, wl,))[0]
			cosmobit.append(integral*np.sqrt(curve))
		 
		return const * cosmobit ###calcul de DL

	elif wm+wl > 1.:
		#courbure positive univers ferme
		#print 'on est dans le cas 2'
		curve = 1.0 - wm - wl
		const = clum * (1+zed)/(H0*np.sqrt(np.abs(curve)))
		tmp=[]
		for zz in zed:
			integral=integrate.quad(integrale, 0, zz, args=(wm, wl,))[0]
			cosmobit.append(sin(integral*np.sqrt(abs(curve))))
		return const * cosmobit ###calcul de DL
	else:
		#print 'on est dans le cas 3'
		#courbure negative univers ouvert
		curve = 1.0 - wm - wl
		const = clum * (1+zed)/(H0*np.sqrt(np.abs(curve)))

		for zz in zed:
			integral=integrate.quad(integrale, 0, zz, args=(wm, wl,))[0]
			cosmobit.append(sinh(integral*np.sqrt(abs(curve))))
		return const * cosmobit ###calcul de DL


####Prise en compte matrice de covariance####


Ceta = None
sigmaTxt = None
def mu_cov(alpha, beta):
	global Ceta, sigmaTxt

	if Ceta==None:
		for mat in glob.glob('/home/johanna/Bureau/fit_SN/covmat/C*.fits'):
			for mat in glob.glob('/home/johanna/Bureau/fit_SN/covmat/C*.fits'):
				Ceta = sum([pyfits.getdata(mat) for mat in glob.glob('/home/johanna/Bureau/fit_SN/covmat/C*.fits')])
	if sigmaTxt==None: 
		sigmaTxt = numpy.loadtxt('/home/johanna/Bureau/fit_SN/covmat/sigma_mu.txt')

	sigma=sigmaTxt
	Cmu = numpy.zeros_like(Ceta[::3,::3])



	for i, coef1 in enumerate([1., alpha, -beta]):
		for j, coef2 in enumerate([1., alpha, -beta]):
			Cmu += (coef1 * coef2) * Ceta[i::3,j::3]

	Cmu_avant_diag = Cmu.copy()
	# Add diagonal term from Eq. 13

	sigma_pecvel = (5 * 150 / 3e5) / (numpy.log(10.) * sigma[:, 2])
	Cmu[numpy.diag_indices_from(Cmu)] += sigma[:, 0] ** 2 + sigma[:, 1] ** 2 + sigma_pecvel ** 2
    

	return Cmu

def delta(i, j,k): 
	if 3*i == j+k: 
		return 1 
	else: 
		return 0 


def deltaM(masse,deltam):
	tab_deltam=np.zeros((740,))

	for i in range(0,len(masse)):
		if masse[i]  > 10:
			tab_deltam[i] = -deltam

	return tab_deltam

def modelTheo(z,omegam,omegal):
	tab=[]
	cosmo = {'omega_M_0' : omegam, 'omega_lambda_0' : omegal, 'h' : 0.70}
	cosmo = cd.set_omega_k_0(cosmo)
	for i in range(0,len(z)):
		tab.append(cd.luminosity_distance(z[i], **cosmo)*10**5)
	return tab

nbAppel = 0




def calculResidu2(params,eta,masse,tab_cov_m_s,tab_cov_m_c,tab_cov_s_c,redshift,bmag,x0,x1,c,redshift_err,bmag_err,x0_err,x1_err,c_err):
	global nbAppel
	nbAppel=nbAppel+1

	alpha = params['alpha'].value
	beta = params['beta'].value
	omega_m = params['omega_m'].value
	omega_l = params['omega_l'].value
	M = params['M'].value
	deltam=params['deltam'].value
	A=np.zeros((740,3*740))

	print nbAppel, ")", alpha,beta,omega_m,omega_l,M,deltam

	Cmu = mu_cov(alpha, beta)


	for i in range(0,740):
		A[i][i*3]=1.
		A[i][i*3+1]=alpha
		A[i][i*3+2]=-beta
		

	mu=np.dot(A,eta)-M+deltaM(masse,deltam)#+x0
	#model = 5.0*np.log10(np.array(modelTheo(redshift,omega_m, omega_l)))

	model = 5.0 * np.log10(np.array(script_lumdist(omega_m, omega_l, redshift))) 
	tab_res=mu-model


	#fig=plt.figure()
	#plt.scatter(redshift,script_lumdist(omega_m, omega_l, redshift))
	#nom='dll'+'.pdf'
	#plt.savefig(nom)
	#plt.close(fig)
	#exit(1)


	#**********************************
	#Mise a jour matrice covariance
	
	for i in range(0,740):
		cov_m_s=tab_cov_m_s[i] #================?
		cov_m_c=tab_cov_m_c[i]  #================?
		cov_s_c=tab_cov_s_c[i]  #================?
		Cmu[i,i]=Cmu[i, i] + bmag_err[i]**2 + (alpha*x1_err[i])**2 + (beta * c_err[i])**2 + 2.0* (alpha*cov_m_s - beta *cov_m_c -alpha*beta*cov_s_c)
	
	#********************************




	#**********************************
	#Premier methode
	Cinverse= np.linalg.inv(Cmu)

	residu=(tab_res.transpose()*(Cinverse*tab_res))
	#**********************************



	#**********************************
	#Deuxieme methode
	#dpotrf_matrix, info = dpotrf(Cmu);
	#if(info!=0):
	#	print "problem info = ", info, "dans dpotrf"
	#residu, entier=dtrtrs(dpotrf_matrix, tab_res);  
	#**********************************


	#print tab_res.T.shape
	#print Cinverse.shape
	#print tab_res.shape
	#print residu.shape
	#exit(1)

	return residu


def fonctionMinimizer(alpha,beta,omega_m,omega_l,M,deltam):
	global nbAppel
	nbAppel=nbAppel+1
	global zcmb_ar,bmag_ar,x1_ar,c_ar,tab_masse

	A=np.zeros((740,3*740))
	eta_ar=np.zeros(3*740)


	for i in range(0,len(bmag_ar)):
              eta_ar[i*3] = bmag_ar[i]
              eta_ar[i*3+1]=x1_ar[i]
              eta_ar[i*3+2]=c_ar[i]

	print nbAppel, ")", alpha,beta,omega_m,omega_l,M,deltam

	Cmu = mu_cov(alpha, beta)


	for i in range(0,740):
		A[i][i*3]=1.
		A[i][i*3+1]=alpha
		A[i][i*3+2]=-beta
		

	mu=np.dot(A,eta_ar)-M+deltaM(tab_masse,deltam)


	model = 5.0 * np.log10(np.array(script_lumdist(omega_m, omega_l, zcmb_ar))) 
	tab_res=mu-model

	#Mise a jour matrice covariance
	
	for i in range(0,740):
		cov_m_s=tab_cov_m_s[i] #================?
		cov_m_c=tab_cov_m_c[i]  #================?
		cov_s_c=tab_cov_s_c[i]  #================?
		Cmu[i,i]=Cmu[i, i] + bmag_err_ar[i]**2 + (alpha*x1_err_ar[i])**2 + (beta * c_err_ar[i])**2 + 2.0* (alpha*cov_m_s - beta *cov_m_c -alpha*beta*cov_s_c)
	

	Cinverse= np.linalg.inv(Cmu)


	residu=(tab_res.transpose()*(Cinverse*tab_res))
	s = 0.0
	for i in range(0,residu.shape[0]):
 		for j in range(0,residu.shape[1]):
 			s = s + residu[i,j]**2

	
	return s
	


print "Univers plat!"




liste_sn=recupererData("/home/johanna/Bureau/fit_SN/snfit_data/output_final2salt")
tab_masse=[]
tab_cov_m_s=[]
tab_cov_m_c=[]
tab_cov_s_c=[]

ligne_masse=openFichier("/home/johanna/Bureau/fit_SN/snfit_data/masse_galaxie_hote")
for i in range(1,len(ligne_masse)):
	tab_masse.append(float(ligne_masse[i]))


ligne_covariance= openFichier("/home/johanna/Bureau/fit_SN/snfit_data/terme_diagonaux_matrice_covariance")
for i in range(1,len(ligne_covariance)):
	tab_cov_m_s.append(float(ligne_covariance[i].split(" ")[0]))
	tab_cov_m_c.append(float(ligne_covariance[i].split(" ")[1]))
	tab_cov_s_c.append(float(ligne_covariance[i].split(" ")[2]))





redshift=[]
bmag=[]
x0=[]
x1=[]
c=[]
redshift_err=[]
bmag_err=[]
x0_err=[]
x1_err=[]
c_err=[]

redshift_ar=[]
bmag_ar=[]
x0_ar=[]
x1_ar=[]
c_ar=[]
redshift_err_ar=[]
bmag_err_ar=[]
x0_err_ar=[]
x1_err_ar=[]
c_err_ar=[]
zcmb_ar=[]

ligne_param= openFichier("/home/johanna/Bureau/fit_SN/snfit_data/params")
for i in range(0,len(ligne_param)):
	redshift_ar.append(float(ligne_param[i].split(" ")[0]))
	redshift_err_ar.append(float(ligne_param[i].split(" ")[1]))
	bmag_ar.append(float(ligne_param[i].split(" ")[2]))
	bmag_err_ar.append(float(ligne_param[i].split(" ")[3]))
	x1_ar.append(float(ligne_param[i].split(" ")[4]))
	x1_err_ar.append(float(ligne_param[i].split(" ")[5]))
	c_ar.append(float(ligne_param[i].split(" ")[6]))
	c_err_ar.append(float(ligne_param[i].split(" ")[7]))
	zcmb_ar.append(float(ligne_param[i].split(" ")[8]))


redshift_ar=np.array(zcmb_ar)
bmag_ar=np.array(bmag_ar)
x1_ar=np.array(x1_ar)
c_ar=np.array(c_ar)
redshift_err_ar=np.array(redshift_err_ar)
bmag_err_ar=np.array(bmag_err_ar)
x1_err_ar=np.array(x1_err_ar)
c_err_ar=np.array(c_err_ar)
zcmb_ar=np.array(zcmb_ar)


for i in range(0,len(liste_sn)):
	redshift.append(liste_sn[i].z)
	bmag.append(liste_sn[i].bmax)
	x0.append(liste_sn[i].x0)
	x1.append(liste_sn[i].x1)
	c.append(liste_sn[i].colour)

	redshift_err.append(liste_sn[i].z_err)
	bmag_err.append(liste_sn[i].bmax_err)
	x0_err.append(liste_sn[i].x0_err)
	x1_err.append(liste_sn[i].x1_err)
	c_err.append(liste_sn[i].colour_err)





redshift=np.array(redshift)
bmag=np.array(bmag)
x1=np.array(x1)
c=np.array(c)
redshift_err=np.array(redshift_err)
bmag_err=np.array(bmag_err)
x1_err=np.array(x1_err)
c_err=np.array(c_err)


n=0
diff1=[]
diff2=[]
diff3=[]
diff4=[]
for i in range(0,len(bmag)):
	diff1.append(bmag[i]-bmag_ar[i])
	diff2.append(x1[i]-x1_ar[i])
	diff3.append(c[i]-c_ar[i])
	diff4.append(redshift[i]-redshift_ar[i])
	n=n+1
print n

'''
plt.subplot(2,2,1)
plt.hist(diff1,bins=200,log=True)


plt.subplot(2,2,2)
plt.hist(diff2,bins=200,log=True)


plt.subplot(2,2,3)
plt.hist(diff3,bins=200,log=True)


plt.subplot(2,2,4)
plt.hist(diff4,bins=200,log=True)

plt.show()
exit(1)
'''

eta=np.zeros(3*740)
eta_ar=np.zeros(3*740)

for i in range(0,len(bmag)):
              eta[i*3] = bmag[i]
              eta[i*3+1]=x1[i]
              eta[i*3+2]=c[i]

for i in range(0,len(bmag_ar)):
              eta_ar[i*3] = bmag_ar[i]
              eta_ar[i*3+1]=x1_ar[i]
              eta_ar[i*3+2]=c_ar[i]

params = Parameters()

params.add('alpha', vary=True, value = 0.141, min=0., max = 10.)
params.add('beta', vary=True, value = 3.101, min=0., max = 10.)


params.add('omega_m', vary=True, value=  0.295, min=0.,max=1.)
params.add('omega_l', vary=True, expr='1-omega_m') 
params.add('M', vary=True, value = -19.05, min=-100., max = 100.)
params.add('deltam', vary=False, value = -0.070, min=-100., max = 100.)


#minicov = Minimizer(calculResidu2,params,fcn_args=(eta,tab_masse,tab_cov_m_s,tab_cov_m_c,tab_cov_s_c,redshift,bmag,x0,x1,c,redshift_err,bmag_err,x0_err,x1_err,c_err))
minicov = Minimizer(calculResidu2,params,fcn_args=(eta_ar,tab_masse,tab_cov_m_s,tab_cov_m_c,tab_cov_s_c,zcmb_ar,bmag_ar,x0,x1_ar,c_ar,redshift_err_ar,bmag_err_ar,x0_err,x1_err_ar,c_err_ar))



#m = Minuit(fonctionMinimizer,errordef=1,alpha=0.141,error_alpha=0.001,limit_alpha=(0,10),beta=3.101,error_beta=0.001,limit_beta=(0,10),omega_m=0.3,error_omega_m=0.1,limit_omega_m=(0,1),omega_l=0.7,error_omega_l=0.1,limit_omega_l=(0,1),M=-19.05,limit_M=(-100,100),deltam=-0.04,limit_deltam=(-10,10))
#m.migrad(ncall=100000)
#exit(1)

#resultcov = minicov.minimize(method='nelder')
#resultcov = minicov.minimize(method='slsqp')
#resultcov = minicov.minimize(method='least_squares')
resultcov = minicov.minimize(method=' SLSQP')


#ci = conf_interval(minicov, resultcov)
#lmfit.printfuncs.report_ci(ci)
#resultcov = minicov.nelder(maxfev=10000000)



print resultcov.params.items()

valeurcov=[]
erreur_valeurcov =[]

print ""	

print(resultcov.message)
print "chisqr :", (resultcov.chisqr)
for k,v in resultcov.params.items():
	print k, ' === ', v.value
	valeurcov.append(v.value)
	erreur_valeurcov.append(v.stderr)



alpha_cov=valeurcov[0]
beta_cov=valeurcov[1]
omegam_cov=valeurcov[2]
omegal_cov=valeurcov[3]
M_cov=valeurcov[4]
deltam_cov=valeurcov[5]

alpha_cov_err=erreur_valeurcov[0]
beta_cov_err=erreur_valeurcov[1]
omegam_cov_err=erreur_valeurcov[2]
omegal_cov_err=erreur_valeurcov[3]
M_cov_err=erreur_valeurcov[4]
deltam_cov_err=erreur_valeurcov[5]

print "alpha=",alpha_cov,"+/-",alpha_cov_err
print "beta=",beta_cov,"+/-",beta_cov_err
print "omegal=",omegal_cov,"+/-",omegal_cov_err
print "omegam=",omegam_cov,"+/-",omegam_cov_err
print "M=",M_cov,"+/-",M_cov_err
print "deltaM=",deltam_cov,"+/-",deltam_cov_err


"""
print nbAppel, ")", alpha,beta,omega_m,omega_l,M,deltam
#En gros il faut : 
omega_m = 0.295
omega_l = 1-omega_m
alpha= 0.141
beta = 3.101
M = -19.05
"""

