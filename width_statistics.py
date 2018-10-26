import numpy as np
import pylab as pl
from scipy.integrate import simps
from scipy.stats import chi2
from scipy.optimize import fmin
from scipy.special import gamma, polygamma

#-----------------------------------------------------
#Initialize some constants
a0= 5.2917721e-11 #m
Eh= 4.359744e-18 #J
amu= 1.660539e-27 #Kg
hbar= 1.0545718e-34 #Js

reducedmass= 84.9673795*amu #174 Yb
C6= 2765.7*Eh*a0**6

abar= 0.5*0.955978*np.power(2.0*reducedmass*C6/hbar**2,0.25)
abar*= 1e10 #in Angstroms
print "abar= ",abar

#-----------------------------------------------------
#function for the maximum likelihood
def loglikelihood(nu,delta,deltabar):
    x= delta/deltabar
    logP= 0.5*nu*np.log(x) - 0.5*x - np.log(deltabar) - 0.5*nu*np.log(2.0) - np.log(gamma(0.5*nu))
    result= np.sum(logP)
    return(-result)

#-----------------------------------------------------
#Read in some data on resonance positions and widths
inputfile= open("resonances_3point_test.dat","r")

B0=[]
widths=[]
abg=[]

for line in inputfile:
    a=line.split()
    B0.append(float(a[0].rstrip(',')))
    widths.append(abs(float(a[1].rstrip(','))))
    abg.append(float(a[2].rstrip(',')))



#-----------------------------------------------------
#Start some plotting
pl.figure(figsize=(3.0,1.8))
rbg=np.asarray(abg)/abar
print rbg

#-----------------------------------------------------
#Change definition of widths we're using to remove threshold effects
widths= 2.0*np.abs(np.asarray(widths)*np.asarray(rbg))/(1.0+(1.0-rbg)**2)
bins= int(np.sqrt(len(widths)))
lnwidths= np.log(widths)
pl.hist(lnwidths,normed=True,bins=bins,color='0.8')

#-----------------------------------------------------
#Define the average, mostly for comparison
lndelta= np.arange(-23,10,1e-2)
delta= np.exp(lndelta)
deltabar=np.average(widths)
stdev=np.std((widths[:])/deltabar)
print deltabar
print stdev

#-----------------------------------------------------
#Work out max likelihood for widths distribution (Chi squared function)
#Also calculate the uncertainty.
P= delta*np.exp(-delta/(2.0*(deltabar)))/np.sqrt(2.0*np.pi*delta*deltabar)
pl.plot(lndelta,P,'g--')
optnu,loc,scaling= chi2.fit(widths,1.0,floc=0.0)#,fscale=deltabar
nuhess= np.zeros((2,2))
nuhess[0,0]= -polygamma(1,0.5*optnu)
nuhess[0,1]= -0.5/scaling
nuhess[1,0]=nuhess[0,1]
nuhess[1,1]= np.sum((0.5*optnu - widths/scaling)/scaling**2)
cov= np.linalg.inv(nuhess)

print "optimized degrees of freedom (mine): ", optnu, "+/-", float(np.sqrt(abs(cov[0,0])))
print "with scaling ", scaling, "+/-", float(np.sqrt(abs(cov[1,1])))

#-----------------------------------------------------
#plot everything and save the plot
P2= chi2.pdf(delta,optnu,loc,scaling)
pl.plot(lndelta, delta*P2,'k-')
pl.ylabel("$P(\mathrm{ln}|\\alpha_\mathrm{bg} \Delta|)$",fontsize=8)
pl.xlabel("$|\\tilde{\Delta}|$ (G)",fontsize=8)
pl.axis([-22.0,6.0,0.0,0.25])
pl.xticks(np.log(np.logspace(-10,3,7)),["$10^{-10}$","$10^{-8}$","$10^{-6}$","$10^{-4}$","$10^{-2}$","$10^{0}$","$10^{2}$"],fontsize=8)
pl.yticks(fontsize=8)

# pl.show()
pl.savefig("resonancesstatsnondecayed.pdf", bbox_inches="tight", dpi=200)
