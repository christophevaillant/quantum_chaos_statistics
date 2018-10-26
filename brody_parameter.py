#Script to calculate the Brody parameter for a list of energy levels or
#resonances using maximum likelihood.
import numpy.random as rnd
import pylab as pl
import scipy.stats as st
import numpy as np
from scipy.integrate import quad
import scipy.special as sp
import scipy.optimize as opt
import scipy.misc as ms

#-----------------------------------------------------
#Read in energy levels
filename= raw_input('Enter the file name: ')
inputfile= open(filename,"r")

boundstates=[]

for line in inputfile:
    boundstates.append(float(line))

boundstates=np.asarray(boundstates)
boundstates=np.sort(boundstates)
boundstates-= min(boundstates)

#Prepare staircase function
staircase=np.arange(0,len(boundstates))

#----------------------------------------------------
#Here we just brute force a large degree polynomial.
#Works okay, but we may need to think a bit harder
#about the functional form.

order= raw_input('Enter the order of the fitting polynomial: ')
coeffs= np.polyfit(boundstates,staircase,order)
print coeffs
zeta=np.polyval(coeffs,boundstates)

#----------------------------------------------------
#----------------------------------------------------
#Get the spacings!
energyspacings=np.zeros(len(boundstates)-1)
spacings= np.zeros(len(zeta)-1)

for i in range(len(zeta)-1):
    spacings[i]= zeta[i+1] - zeta[i]

for i in range(len(boundstates)-1):
    energyspacings[i]= boundstates[i+1] - boundstates[i]

s=np.arange(0.1,10,0.01)
poisson= np.exp(-s)
wigner= 0.5*np.pi*s*np.exp(-0.25*np.pi*s**2)

norm= sum(spacings)
bins= int(np.sqrt(len(spacings)))
print "number of bins: ",bins

#----------------------------------------------------
#----------------------------------------------------
#Get the Brody Parameter

def brodyfunc(s,brodyparam):
    B= sp.gamma((brodyparam+2.0)/(brodyparam+1.0))**(brodyparam+1.0)
    A= B*(brodyparam+1.0)
    p= A*np.power(s,brodyparam)*np.exp(-B*np.power(s,brodyparam+1.0))
    return(p)

def loglikelihood(brodyparam,s):
    P=  np.zeros_like(brodyparam)
    for i in range(len(brodyparam)):
        P[i]= np.sum(np.log(brodyfunc(sortspacings,brodyparam[i])))
    if (len(brodyparam)>1):
        P-=max(P)
    return(-P)

sortspacings= np.sort(spacings)
for i in range(len(sortspacings)):
    if (sortspacings[i]<0.0):
        np.delete(sortspacings,i)

eta= opt.fmin(loglikelihood,0.5,args=(sortspacings,))
dbleprime= ms.derivative(loglikelihood,eta,n=2,dx=0.01,args=(sortspacings,))
var= 1.0/abs(dbleprime)

print "brody parameter is", float(eta)
print "error is ", float(np.sqrt(var))
