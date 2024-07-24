# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:26:01 2018

@author: DNodop
"""

import numpy as np
from scipy import fftpack
#import cv2
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

#size of the Fourier window or the SLM itself
SLMX = 1280     
SLMY = 1024

# Start from Zero
xwidth=SLMX-1
ywidth=SLMY-1

# x and y as arrays
x = np.linspace(0, xwidth, xwidth+1)
y = np.linspace(0, ywidth, ywidth+1)

#Zero of the coordinate system
x0=SLMX/2


#Number of iterations
iter=100

#-------------------------------Start amplitude distribution (SAD) ---------------------------------------
# We use a Gauss function just to improve clarity, 
# but in our source code, it is not really a requirement for proper function.
# An important feature is that we work with an even wave front, 
# which means that we simply leave away any phase term.

# 1/e² width of the start amplitude distribution
omegax=108

# Gauss function
z= np.exp(-2*((x-x0)/omegax)**2) 
                
#Normation to one        
z= z/z.sum()

print("Start amplitude distribution...")

# x- und y-skales are equal
plt.gca().set_aspect("equal") 
f1 = plt.figure()
plt.plot(x, z, color='blue')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show() 


#-------------------------Target amplitude distribution (TAD)---------------------

#This is the width of the gaussians forming the beam array. 
#It is an essential feature for proper function to work wit an 1/e² width of ONE PIXEL!
omegax2=1.0

#Distance of the Gaussians relative to each other
xdist=20

def z1(X):  
    return (
            1.0 *   np.exp(-2*((X-x0-0.0*xdist)/omegax2)**2)  + 
            1.0 *   np.exp(-2*((X-x0+1.0*xdist)/omegax2)**2)  +
            1.0 *   np.exp(-2*((X-x0-1.0*xdist)/omegax2)**2)  + 
            0.0 *   np.exp(-2*((X-x0+2.0*xdist)/omegax2)**2)  +
            0.0 *   np.exp(-2*((X-x0-2.0*xdist)/omegax2)**2)  + 
            0.0 *   np.exp(-2*((X-x0+3.0*xdist)/omegax2)**2)  +
            0.0 *   np.exp(-2*((X-x0-3.0*xdist)/omegax2)**2)  + 
            0
            )

z2=z1(x)     

#Normation to one    
z2= z2/z2.sum()

print("Target Amplitude Distribution...")
f2 = plt.figure()
plt.plot(x, z2, color='red')
plt.xlabel('x')
plt.ylabel('f2(x)')
plt.show() 


#-----------------IFTA-Code----------------------------------------------------

# A = Start Amplitude Distribution (SAD)
A = z 

TargetAmplitude = z2 #Zielverteilung

for i in range(iter):
    print("--- iteration [%d] ---" % i)
    
    #FFT of the SID with flat phase
    B = fftpack.fftshift(fftpack.fft(fftpack.fftshift(A)))
    #fftshift(x[, axes]) 	Shift the zero-frequency component to the center of the spectrum.
    #ifftshift(x[, axes]) 	The inverse of fftshift.
    #fft2(x[, shape, axes, overwrite_x]) 	2-D discrete Fourier transform.
    
    #Normation of the result of the FFT to 1.
    B = B / B.sum()
    
    #Absolute vaule of B
    Babs = np.abs(B)
    
    #Here we move from the amplitude of the electric field to the intensity.
    Ifinal = Babs**2
    Ifinal = Ifinal / Ifinal.sum()
    
    #Phase of the Fourier transformation
    Bang = np.angle(B)  
    
    #Multiply the TAD with the above mentioned Phase
    C = TargetAmplitude * np.exp(1j * Bang)
    
    ##Normation to one
    C = C / C.sum()
    
    #Inverse Fourier transformation
    D = fftpack.fftshift(fftpack.ifft(fftpack.ifftshift(C)))
    
    #Normation to one
    D = D / D.sum()
    
    #Phase of the inverse Fourier transformation
    phiD=np.angle(D)
    
    #Phase is multiplied with the SAD
    A = z * np.exp(1j * phiD)
    
    # This is the most essential part of the code, the adaption of the TID 
    # to improve the beam uniformity
    # It is not working stable under all circumstances, more about this in the meeting
    Z4=TargetAmplitude*Babs
    Z4=Z4/np.max(Z4)
    TargetAmplitude=TargetAmplitude-0.03*Z4
    TargetAmplitude=TargetAmplitude/TargetAmplitude.sum()


#Conversion to gray scale with modulo operation to make it suitable for the spatial light modulator
phimax = 127    
phimod = 255    
#data = np.mod(((phiD/np.pi)+1)*phimax , phimod)
data = phiD-np.min(phiD)

#This is to cut out a slice of the phase, it can be removed if neccessary
cut=150
a=int(SLMX/2-cut)
b=int(SLMX/2+cut)


fig, ax1 = plt.subplots()
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.plot(x[a:b], data[a:b], color='blue', linestyle='solid',linewidth=1.0)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
f=plt.show()
fig.savefig("foo.jpg", bbox_inches='tight', dpi=600)



fig, ax2 = plt.subplots()
ax2.set_xlabel('x')
ax2.set_ylabel('I(x)')
ax2.plot(x[a:b], Ifinal[a:b]/np.max(Ifinal[a:b]), color='blue', linestyle='solid',linewidth=1.0)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
f=plt.show()
fig.savefig("Ifinal.jpg", bbox_inches='tight', dpi=600)


#Convert the phase into a 2D-Image for the spatial light modulator (slm)
x,y = np.meshgrid(x,y)

def z_func(x,y):
 
 return data + 0*y
                        
Z = z_func(x, y) # evaluation of the function on the grid

#Show phase for the slm
plt.figure()
plt.pcolormesh(x, y, Z, cmap='gray')
plt.gca().set_aspect("equal") # x- und y-Skala im gleichen Maßstab
plt.show()

#Store phase for the slm
Z=np.rint(Z)
#cv2.imwrite("Hologramm7xadapneu.pgm", Z)



