# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 11:59:16 2022

@author: aniru

Basic Perturbation Method prepared from Catalanotti and Sebaey
"""
import numpy as np
import math
from math import factorial as fact
import matplotlib
from matplotlib import pyplot as plt

def bezier(t,n,Ps):
    B = np.zeros([3])
    
    for i in range(1,n+1):
        
        coeff = (fact(n)/(fact(i)*fact(n-i))) * (1-t)**(n-i) * (t)**i
        B += coeff * Ps[i-1]
    
    return B

def Controlpts(n):
    
    Ps = np.zeros([n,3])
    Ps[1,0] = 1
    Ps[2,0] = 2
    Ps[1,1] = 2
    
    Ts = np.arange(0,1.05,0.05)
    
    Curve = np.zeros([Ts.shape[0],3])
    
    for pt,t in enumerate(Ts):
        Curve[pt,:] = bezier(t,n,Ps)
        
    plt.plot(Curve[:,0],Curve[:,1])
    plt.plot(Ps[:,0],Ps[:,1])

def fibrePerturb(n,Points,fvf,fibreRadius,Dims,rhomaxval):
        for i in range(n):
            P = Points[i,:]
            uk = stochasticDisp(n, Points, fvf, fibreRadius, Dims,rhomaxval)
            P[0] += uk[1]
            P[2] += uk[2]
            P[1] += uk[0]
            Points[i,:] = P
        return Points

def stochasticDisp(n,Points,fvf,fibreRadius,Dims,rhomaxval):
        uk = np.zeros([3])
        rand1 = np.random.ranf()
        thetak = 2*np.pi*rand1
        
        lambdak = np.random.ranf()
        rhok = lambdak*rhomaxk(0, n, Points,rhomaxval)
        
        uk[0] = 0
        uk[1] = rhok*np.cos(thetak)
        uk[2] = rhok*np.sin(thetak)
        
        return uk
    
def rhomaxk(Dims,n,Points,rhomaxval):
    
    rhomaxk = rhomaxval-0.00001
    return rhomaxk