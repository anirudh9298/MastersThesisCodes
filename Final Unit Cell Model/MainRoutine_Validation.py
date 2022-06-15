# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 18:57:02 2022

@author: aniru

Looped runner for Caruso Verification

"""
import numpy as np
import csv

import Wedge_Quadratic_UnitCell_Solver as solverQ

def domaincalc(r,kf):
    af = np.pi*r**2
    ad = af/kf
    a = np.sqrt(ad)
    return a

OutputList = []
FvfList = [0.622,0.466,0.224,0.069]
# LoadList = [1,3,5,5] #Face numbers to load and clamp as listed in internal mesher
# ClampList = [0,2,4,4]
# AxisList = [1,2,1,2] # Load Axis
Cellsizes = [[1,1,1],[3,1,3]]
Celltypes = [[0.0,0.0,0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0,1.0,1.0]]

LoadList = [[1],[3],[5,5],[5,5]]
AxisList = [[1],[2],[1,0],[2,0]]
ClampList = [[0],[2],[4],[4]]
PrescribedDispList = [[0.00012],[0.00012],[0.00012,0],[0.00012,0]]

]]

for iteration in range(2):
    Cells = Cellsizes[iteration]
    nIntElem = [10,4,4]
    OutputList.append(Celltypes[iteration])
    fibreRadius = 0.0056/2
    for fibreVolumeFraction in FvfList:
    
        squareside = domaincalc(fibreRadius, fibreVolumeFraction)
        
        domainLengthY = 0.06 #inches
        domainWidthX = squareside*Cells[0] #inches
        domainHeightZ = squareside*Cells[2] #inches
    
        Dims = [domainWidthX,domainLengthY,domainHeightZ]
        
        Matproplist = [[0.75,0.35],[58,0.2]]
        
        
        
        OutList = []
        OutList.append(fibreVolumeFraction)
        for testnum in range(0,4):
            findEL = [0,0,0,0]
            findEL[testnum] = 1
            Loadfaces = LoadList[testnum]
            Clampfaces = ClampList[testnum]
            Loadaxes = AxisList[testnum]
            PrescribedDisps = PrescribedDispList[testnum]
            O1,O2 = solverQ.validation_solver(Cells,nIntElem,fibreRadius,fibreVolumeFraction,Dims,Matproplist,Clampfaces,Loadfaces,Loadaxes,PrescribedDisps,findEL)
            OutList.append(O1)
            if (testnum<2):
                OutList.append(O2)
            
        OutputList.append(OutList)

if (1):
    
    with open(f"CarusoValidation{nIntElem[0]}x{nIntElem[1]}x{nIntElem[2]}.csv", 'w') as f:
      
    # using csv.writer method from CSV package
        write = csv.writer(f)
      
        write.writerows(OutputList)
    
    
                    
                












