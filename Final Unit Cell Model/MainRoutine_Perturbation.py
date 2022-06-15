# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:48:04 2022

@author: aniru

Single or small instance runner of Wedge Unit Cell solver
including perturbation setup
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
FvfList = [0.466,0.224] #[0.622,0.466,0.224,0.069]

Cellsizes = [[1,1,1],[3,1,3]]
Celltypes = [[0.0,0.0,0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0,1.0,1.0]]

LoadList = [[1,1,1],[3],[5,5],[5,5]]
AxisList = [[1,0,2],[2],[1,0],[2,0]]
ClampList = [[0],[2],[4],[4]]
PrescribedDispList = [[-0.00012,0,0],[0.00012],[0.00012,0],[0.00012,0]]



perturbCheck = 1

fibreRadius = 0.0056/2

rhomax = 0.000346355417501517 # (domaincalc(fibreRadius, FvfList[0])/2) - fibreRadius 
# <-- use domain calc method for normal runs
for rhoiter in range(5):
    rhomaxval = (rhomax/4)*(rhoiter)
    for iteration in range(2):
        Cells = Cellsizes[iteration]
        nIntElem = [10,4,4]
        OutputList.append(Celltypes[iteration][0:2])

        for fibreVolumeFraction in FvfList:
        
            squareside = domaincalc(fibreRadius, fibreVolumeFraction)
            # rhomax = squareside/2 - fibreRadius
            # rhomaxval = rhomax/4
            domainLengthY = 0.06 #inches
            domainWidthX = squareside*Cells[0] #inches
            domainHeightZ = squareside*Cells[2] #inches
        
            Dims = [domainWidthX,domainLengthY,domainHeightZ]
            
            Matproplist = [[0.75,0.35],[58,0.2]]
            
            
            if (rhoiter == 0):
                numtests = 1
                perturbCheck = 0
            else:
                numtests = 10
                perturbCheck = 1
                
            for testnum1 in range(numtests):
                print(f'\n{Cells}, rhoiter:{rhoiter}, pertrub?: {perturbCheck}, rhomaxval:{rhomaxval}, testnum1:{testnum1}\n')
                OutList = []
                OutList.append(fibreVolumeFraction)
                testnum=0
                findEL = [0,0,0,0]
                findEL[testnum] = 1
                Loadfaces = LoadList[testnum]
                Clampfaces = ClampList[testnum]
                Loadaxes = AxisList[testnum]
                PrescribedDisps = PrescribedDispList[testnum]
                O1,O2 = solverQ.validation_solver(Cells,nIntElem,fibreRadius,fibreVolumeFraction,Dims,Matproplist,Clampfaces,Loadfaces,Loadaxes,PrescribedDisps,findEL,perturbCheck,rhomaxval)
                OutList.append(O1)
                if (testnum<2):
                    OutList.append(O2)
                
                OutputList.append(OutList)
    
    OutputList.append([2,2])
    OutputList.append([rhoiter,rhomaxval,rhomax,squareside])
    
if (1):
    
    with open(f"CompressiveStiffnessWavyALLfvf_0.000346_MINMAXrho_{nIntElem[0]}x{nIntElem[1]}x{nIntElem[2]}.csv", 'w') as f:
      #f"CompressiveStiffnessWavy{FvfList[0]}_{(rhomaxval*1e3):4}rhomaxk_{nIntElem[0]}x{nIntElem[1]}x{nIntElem[2]}.csv", 'w'
    # using csv.writer method from CSV package
        write = csv.writer(f)
      
        write.writerows(OutputList)
        
    
