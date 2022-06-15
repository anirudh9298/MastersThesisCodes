# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 21:39:52 2021

@author: aniru

Multi-element External node mesher

"""

import numpy as np
from numpy import linalg as la

## DOMAIN:
# x_G = 6
# y_G = 10
# z_G = 6

# nCellX = 3
# nCellY = 1
# nCellZ = 3

# L = y_G/nCellY #y
# B = x_G/nCellX  #x
# H = z_G/nCellZ  #z

# # No. of wedges along dimensions
# nL = 10
# nB = 3
# nH = 3

def ExternalMeshing_Quad(x_G,y_G,z_G,nCellX,nCellY,nCellZ,L,B,H,nL,nB,nH):
    
    tNodesX = nB * nCellX + 1
    tNodesZ = nH * nCellZ + 1
    tNodesY = nL * nCellY + 1
    
    Origin = np.array([0.,0.,0.])
    
    tBNodesface = (tNodesX)*(nCellZ+1) + (nCellX + 1)*(nH-1)*nCellZ
    tBNodesLinear = tBNodesface*(nL*nCellY + 1)
    tBNodesQuadratic = tBNodesface*(nL*nCellY)
    tBNodesQ2faceA = (nB*nCellX)*(nCellZ+1)
    tBNodesQ2faceB = (nH*nCellZ)*(nCellX+1)
    
    print(tBNodesQ2faceA,tBNodesQ2faceB)
    tBNodesQuadratic2 = (tBNodesQ2faceA + tBNodesQ2faceB)*(nL*nCellY+1)  #tBNodesface + (nCellX + 1)*nCellZ
    print(tBNodesQuadratic2)
    tBNodes1 = tBNodesLinear + tBNodesQuadratic
    tBNodes2 = tBNodes1 + tBNodesQuadratic2
    
    BoundaryNodesLin = np.zeros([tBNodesLinear,3])
    BoundaryNodesQuad = np.zeros([tBNodesQuadratic,3])
    BoundaryNodesQuad2 = np.zeros([tBNodesQuadratic2,3])
    BoundaryNodesTotal = np.zeros([tBNodes2,3])
    
    countL = 0
    for nY in range(tNodesY):
        y = Origin[1] + (L/nL)*nY
        
        for nZ in range(tNodesZ):
            if (nZ%nH == 0):
                tNodesXr = tNodesX
            else:
                tNodesXr = nCellX + 1
            z = Origin[2] + (H/nH)*nZ
            
            for nX in range (tNodesXr):
                x = Origin[0] + (x_G/(tNodesXr-1))*nX
                BoundaryNodesLin[countL,:] = np.array([x,y,z])
                countL += 1
    
    countQ = 0
    for nY in range(tNodesY-1):
        y = Origin[1] + (L/nL)*nY + (L/nL)*0.5
        
        for nZ in range(tNodesZ):
            if (nZ%nH == 0):
                tNodesXr = tNodesX
            else:
                tNodesXr = nCellX + 1
            z = Origin[2] + (H/nH)*nZ
            
            for nX in range (tNodesXr):
                x = Origin[0] + (x_G/(tNodesXr-1))*nX
                BoundaryNodesQuad[countQ,:] = np.array([x,y,z])
                countQ += 1
    
    countQ2 = 0
    for nY in range(tNodesY):
        y = Origin[1] + (L/nL)*nY

        for nZ in range(nCellZ + 1):
            tNodesXr = tNodesX-1
            z = Origin[2] + (nZ) * H
            xoff = B/nB * 0.5
            xlen = B/nB
            # else:
            #     tNodesXr = nCellX + 1
            #     z = Origin[2] + (H/nH)*nZ - H/nH*0.5
            #     xoff = 0
            #     xlen = B
            
            for nX in range (tNodesXr):
                x = Origin[0] + xlen*nX + xoff
                BoundaryNodesQuad2[countQ2,:] = np.array([x,y,z])
                countQ2 += 1
    
    sbnodeswitch = countQ2
    print(sbnodeswitch)
    for nY in range(tNodesY):
        y = Origin[1] + (L/nL)*nY

        for nZ in range(tNodesZ-1):
            tNodesXr = nCellX + 1
            z = Origin[2] + (H/nH)*nZ + H/nH*0.5
            xoff = 0
            xlen = B
            
            for nX in range (tNodesXr):
                x = Origin[0] + xlen*nX + xoff
                BoundaryNodesQuad2[countQ2,:] = np.array([x,y,z])
                countQ2 += 1
    
    print(countQ2)
    ############## ELEM CONN ############
    
    nCell = nCellX * nCellY * nCellZ
    
    nLayersLin = nL + 1
    nLayersQuad = nL
    nLayersT = nLayersLin + nLayersQuad
    tLayerNodes = 2*(nB) + 2*(nH)
    
    BConnLin = np.zeros([nLayersLin,tLayerNodes,nCell],dtype=int)
    BConnQuad = np.zeros([nLayersQuad,tLayerNodes,nCell],dtype=int)
    BConnQuad2 = np.zeros([nLayersLin,tLayerNodes,nCell],dtype=int)
    BConnT = np.zeros([nLayersT,tLayerNodes,nCell],dtype=int)
    
    cellnum = 0
    for elY in range(nCellY):
        
        for elZ in range(nCellZ):
            
            for elX in range(nCellX):
                
                for layer in range(nLayersLin):
                    
                    conn = np.zeros([tLayerNodes],dtype = int)
                    count = 0
                    seed0 = tBNodesface*(elY*nL+layer) + ((nH-1)*(nCellX+1)+tNodesX)*elZ + (nB)*elX
                    seed1 = seed0 + tNodesX - (elX*(nB-1))
                    seed2 = tBNodesface*(elY*nL+layer) + ((nH-1)*(nCellX+1)+tNodesX)*(elZ+1) + (nB)*elX
                    for i in range(nB+1):
                        conn[i] = seed0 + i
                    count += nB+1
                    
                    for i in range(nH-1):
                        conn[count+i] = seed1 + (nCellX+1)*i + 1
                    count += nH-1
                    
                    for i in range(nB+1):
                        conn[count+i] = seed2 + nB - i
                    count += nB+1
                    
                    for i in range(nH-1):
                        conn[count+i] = seed1 + (nCellX+1)*(nH-2-i)
                    count += nH-1
                    
                    BConnLin[layer,:,cellnum] = conn
                
                cellnum += 1
                    

    cellnum = 0
    for elY in range(nCellY):
        
        for elZ in range(nCellZ):
            
            for elX in range(nCellX):
                
                for layer in range(nLayersQuad):
                    
                    conn = np.zeros([tLayerNodes],dtype = int)
                    count = 0
                    seed0 = tBNodesLinear + tBNodesface*(elY*nL+layer) + ((nH-1)*(nCellX+1)+tNodesX)*elZ + (nB)*elX
                    seed1 = seed0 + tNodesX - (elX*(nB-1))
                    seed2 = tBNodesLinear + tBNodesface*(elY*nL+layer) + ((nH-1)*(nCellX+1)+tNodesX)*(elZ+1) + (nB)*elX
                    for i in range(nB+1):
                        conn[i] = seed0 + i
                    count += nB+1
                    
                    for i in range(nH-1):
                        conn[count+i] = seed1 + (nCellX+1)*i + 1
                    count += nH-1
                    
                    for i in range(nB+1):
                        conn[count+i] = seed2 + nB - i
                    count += nB+1
                    
                    for i in range(nH-1):
                        conn[count+i] = seed1 + (nCellX+1)*(nH-2-i)
                    count += nH-1
                    
                    BConnQuad[layer,:,cellnum] = conn
                
                cellnum += 1
    
    cellnum = 0
    for elY in range(nCellY):
        
        for elZ in range(nCellZ):
            
            for elX in range(nCellX):
                
                for layer in range(nLayersLin):
                    
                    conn = np.zeros([tLayerNodes],dtype = int)
                    count = 0
                    seed0 = tBNodes1 + (elY*nL+layer)*(tBNodesQ2faceA) + (elZ*nB*nCellX) + elX*nB
                    seed3 = tBNodes1 + sbnodeswitch + (elY*nL+layer)*(tBNodesQ2faceB) + elZ*nH*(nCellX+1) + elX
                    seed2 = seed0 + nCellX*nB
                    seed1 = seed3 + 1
                    for i in range(nB):
                        conn[i] = seed0 + i
                    count += nB
                    
                    for i in range(nH):
                        conn[count+i] = seed1 + (nCellX+1)*i
                    count += nH
                    
                    for i in range(nB):
                        conn[count+i] = seed2 + nB-1 - i
                    count += nB
                    
                    for i in range(nH):
                        conn[count+i] = seed3 + (nCellX+1)*(nH-1-i)
                    count += nH
                    
                    BConnQuad2[layer,:,cellnum] = conn
                cellnum += 1
    
    nCentroidsLin = (nCellY*(nL) + 1)*(nCellX*nCellZ)
    print(nCentroidsLin)
    nCentroidsQuad = (nCellY*(nL))*(nCellX*nCellZ)
    CentroidListLin = np.zeros([nCentroidsLin,3])
    CentroidListQuad = np.zeros([nCentroidsQuad,3])
    
    countc = 0        
    for elZ in range(nCellZ):
        z = H*0.5 + elZ*H + Origin[2]
        for elX in range(nCellX):
            x = B*0.5 + elX*B + Origin[0]
            for elY in range(nCellY):
                y0 = L*elY + Origin[1]
                if (elY == nCellY-1):
                    nLr = nL+1
                else:
                    nLr = nL
                for layer in range(nLr):
                    y = y0 + (L/nL)*layer
                    CentroidListLin[countc,:] = [x,y,z]
                    countc+=1

    countc = 0
    for elZ in range(nCellZ):
        z = H*0.5 + elZ*H + Origin[2]
        for elX in range(nCellX):
            x = B*0.5 + elX*B + Origin[0]
            for elY in range(nCellY):
                y0 = L*elY + Origin[1]
                if (elY == nCellY-1):
                    nLr = nL
                else:
                    nLr = nL
                for layer in range(nLr):
                    y = y0 + (L/nL)*(layer + 0.5)
                    CentroidListQuad[countc,:] = [x,y,z]
                    countc+=1
                    
    CentroidConnLin = np.zeros([nCell,nL+1],dtype = int)
    CentroidConnQuad = np.zeros([nCell,nL],dtype = int)
    CentroidConn = np.zeros([nCell,2*nL+1],dtype = int)
    
    cellnum = 0
    for elY in range(nCellY):
        seed0 = elY*(nL)
        for elZ in range(nCellZ):
            seed1 = seed0 + elZ*(nCellX*(nCellY*nL+1))
            for elX in range(nCellX):
                seed2 = seed1 + elX*(nCellY*nL+1)
                for layer in range(nL+1):
                    CentroidConnLin[cellnum,layer] = seed2 + layer
                cellnum+=1
    cellnum = 0
    for elY in range(nCellY):
        seed0 = elY*(nL) + nCentroidsLin
        for elZ in range(nCellZ):
            seed1 = seed0 + elZ*(nCellX*(nCellY*nL))
            for elX in range(nCellX):
                seed2 = seed1 + elX*(nCellY*nL)
                for layer in range(nL):
                    CentroidConnQuad[cellnum,layer] = seed2 + layer
                cellnum+=1
    
    
    # from matplotlib import pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    
    # ax.scatter(BoundaryNodesLin[:,0],BoundaryNodesLin[:,1],BoundaryNodesLin[:,2])
    # ax.scatter(BoundaryNodesQuad[:,0],BoundaryNodesQuad[:,1],BoundaryNodesQuad[:,2])
    return (BoundaryNodesLin,BoundaryNodesQuad,BoundaryNodesQuad2,BoundaryNodesTotal,BConnQuad,BConnLin,BConnT,BConnQuad2,CentroidListLin,CentroidListQuad,CentroidConnLin,CentroidConnQuad)                            
                        
            


# from matplotlib import pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(BoundaryNodesLin[:,0],BoundaryNodesLin[:,1],BoundaryNodesLin[:,2])
# ax.scatter(BoundaryNodesQuad[:,0],BoundaryNodesQuad[:,1],BoundaryNodesQuad[:,2])