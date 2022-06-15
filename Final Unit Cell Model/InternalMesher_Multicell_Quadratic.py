# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 12:23:06 2021

@author: aniru

Multi Cell Quadratic Internal mesher

Requires a seeded global boundary node list and centroid list (including quad)
Outputs a global node coordinate list with a global connectivity matrix

"""
import ExternalMesher_Multicell_Quadratic as exmesh
import Perturbation_Random as perturb
import numpy as np
from numpy import linalg as la

#### FUNCTIONS

def splitlayers(LayerConn,Centroids,module):
    L1 = LayerConn[module,:]
    L2 = LayerConn[module+1,:]
    L1c = Centroids[module]
    L2c = Centroids[module+1]
    L1 = np.concatenate((L1, np.array([L1[0]])),axis=0)
    L2 = np.concatenate((L2, np.array([L2[0]])),axis=0)
    return L1, L2, L1c, L2c

def distance(A,B):
    return la.norm(A-B)

def section(A,B,r):
    D = A-B
    R = r*D + B
    return R
    
def fibrenodes(BoundaryNodeCoords,L0,L0c,tLayerNodes,fibreRadius):
    INodeCoords = np.zeros([tLayerNodes,3])
    nodesadded = 0
    for n in range(tLayerNodes):
        Node = BoundaryNodeCoords[L0[n],:]
        Cent = L0c[:]
        dist = distance(Node,Cent)
        ratio = fibreRadius/dist
        NewNode = section(Node,Cent,ratio)
        INodeCoords[nodesadded,:] = NewNode
        nodesadded += 1
    return INodeCoords, nodesadded



def quadnodes1(nLayers,tLayerNodes,Centroids,CentroidListLin,seed0,pointer):
    global GlobalNodeCoords
    count = 0
    Coords = np.zeros([nLayers*tLayerNodes,3])
    for layer in range (nLayers):
        centre = CentroidListLin[Centroids[layer],:]
        for j in range (tLayerNodes):
            fnode = GlobalNodeCoords[seed0+count,:]
            irnode = section(fnode,centre,0.5)
            Coords[count,:] = irnode
            count += 1
    irnodes = count
    return(Coords,irnodes)

def quadnodes2(nLayers,tLayerNodes,Centroids,CentroidListLin,seed0,pointer,fibreRadius):
    global GlobalNodeCoords
    count = 0
    Coords = np.zeros([nLayers*tLayerNodes,3])
    for layer in range (nLayers):
        centre = CentroidListLin[Centroids[layer],:]
        for j in range (tLayerNodes):
            if (j != tLayerNodes-1):
                fnode1 = GlobalNodeCoords[seed0+count,:]
                fnode2 = GlobalNodeCoords[seed0+1+count,:]
            else:
                fnode1 = GlobalNodeCoords[seed0+count,:]
                fnode2 = GlobalNodeCoords[seed0+1+count-tLayerNodes,:]
            
            fnodem = section(fnode1,fnode2,0.5)
            ornode = section(fnodem,centre,fibreRadius/distance(fnodem,centre))
            Coords[count,:] = ornode
            count += 1
    ornodes = count
    return(Coords,ornodes)

def quadnodes3(nLayers,tLayerNodes,LayerConn,seed0,pointer):
    global GlobalNodeCoords
    count = 0
    Coords = np.zeros([nLayers*tLayerNodes,3])
    for layer in range (nLayers):
        L0 = LayerConn[layer,:]
        for j in range(tLayerNodes):
            mnode1 = GlobalNodeCoords[seed0+count,:]
            mnode2 = GlobalNodeCoords[L0[j],:]
            pmnode = section(mnode1,mnode2,0.5)
            Coords[count,:] = pmnode
            count += 1
    pmnodes = count
    return(Coords,pmnodes)
    
def quadnodes4(nLayers,tLayerNodes,LayerConn,seed0,pointer):
    global GlobalNodeCoords
    count = 0
    Coords = np.zeros([nLayers*tLayerNodes,3])
    for layer in range(nLayers):
        L0 = LayerConn[layer,:]
        for j in range(tLayerNodes):
            if (j != tLayerNodes-1):
                sq0 = np.array([L0[j], L0[j+1], seed0+j+1+layer*tLayerNodes, seed0+j+layer*tLayerNodes], dtype = int)
            else:
                sq0 = np.array([L0[j], L0[0], seed0+0+layer*tLayerNodes, seed0+j+layer*tLayerNodes], dtype = int)
                
            if ((distance(GlobalNodeCoords[sq0[0]],GlobalNodeCoords[sq0[2]]))>(distance(GlobalNodeCoords[sq0[1]],GlobalNodeCoords[sq0[3]]))):
                sl = [0,2] #split line
            else:
                sl = [1,3]
            mnode1 = GlobalNodeCoords[sq0[sl[0]]]
            mnode2 = GlobalNodeCoords[sq0[sl[1]]]
            smnode = section(mnode1,mnode2,0.5)
            Coords[count,:] = smnode
            count += 1
    smnodes = count
    return(Coords,smnodes)

def quadnodes5(nLayers,tLayerNodes,seed0,pointer):
    global GlobalNodeCoords
    count = 0
    Coords = np.zeros([nLayers*tLayerNodes,3])
    for layer in range(nLayers):
        for i in range(tLayerNodes):
            fnode1 = GlobalNodeCoords[seed0+count,:]
            fnode2 = GlobalNodeCoords[seed0+tLayerNodes+count,:]
            fmnode = section(fnode1,fnode2,0.5)
            Coords[count,:] = fmnode
            count += 1
    fmnodes = count
    return(Coords,fmnodes)
    

def updateGlobal(Array):
    global GlobalNodeCoords, gPos
    length = Array.shape[0]
    GlobalNodeCoords[gPos:gPos+length,:] = Array
    gPos += length

def updateGlobalEl(Array,n,tag):
    global GlobalElemConn, GlobalElemTag, gEl, gEq
    l = Array.shape[0]
    if (n==6):
        GlobalElemConn[gEl:gEl+l,0:6] = Array
        GlobalElemTag[gEl:gEl+l] = tag
        gEl = gEl + l
    else:
        GlobalElemConn[gEq:gEq+l,6:15] = Array
        gEq = gEq + l
    
def findFaceNodes(axis,val):
    global GlobalNodeCoords
    global gPos
    l = []
    for i in range(gPos):
        if (GlobalNodeCoords[i,axis] == val):
            l.append(i)
    return l



##### INPUTs
def InternalMesh(quadElems,Cells,nIntElem,fibreRadius,kf,Dims,perturbCheck,rhomaxval):
    global GlobalNodeCoords
    global GlobalElemConn
    global GlobalElemTag
    global gPos
    global gEl
    global gEq
    
    # Cells,[nL,nB,nH],fibreRadius,kf,Dims = inp.inputs()
    nL,nB,nH = nIntElem
    # fibreRadius = 0.0056/2 #0.8
    # kf = 0.622

    nCellX,nCellY,nCellZ = Cells

    # nCellX = 1
    # nCellY = 1
    # nCellZ = 1
    nCells = nCellX*nCellY*nCellZ
    
    # x_G = squareside*nCellX #2
    # y_G = 0.01#10
    # z_G = squareside*nCellZ #2
    
    x_G,y_G,z_G = Dims

    L = y_G/nCellY #y
    B = x_G/nCellX  #x
    H = z_G/nCellZ  #z
    
    # No. of wedges along dimensions
    # nL = 20
    # nB = 2
    # nH = 2
    
    vfF = (np.pi*fibreRadius**2)/(B*H)
    
    BoundaryNodesLin,BoundaryNodesQuad,BoundaryNodesQuad2,BoundaryNodesTotal,BConnQuad, BConnLin, \
    BConnT,BConnQuad2,CentroidListLin,CentroidListQuad,CentroidConnLin,CentroidConnQuad \
        = exmesh.ExternalMeshing_Quad(x_G, y_G, z_G, nCellX, nCellY, nCellZ, L, B, H, nL, nB, nH)
    

    GlobalNodeCoords = np.zeros([30000,3])
    GlobalElemConn = np.zeros([10000,15],dtype = int)
    GlobalElemTag = np.zeros([10000],dtype=int)
    gPos = 0
    gEl = 0
    gEq = 0
    
    tBNodesLin = BoundaryNodesLin.shape[0]
    tBNodesQuad = BoundaryNodesQuad.shape[0]
    tBNodesQuad2 = BoundaryNodesQuad2.shape[0]
    tCentroidsLin = CentroidListLin.shape[0]
    tCentroidsQuad = CentroidListQuad.shape[0]
    
    
    if (perturbCheck):
        CentroidListLin = perturb.fibrePerturb(tCentroidsLin, CentroidListLin, vfF, fibreRadius, [L,B,H],rhomaxval)
        if (quadElems):
            CentroidListQuad = perturb.fibrePerturb(tCentroidsQuad,CentroidListQuad,vfF,fibreRadius,[L,B,H],rhomaxval)
    
    if (quadElems):
        tBoundNodes = tBNodesLin + tBNodesQuad + tBNodesQuad2
    else:
        tBoundNodes = tBNodesLin

    
    updateGlobal(BoundaryNodesLin)
    
    if (quadElems):
        updateGlobal(BoundaryNodesQuad)
        updateGlobal(BoundaryNodesQuad2)
        
    updateGlobal(CentroidListLin)
    
    if (quadElems):
        updateGlobal(CentroidListQuad)
        
    Faces = []
    #################### START OF INTERIOR NODES ##################################
    intstart = gPos
    seedC = intstart
    
    
    for cell in range(nCells): #nCells
        
        LayerConn = BConnLin[:,:,cell]
        tLayerNodes = LayerConn.shape[1]
        Centroids = CentroidConnLin[cell,:]
        QCentroids = CentroidConnQuad[cell,:]
        LayerConnQ = BConnQuad[:,:,cell]
        LayerConnQ2 = BConnQuad2[:,:,cell]
        # Make fibre Nodes
        for layer in range(nL+1):
            
            L0 = LayerConn[layer,:]
            L0c = CentroidListLin[Centroids[layer]]
            INodeCoords, nodesadded = fibrenodes(BoundaryNodesLin,L0,L0c,tLayerNodes,fibreRadius)
            updateGlobal(INodeCoords)
        
        # Connectivity for fibre wedges
        
        ElConnFLin = np.zeros([tLayerNodes*nL,6], dtype=int)
        ElConnMLin = np.zeros([tLayerNodes*nL*2,6], dtype=int)
    
        for module in range(nL):
            
            #L1,L2,L1c,L2c = splitlayers(LayerConn,Centroids,module)
            for elem in range(tLayerNodes):
                conn = np.zeros([6],dtype=int)
                
                if (elem != tLayerNodes-1):
                    conn[0] = Centroids[module] + tBoundNodes
                    conn[2] = seedC + elem + module*tLayerNodes
                    conn[1] = seedC + elem + 1 + module*tLayerNodes
                    conn[3] = Centroids[module+1] + tBoundNodes
                    conn[5] = seedC + elem + (module+1)*tLayerNodes
                    conn[4] = seedC + elem + 1 + (module+1)*tLayerNodes
                else:
                    conn[0] = Centroids[module] + tBoundNodes
                    conn[2] = seedC + elem + module*tLayerNodes
                    conn[1] = seedC + 0 + module*tLayerNodes
                    conn[3] = Centroids[module+1] + tBoundNodes
                    conn[5] = seedC + elem + (module+1)*tLayerNodes
                    conn[4] = seedC + 0 + (module+1)*tLayerNodes
                    
                ElConnFLin[module*tLayerNodes+elem,:] = conn
        updateGlobalEl(ElConnFLin,6,1)
        #seedC = gPos  ############# IMPORTANT, NEEDED FOR NEXT CELL ##################
        
        
        for module in range(nL):
            
            L1,L2,L1c,L2c = splitlayers(LayerConn,Centroids,module)
            
            for elem in range(tLayerNodes):
                if (elem != tLayerNodes-1):
                    sq1 = np.array([L1[elem], L1[elem+1], seedC+elem+1+module*tLayerNodes, seedC+elem+module*tLayerNodes], dtype = int)
                    sq2 = np.array([L2[elem], L2[elem+1], seedC+elem+(module+1)*tLayerNodes+1, seedC+elem+(module+1)*tLayerNodes], dtype = int)
                else:
                    sq1 = np.array([L1[elem], L1[elem+1], seedC+0+module*tLayerNodes, seedC+elem+module*tLayerNodes], dtype = int)
                    sq2 = np.array([L2[elem], L2[elem+1], seedC+0+(module+1)*tLayerNodes, seedC+elem+(module+1)*tLayerNodes], dtype = int)
                # brickconn = np.concatenate((sq1,sq2),axis=0)
                if ((distance(GlobalNodeCoords[sq1[0]],GlobalNodeCoords[sq1[2]]))>(distance(GlobalNodeCoords[sq1[1]],GlobalNodeCoords[sq1[3]]))):
                    sl = [0,2] #split line
                    sc = [1,3] #split corner nodes (not shared with split element)
                else:
                    sl = [1,3]
                    sc = [2,0]
                
                conn1 = np.zeros([6],dtype=int)
                conn2 = np.zeros([6],dtype=int)
                
                conn1[0] = sq1[sc[0]]
                conn1[1] = sq1[sl[0]]
                conn1[2] = sq1[sl[1]]
                conn1[3] = sq2[sc[0]]
                conn1[4] = sq2[sl[0]]
                conn1[5] = sq2[sl[1]]
                
                conn2[0] = sq1[sc[1]]
                conn2[1] = sq1[sl[1]]
                conn2[2] = sq1[sl[0]]
                conn2[3] = sq2[sc[1]]
                conn2[4] = sq2[sl[1]]
                conn2[5] = sq2[sl[0]]
                
                ElConnMLin[module*(2*tLayerNodes)+ 2*elem,:] = conn1
                ElConnMLin[module*(2*tLayerNodes)+ 2*elem + 1,:] = conn2
        updateGlobalEl(ElConnMLin,6,0)
        
        if (quadElems):
            #hari om#
            QList = []
            QList.append(gPos)
            # INNER RING NODES 0-1
            TempCoords,nPos = quadnodes1(nL+1,tLayerNodes,Centroids,CentroidListLin,seedC,0)
            updateGlobal(TempCoords)
            QList.append(gPos)
            # OUTER RING NODES 1-2
            TempCoords,nPos = quadnodes2(nL+1,tLayerNodes,Centroids,CentroidListLin,seedC,0,fibreRadius)
            updateGlobal(TempCoords)
            QList.append(gPos)
            #PRIMARY MATRIX NODES 2-3
            TempCoords,nPos = quadnodes3(nL+1, tLayerNodes, LayerConn, seedC, 0)
            updateGlobal(TempCoords)
            QList.append(gPos)
            #SPLIT MATRIX NODES 3-4
            TempCoords,nPos = quadnodes4(nL+1, tLayerNodes, LayerConn, seedC, 0)
            updateGlobal(TempCoords)
            QList.append(gPos)
            #FIBRE MID NODES 4-5
            TempCoords,nPos = quadnodes5(nL, tLayerNodes, seedC, 0)
            updateGlobal(TempCoords)
            QList.append(gPos)
            
            #PLOT the coords
            from matplotlib import pyplot as plt
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            
            # ax.scatter(GlobalNodeCoords[QList[0]:QList[1],0],GlobalNodeCoords[QList[0]:QList[1],1],GlobalNodeCoords[QList[0]:QList[1],2])
            # ax.scatter(GlobalNodeCoords[QList[1]:QList[2],0],GlobalNodeCoords[QList[1]:QList[2],1],GlobalNodeCoords[QList[1]:QList[2],2])
            # ax.scatter(GlobalNodeCoords[QList[2]:QList[3],0],GlobalNodeCoords[QList[2]:QList[3],1],GlobalNodeCoords[QList[2]:QList[3],2])
            # ax.scatter(GlobalNodeCoords[QList[3]:QList[4],0],GlobalNodeCoords[QList[3]:QList[4],1],GlobalNodeCoords[QList[3]:QList[4],2])
            # ax.scatter(GlobalNodeCoords[QList[4]:QList[5],0],GlobalNodeCoords[QList[4]:QList[5],1],GlobalNodeCoords[QList[4]:QList[5],2])
            
            ######## QUADRATIC CONNECTIVITY ##############
            ElConnFQuad = np.zeros([tLayerNodes*nL,9], dtype=int)
            ElConnMQuad = np.zeros([tLayerNodes*nL*2,9], dtype=int)
            
                ## FIBRE QUADRATIC CONNECTIVITY
            
            for module in range (nL):
                for elem in range (tLayerNodes):
                    elnum = module*tLayerNodes+elem
                    qconn = np.zeros([9],dtype=int)
                    
                    if (elem != tLayerNodes-1):
                        
                        qconn[2] = QList[0] + elnum
                        qconn[1] = QList[1] + elnum
                        qconn[0] = QList[0] + elnum + 1
                        
                        qconn[5] = QList[0] + tLayerNodes + elnum
                        qconn[4] = QList[1] + tLayerNodes + elnum
                        qconn[3] = QList[0] + tLayerNodes + elnum + 1
                        
                        qconn[6] = QCentroids[module] + tBoundNodes
                        qconn[8] = QList[4] + elnum
                        qconn[7] = QList[4] + elnum + 1
                    else:
                        
                        qconn[2] = QList[0] + elnum
                        qconn[1] = QList[1] + elnum
                        qconn[0] = QList[0] + elnum + 1 - tLayerNodes
                        
                        qconn[5] = QList[0] + tLayerNodes + elnum
                        qconn[4] = QList[1] + tLayerNodes + elnum
                        qconn[3] = QList[0] + tLayerNodes + elnum + 1 - tLayerNodes
                        
                        qconn[6] = QCentroids[module] + tBoundNodes
                        qconn[8] = QList[4] + elnum
                        qconn[7] = QList[4] + elnum + 1 - tLayerNodes
                    
                    ElConnFQuad[elnum,:] = qconn
                    
            updateGlobalEl(ElConnFQuad, 9, 1)
                    
            
            for module in range(nL):
                
                L1,L2,L1c,L2c = splitlayers(LayerConn,Centroids,module)
                L1Q = LayerConnQ[module,:]
                L1Q2 = LayerConnQ2[module,:]
                L2Q2 = LayerConnQ2[module+1,:]
                for elem in range(tLayerNodes):
                    elnum = module*tLayerNodes+elem
                    if (elem != tLayerNodes-1):
                        sq1 = np.array([L1[elem], L1[elem+1], seedC+elem+1+module*tLayerNodes, seedC+elem+module*tLayerNodes], dtype = int)
                        sq2 = np.array([L2[elem], L2[elem+1], seedC+elem+(module+1)*tLayerNodes+1, seedC+elem+(module+1)*tLayerNodes], dtype = int)
                        qsq1 = np.array([QList[2]+elnum, L1Q2[elem], QList[2]+elnum+1, QList[1]+elnum, QList[3]+elnum],dtype=int)
                        qsq2 =  np.array([QList[2]+elnum+tLayerNodes, L2Q2[elem], QList[2]+elnum+tLayerNodes+1, QList[1]+elnum+tLayerNodes, QList[3]+elnum+tLayerNodes],dtype=int)
                        qsqm = np.array([L1Q[elem], L1Q[elem+1], QList[4]+elnum+1, QList[4]+elnum],dtype=int)
                    else:
                        sq1 = np.array([L1[elem], L1[elem+1], seedC+0+module*tLayerNodes, seedC+elem+module*tLayerNodes], dtype = int)
                        sq2 = np.array([L2[elem], L2[elem+1], seedC+0+(module+1)*tLayerNodes, seedC+elem+(module+1)*tLayerNodes], dtype = int)
                        qsq1 = np.array([QList[2]+elnum, L1Q2[elem], QList[2]+elnum+1-tLayerNodes, QList[1]+elnum, QList[3]+elnum],dtype=int)
                        qsq2 = np.array([QList[2]+elnum+tLayerNodes, L2Q2[elem], QList[2]+elnum+tLayerNodes+1-tLayerNodes, QList[1]+elnum+tLayerNodes, QList[3]+elnum+tLayerNodes],dtype=int)
                        qsqm =np.array([L1Q[elem], L1Q[elem+1-tLayerNodes], QList[4]+elnum+1-tLayerNodes, QList[4]+elnum],dtype=int)
                    # brickconn = np.concatenate((sq1,sq2),axis=0)
                    if ((distance(GlobalNodeCoords[sq1[0]],GlobalNodeCoords[sq1[2]]))>(distance(GlobalNodeCoords[sq1[1]],GlobalNodeCoords[sq1[3]]))):
                        sl = [0,2] #split line
                        sc = [1,3] #split corner nodes (not shared with split element)
                        qsa = [1,4,2]
                        qsb = [3,4,0]
                    else:
                        sl = [1,3]
                        sc = [2,0]
                        qsa = [2,4,3]
                        qsb = [0,4,1]
                    
                    qconn1 = np.zeros([9],dtype=int)
                    qconn2 = np.zeros([9],dtype=int)
                    # QelConnMatrix[module*(2*tLayerNodes)+ 2*elem,0:6] = ElConnMatrix[module*(2*tLayerNodes)+ 2*elem,:]
                    # QelConnMatrix[module*(2*tLayerNodes)+ 2*elem + 1,0:6] = ElConnMatrix[module*(2*tLayerNodes)+ 2*elem + 1,:]
                    
                    qconn1[0] = qsq1[qsa[0]]
                    qconn1[1] = qsq1[qsa[1]]
                    qconn1[2] = qsq1[qsa[2]]
                    qconn1[3] = qsq2[qsa[0]]
                    qconn1[4] = qsq2[qsa[1]]
                    qconn1[5] = qsq2[qsa[2]]
                    qconn1[6] = qsqm[sc[0]]
                    qconn1[7] = qsqm[sl[0]]
                    qconn1[8] = qsqm[sl[1]]
                    
                    qconn2[0] = qsq1[qsb[0]]
                    qconn2[1] = qsq1[qsb[1]]
                    qconn2[2] = qsq1[qsb[2]]
                    qconn2[3] = qsq2[qsb[0]]
                    qconn2[4] = qsq2[qsb[1]]
                    qconn2[5] = qsq2[qsb[2]]
                    qconn2[6] = qsqm[sc[1]]
                    qconn2[7] = qsqm[sl[1]]
                    qconn2[8] = qsqm[sl[0]]
                
                
                    ElConnMQuad[module*(2*tLayerNodes)+ 2*elem,:] = qconn1
                    ElConnMQuad[module*(2*tLayerNodes)+ 2*elem + 1,:] = qconn2
            
            updateGlobalEl(ElConnMQuad, 9, 0)
        
        seedC = gPos
        # gEq = gEl
        
        
    import WedgePost as post
    
    if (quadElems):
    
        jobname = f"UnitCellFiles/ReportMC{perturbCheck}_UnitCell_{nCellX}x{nCellZ}x{nCellY}_Complete_Quad_{int(nB)}x{int(nH)}x{int(nL)}"
        #f"UnitCellFiles/WavyCompression{perturbCheck}_UnitCell_{nCellX}x{nCellZ}x{nCellY}_Complete_Quad_{int(nB)}x{int(nH)}x{int(nL)}"
        nodes_per_elem = 15
        eltype = 26
        Xout = np.zeros(6)
        StrainNode = 0
        StressNode = 0
        ptdat = 0
        
        C = GlobalNodeCoords[0:gPos,:]
        E = GlobalElemConn[0:gEl,0:15]
        Etag = GlobalElemTag[0:gEl]
        post.vtk_output_format(jobname+'_meshQ', C, C.shape[0], E, E.shape[0], nodes_per_elem, eltype, Xout, StrainNode, StressNode, ptdat)
    
    else:
    
        jobname = f"UnitCellFiles/UnitCell_{nCellX}x{nCellZ}x{nCellY}_Complete_Lin_{int(nB)}x{int(nH)}x{int(nL)}"
        
        nodes_per_elem = 6
        eltype = 13
        Xout = np.zeros(6)
        StrainNode = 0
        StressNode = 0
        ptdat = 0
        
        C = GlobalNodeCoords[0:gPos,:]
        E = GlobalElemConn[0:gEl,0:nodes_per_elem]
        Etag = GlobalElemTag[0:gEl]
        post.vtk_output_format(jobname+'_meshL', C, C.shape[0], E, E.shape[0], nodes_per_elem, eltype, Xout, StrainNode, StressNode, ptdat)
    # Faces along Length     
    Faces.append(findFaceNodes(1, 0))
    Faces.append(findFaceNodes(1, y_G))
    # Faces along Height
    Faces.append(findFaceNodes(2, 0))
    Faces.append(findFaceNodes(2, z_G))
    # Faces along Width
    Faces.append(findFaceNodes(0, 0))
    Faces.append(findFaceNodes(0, x_G))
    return (C,E,Etag,Faces,jobname,Dims,Cells)