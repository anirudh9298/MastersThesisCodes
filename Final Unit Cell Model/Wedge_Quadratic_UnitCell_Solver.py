# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 23:19:01 2021
Keep it going....
@author: aniru

Wedge Element - Quadratic
"""
'''_______________________________________________
                QUADRATIC
   _______________________________________________'''
import numpy as np
from numpy import linalg as la

from scipy.sparse import lil_matrix # sparse matrix format as linked list
import scipy.linalg as sla
from pypardiso import spsolve #Intel MKL Pardiso parrallel(shared memory) sparse matrix solver

from time import time

def shapefuns(g,h,r):
    
    N = np.array([0.5*((1-g-h)*(2*(1-g-h)-1)*(1-r)-(1-g-h)*(1-r**2)),
                  0.5*(g*(2*g-1)*(1-r)-g*(1-r**2)),
                  0.5*(h*(2*h-1)*(1-r)-h*(1-r**2)),
                  0.5*((1-g-h)*(2*(1-g-h)-1)*(1+r)-(1-g-h)*(1-r**2)),
                  0.5*(g*(2*g-1)*(1+r)-g*(1-r**2)),
                  0.5*(h*(2*h-1)*(1+r)-h*(1-r**2)),
                  2.0*(1-g-h)*g*(1-r),
                  2.0*g*h*(1-r),
                  2.0*(1-g-h)*h*(1-r),
                  2.0*(1-g-h)*g*(1+r),
                  2.0*g*h*(1+r),
                  2.0*(1-g-h)*h*(1+r),
                  1.0*(1-g-h)*(1-r**2),
                  1.0*g*(1-r**2),
                  1.0*h*(1-r**2)])
    
    dNg = np.array([1/2 - (r - 1)*(g + h - 1) - r**2/2 - ((r - 1)*(2*g + 2*h - 1))/2,
                    r**2/2 - g*(r - 1) - ((2*g - 1)*(r - 1))/2 - 1/2,
                    0,
                    ((r + 1)*(2*g + 2*h - 1))/2 + (r + 1)*(g + h - 1) - r**2/2 + 1/2,
                    ((2*g - 1)*(r + 1))/2 + g*(r + 1) + r**2/2 - 1/2,
                    0,
                    (r - 1)*(2*g + 2*h - 2) + 2*g*(r - 1),
                    -2*h*(r - 1),
                    2*h*(r - 1),
                    - (r + 1)*(2*g + 2*h - 2) - 2*g*(r + 1),
                    2*h*(r + 1),
                    -2*h*(r + 1),
                    r**2 - 1,
                    1 - r**2,
                    0])
    
    dNh = np.array([1/2 - (r - 1)*(g + h - 1) - r**2/2 - ((r - 1)*(2*g + 2*h - 1))/2,
                    0,
                    r**2/2 - h*(r - 1) - ((2*h - 1)*(r - 1))/2 - 1/2,
                    ((r + 1)*(2*g + 2*h - 1))/2 + (r + 1)*(g + h - 1) - r**2/2 + 1/2,
                    0,
                    ((2*h - 1)*(r + 1))/2 + h*(r + 1) + r**2/2 - 1/2,
                    2*g*(r - 1),
                    -2*g*(r - 1),
                    (r - 1)*(2*g + 2*h - 2) + 2*h*(r - 1),
                    -2*g*(r + 1),
                    2*g*(r + 1),
                    - (r + 1)*(2*g + 2*h - 2) - 2*h*(r + 1),
                    r**2 - 1,
                    0,
                    1 - r**2])
    
    dNr = np.array([- ((g + h - 1)*(2*g + 2*h - 1))/2 - r*(g + h - 1),
                    g*r - (g*(2*g - 1))/2,
                    h*r - (h*(2*h - 1))/2,
                    ((g + h - 1)*(2*g + 2*h - 1))/2 - r*(g + h - 1),
                    g*r + (g*(2*g - 1))/2,
                    h*r + (h*(2*h - 1))/2,
                    g*(2*g + 2*h - 2),
                    -2*g*h,
                    h*(2*g + 2*h - 2),
                    -g*(2*g + 2*h - 2),
                    2*g*h,
                    -h*(2*g + 2*h - 2),
                    2*r*(g + h - 1),
                    -2*g*r,
                    -2*h*r])
    
    
    return N, dNg, dNh, dNr


def Mapping(xa,g,h,r):
    
    N,dNg,dNh,dNr = shapefuns(g,h,r)
    x = N[None,:] @ xa
    
    return x

def ParamCoords():
    X = np.array([[0,0,-1],
                  [1,0,-1],
                  [0,1,-1],
                  [0,0,1],
                  [1,0,1],
                  [0,1,1],
                  [0.5,0,-1],
                  [0.5,0.5,-1],
                  [0,0.5,-1],
                  [0.5,0,1],
                  [0.5,0.5,1],
                  [0,0.5,1],
                  [0,0,0],
                  [1,0,0],
                  [0,1,0]],dtype = float)
    for i in range(X.shape[0]):
        pt = X[i,:]
        g = pt[0]
        h = pt[1]
        r = pt[2]
        N,dNg,dNh,dNr = shapefuns(g,h,r)
        for j in range(X.shape[0]):
            if j==i:
                print(N[j],j)
            elif N[j]!=0:
                print(f"Error i={i} j={j}, N[j] = {N[j]}\n")
            else:
                print('ok')
    return X
                
                
'''_______________________________________________
                QUADRATIC
   _______________________________________________'''        
                
        

def Material(Matproplist,tag):
    
    E = Matproplist[tag][0]
    nu = Matproplist[tag][1]
    
    G = E/(2*(1+nu))
    lam = nu*E/((1+nu)*(1-2*nu))
    Cmat = np.array([[2*G+lam, lam, lam, 0, 0, 0],\
                     [lam, 2*G+lam, lam, 0, 0, 0],\
                     [lam, lam, 2*G+lam, 0, 0, 0],\
                     [0, 0, 0, G, 0, 0],\
                     [0, 0, 0, 0, G, 0],\
                     [0, 0, 0, 0, 0, G]])
    return Cmat


def Jacobian(elnum, NodeCoords, N, dNg, dNh, dNr, errF):
    
    localderivmat = np.array([dNg,dNh,dNr])
    J = localderivmat @ NodeCoords
    detJ = la.det(J)
    if (detJ <= 0):
        print("invalid Jacobian",elnum,detJ)       #ngp
    Jinv = la.inv(J)
    globalderivmat = Jinv @ localderivmat
    dNx = globalderivmat[0,:]
    dNy = globalderivmat[1,:]
    dNz = globalderivmat[2,:]
    
    return detJ,dNx,dNy,dNz,J,Jinv



def Elmat(elnum,NodeCoords,N,dNx,dNy,dNz,nodes,dofn):
    B = np.zeros([6,nodes*dofn])  # 6nodes per element
    for i in range(nodes): #6 nodes
        Bi = np.array([[dNx[i],0,0],
                       [0,dNy[i],0],
                       [0,0,dNz[i]],
                       [0,dNz[i],dNy[i]],
                       [dNz[i],0,dNx[i]],
                       [dNy[i],dNx[i],0]])
        B[:,i*dofn:(i+1)*dofn] = Bi
            
    return B


def Elemnodes(elnum,Coords,Elemconn,nodes):
    
    NodeCoords = np.zeros([nodes,3]) #nodes = 6 -> Lin wedge
    Conn = Elemconn[elnum,:]
    for i in range(nodes):
        NodeCoords[i,:] = Coords[Conn[i],:]
        
    return NodeCoords,Conn

def EleStiff(elnum,B,D,stifchk,ngp,g,h,r):
    
    K = B.T @ D @ B
    
    # if stifchk:
    #     print("checking symmetry")
    #     nullmat = K-K.T
    #     if np.amax(nullmat) > 1e-10:
    #         print (np.amax(nullmat),ngp,g,h,r)
    return K


def GaussPts(Tngp):
    r1 = 1/6
    r2 = 1/(3**0.5)
    wgp = np.array([r1,r1,r1,r1,r1,r1])
    gpt = np.array([[r1,r1,-r2],
                    [4*r1,r1,-r2],
                    [r1,4*r1,-r2],
                    [r1,r1,r2],
                    [4*r1,r1,r2],
                    [r1,4*r1,r2]])
    
    return gpt,wgp,Tngp


def GaussLoop(elnum,NodeCoords,D,nodes,dofn):
    
    eldof = nodes*dofn
    Kel = np.zeros([eldof,eldof])
    Pel = np.zeros([eldof])
    gpt,wgp,Tngp = GaussPts(6)
    
    for ngp in range(Tngp):
        
        g,h,r = gpt[ngp,:]
        wt = wgp[ngp]
        errF = 0
        stifchk = 0
        
        N,dNg,dNh,dNr = shapefuns(g, h, r)
        detJ,dNx,dNy,dNz,J,Jinv = Jacobian(elnum, NodeCoords, N, dNg, dNh, dNr, errF)
        B = Elmat(elnum, NodeCoords, N, dNx, dNy, dNz, nodes, dofn)
        
        kk = EleStiff(elnum, B, D, stifchk, ngp, g, h, r)
        
        Kel += kk*detJ*wt
        
    return Kel,Pel


def BCs(NodesClamp,Constset,NDOF,dofn):
    bu = np.zeros([NDOF])
    for i in NodesClamp:
        bu[i*dofn:(i+1)*dofn] = 1
    
    bu, Xvals = Constrain(Constset,NDOF,dofn,bu)
    
    bu = bu.astype(bool)
    
    bk = ~bu
    
    return bk,bu,Xvals


def Constrain(Constset,NDOF,dofn,bu):
    Xvals = np.zeros([NDOF])
    for nd in Constset:
        node = nd[0]
        dof = nd[1]
        const = nd[2]
        # if const == 0:
        bu[dofn*node+dof] = 1
        Xvals[dofn*node+dof] = const
    return bu, Xvals
    
def TrimBC(BigP,NodesClamp,Constset,NDOF,dofn):
    
    bk,bu,Xvals = BCs(NodesClamp,Constset,NDOF,dofn)
    slicek = np.array([i for i in range(bk.shape[0]) if (bk[i])])
    sliceu = np.array([i for i in range(bu.shape[0]) if (bu[i])])
    # Kk = BigK[bk,:][:,bk]
    # Kku = BigK[bk,:][:,bu]
    # Kk = BigK[np.ix_(slicek[:],slicek[:])]
    # Kku = BigK[np.ix_(slicek[:],sliceu[:])]
    Pp = BigP[bk]
    
    return Pp,bk,bu,slicek,sliceu,Xvals


def PointLoads(Loadset,BigP): #Loadset = [NodeNum, DofNum, LoadVal]
    
    # Pel = np.zeros([18])
    
    for ld in Loadset:
        node = ld[0]
        dof = ld[1]
        loadval = ld[2]
        BigP[3*node+dof] += loadval
    
    return BigP

    
def Assembly(elnum,Kel,Pel,BigK,BigP,Conn,nodes,dofn):
    rows = np.zeros([nodes*dofn])
    cols = np.zeros([nodes*dofn])
    for i in range(nodes):
        ii = i*dofn
        row = Conn[i]*dofn
        rows[ii:ii+dofn] = np.arange(row,row+dofn)
    for j in range(nodes):
        jj = j*dofn
        col = Conn[j]*dofn
        cols[jj:jj+dofn] = np.arange(col,col+dofn)
            # BigK[row:row+dofn, col:col+dofn] += Kel[ii:ii+dofn, jj:jj+dofn]

        BigP[row:row+dofn] += Pel[ii:ii+dofn]
    
    BigK[np.ix_(rows[:],cols[:])] += Kel[:, :]
    return BigK,BigP

def Autoload(L,dof,Load):
    Loadset = []
    Load = Load/len(L)
    for i,node in enumerate(L):
        Loadset.append([node,dof,Load])
    return Loadset

def Autodisp(L,dof,disp):
    Constset = []
    for i,node in enumerate(L):
        Constset.append([node,dof,disp])
    return Constset


def ElemStrainGP(Xout,Conn,NodeCoords,elnum,dofn,nodes,eldisp,D,Dinv, errF): #eldisp from WedgePost.py
    Tngp = 6
    gpt,wgp,Tngp = GaussPts(Tngp)
    StrainGP = np.zeros([Tngp,6])  #6 is no. of strain components
    StressGP = np.zeros([Tngp,6])
    # ExtrapMat = np.zeros([Tngp,Tngp]) # 'A' Matrix for GP -> node extrap
    for i in range(Tngp):
        gp = gpt[i,:]
        N,dNg,dNh,dNr = shapefuns(gp[0],gp[1],gp[2])
        detJ,dNx,dNy,dNz,J,Jinv = Jacobian(elnum, NodeCoords, N, dNg, dNh, dNr, errF)
        B = Elmat(elnum,NodeCoords,N,dNx,dNy,dNz,nodes,dofn)
        epsilon = B @ eldisp[:,None]
        sigma = D @ epsilon
        StrainGP[i,:] = epsilon[:,0]
        StressGP[i,:] = sigma[:,0]
        # ExtrapMat[i,:] = N
    
    return StrainGP,StressGP #,ExtrapMat

def Extrapolation():
    Tngp = 6
    MainNodes = 6
    gpt,wgp,Tngp = GaussPts(Tngp)
    Extrapmat = np.zeros([Tngp,MainNodes])
    for i in range(Tngp):
        gp = gpt[i,:]
        N,dNg,dNh,dNr = shapefuns(gp[0],gp[1],gp[2])
        Extrapmat[i,:] = N
    Extrapmatinv = la.inv(Extrapmat)
    return Extrapmat, Extrapmatinv

def GPtoNode(StrainGP,StressGP,StrainNode,StressNode,NodeRep,Extrapmatinv,Conn,elnum):
    epsilon = Extrapmatinv @ StrainGP
    sigma = Extrapmatinv @ StressGP
    for i,node in enumerate(Conn):
        NodeRep[node] += 1
        StrainNode[node,:] += epsilon[i,:]
        StressNode[node,:] += sigma[i,:]
    
    return NodeRep,StrainNode,StressNode

def GaussCoords(gpt,NodeCoords,elnum,Conn,eldisp,GPList,GPDisp,Tngp,nodes):
    dispstack = np.zeros([nodes,3])
    dispstack[:,0] = eldisp[0::3]
    dispstack[:,1] = eldisp[1::3]
    dispstack[:,2] = eldisp[2::3]
    for i,gp in enumerate(gpt):
        Xgp = Mapping(NodeCoords,gp[0],gp[1],gp[2])
        Ugp = Mapping(dispstack,gp[0],gp[1],gp[2])
        GPList[elnum*Tngp+i,:] = Xgp
        GPDisp[elnum*Tngp+i,:] = Ugp
    return GPList,GPDisp

def CalcRFvector(BigK,bu,bk,Xout):
    Kuk = BigK[bu,:][:,bk]
    Kuu = BigK[bu,:][:,bu]
    RF = (Kuk @ Xout[bk]) + (Kuu @ Xout[bu])

    return RF

def NetForce(BigP,Facenodes,dof,dofn):
    force = 0
    for i in Facenodes:
        force += BigP[i*dofn+dof]
    return force

def AvgDisp(Xout,Facenodes,dof,dofn,bk):
    avgdisp = 0
    count = 0
    for i in Facenodes: 
        if bk[i*dofn+dof]:
            avgdisp += Xout[i*dofn+dof]
            count += 1
    
    return avgdisp/count

#----------------------------------------------------------------------------#
'''_______________________________________________
                QUADRATIC
   _______________________________________________'''


import InternalMesher_Multicell_Quadratic as mesh



global Failures
global Successes

def validation_solver(Cells,nIntElem,fibreRadius,kf,Dims,Matproplist,Clampfaces,Loadfaces,Loadaxes,PrescribedDisps,findEL,perturbCheck,rhomaxval):
    postProcessing = 1
    Failures = []
    Successes = []

    
    Coords,Elemconn,Elemtag,Faces,jobname,Dims,Cells = mesh.InternalMesh(1,Cells,nIntElem,fibreRadius,kf,Dims,perturbCheck,rhomaxval)
    
    eltype = 26
    dofn = 3
    nodes = 15

    Nel = Elemconn.shape[0]
    # Coords = np.array([[0, 0, 0],
    #                    [1e-0, 0, 0],
    #                    [0.5e-0, 0.866e-0, 0],
    #                    [0, 0, 1],
    #                    [1e-0, 0, 1],
    #                    [0.5e-0, 0.866e-0, 1],
    #                    [0, 0, 2],
    #                    [1e-0, 0, 2],
    #                    [0.5e-0, 0.866e-0, 2],
    #                    [0.5e-0, 0, 0],  #midnodes
    #                    [0.75e-0, 0.433e-0, 0],
    #                    [0.25e-0, 0.433e-0, 0],
    #                    [0.5e-0, 0, 2],
    #                    [0.75e-0, 0.433e-0, 2],
    #                    [0.25e-0, 0.433e-0, 2]])
    
    errF = 0
    
    Tnode = Coords.shape[0]
    NDOF = Tnode*dofn
    stifchk = 0  
    # E = 68e9;nu=0.25

    NodesClamp = []
    for clampface in Clampfaces:
        NodesClamp.extend(Faces[clampface])
    
    Loadset = []
    Constset = []
    

    '''_______________________________________________
                    QUADRATIC
       _______________________________________________'''
       

    loadforce = 0
    loaddisp = 1
    prescribedDisp = PrescribedDisps[0] #0.00012
    
    if (loadforce):    
        Loadset = Autoload(Faces[0],0,700000.0)

    if (loaddisp):
        for nconst,loadface in enumerate(Loadfaces):
            Constset.extend(Autodisp(Faces[loadface],Loadaxes[nconst],PrescribedDisps[nconst])) #0.0026417791763021104
    

    BigK = lil_matrix((NDOF,NDOF))
    BigP = np.zeros([NDOF])
    Xout = np.zeros([NDOF])
    checkt = time()
    for elnum in range(Nel):
        D = Material(Matproplist,Elemtag[elnum])
        Dinv = la.inv(D)
        NodeCoords,Conn = Elemnodes(elnum, Coords, Elemconn, nodes)
        Kel,Pel = GaussLoop(0,NodeCoords,D,nodes,dofn)
        
        BigK,BigP = Assembly(elnum, Kel, Pel, BigK, BigP, Conn, nodes, dofn)
        if (elnum%100 == 0):
            print(f"{time()-checkt :.7f} {elnum}")
            checkt = time()
    BigP = PointLoads(Loadset,BigP)    
    # Kk,Pp,bk,bu,Kku,Xvals = TrimBC(BigK,BigP,NodesClamp,Constset,NDOF,dofn)
    Pp,bk,bu,slicek,sliceu,Xvals = TrimBC(BigP,NodesClamp,Constset,NDOF,dofn)
    
    # Adding constraint force vector to actual.
    Pconst = BigK[np.ix_(slicek,sliceu)] @ Xvals[bu]
    print(f"{time()-checkt :.7f} BigK slice + Imposed disp done")
    checkt = time()
    Pp = Pp - Pconst
    # print(f"{time()-checkt :.7f} Reaction forces done")
    # checkt = time()
    
    # Disp = la.inv(Kk) @ Pp
    Disp = spsolve(BigK[np.ix_(slicek,slicek)].tocsr(), Pp)
    print(f"{time()-checkt :.7f} BigK slice + spsolve done")
    checkt = time()
    
    Xout[bu] = Xvals[bu]
    Xout[bk] = Disp
    print(np.amax(abs(Xout)))
    
    # RF = CalcRFvector(BigK, bu, bk, Xout)
    # BigP[bu] = RF
    
    RF = (BigK[np.ix_(sliceu,slicek)] @ Xout[bk]) + (BigK[np.ix_(sliceu,sliceu)] @ Xout[bu])
    BigP[bu] = RF
    print(f"{time()-checkt :.7f} BigK slice + RF done")
    checkt = time()
    
    ################ POST PROCESSING ##################
    if (postProcessing):
        gpt,wgp,Tngp = GaussPts(6)
        
        import WedgePost as post
        scale = 1.0e1
        ptdat = 1
        
        newCoords = post.Update_Position(Xout, Coords, Nel, Tnode, dofn, scale)
        
        StrainNode = np.zeros([Tnode,6])
        StressNode = np.zeros([Tnode,6])
        StrainGPs = np.zeros([6*Nel,6])
        StressGPs = np.zeros([6*Nel,6])
        GPList = np.zeros([6*Nel,3])
        GPDisp = np.zeros([6*Nel,3])
        NodeRep = np.zeros([Tnode])
        maxs=0
        for elnum in range(Nel):
            D = Material(Matproplist,Elemtag[elnum])
            Dinv = la.inv(D)
            NodeCoords,Conn = Elemnodes(elnum, Coords, Elemconn, nodes)
            eldisp = post.ElemDisp(Xout, Conn, nodes, dofn)
            GPList,GPDisp = GaussCoords(gpt, NodeCoords, elnum, Conn, eldisp, GPList, GPDisp, Tngp,nodes)
            StrainGP,StressGP = ElemStrainGP(Xout, Conn, NodeCoords, elnum, dofn, nodes, eldisp, D, Dinv, 0)
            StrainGPs[elnum*6:(elnum+1)*6,:] = StrainGP
            StressGPs[elnum*6:(elnum+1)*6,:] = StressGP
            
            if np.amax(StressGP)>maxs:
                maxs = np.amax(StressGP)
        '''_______________________________________________
                        QUADRATIC
           _______________________________________________'''
        
        post.vtk_output_format(jobname, Coords, Tnode, Elemconn, Nel, nodes, eltype, Xout, StrainNode, StressNode, ptdat)
        
        gpvtk = 1
        if gpvtk:
            post.vtk_outputgauss_format(jobname, GPList, Tngp*Nel, nodes, eltype, GPDisp, StrainGPs, StressGPs, ptdat)
        
        post.vtk_output_format(jobname+'_disp', newCoords, Tnode, Elemconn, Nel, nodes, eltype, Xout, StrainNode, StressNode, ptdat)

################ Verifying NASA Paper ##################

    if findEL[0]:   # findEL11
        netFaceForce = NetForce(BigP,Faces[0],1,dofn)
        
        areaFace = Dims[0]*Dims[2]
        EL11 = (netFaceForce * Dims[1]) / (areaFace * prescribedDisp)
        
        avgFaceDisp = AvgDisp(Xout, Faces[3], 2, dofn,bk) - AvgDisp(Xout, Faces[2], 2, dofn,bk)
        # areaFace = Dims[0]*Dims[1]
        VL12 = (avgFaceDisp / Dims[2]) / (prescribedDisp / Dims[1])
        
        print(f"EL11 = {EL11} \nVL12 = {VL12}")
        return(-EL11,-VL12)
    if findEL[1]:  # findEL22
        netFaceForce = NetForce(BigP,Faces[2],2,dofn)
        
        areaFace = Dims[0]*Dims[1]
        EL22 = (netFaceForce * Dims[2]) / (areaFace * prescribedDisp)
        
        avgFaceDisp = AvgDisp(Xout, Faces[5], 0, dofn,bk) - AvgDisp(Xout, Faces[4], 0, dofn,bk)
        VL23 = (avgFaceDisp / Dims[0]) / (prescribedDisp / Dims[2])
    
        print(f"EL22 = {EL22} \nVL23 = {VL23}")
        return(-EL22,-VL23)
    if findEL[2]:  # findEL33
        netFaceForce = NetForce(BigP,Faces[5],1,dofn)
        
        areaFace = Dims[1] * Dims[2]
        GL12 = (netFaceForce/areaFace) / (prescribedDisp/Dims[2])
        print(f"GL12 = {GL12}")
        return(GL12,0)
    if findEL[3]:
        netFaceForce = NetForce(BigP,Faces[5],2,dofn)
        areaFace = Dims[1] * Dims[2]
        GL23 = (netFaceForce/areaFace) / (prescribedDisp/Dims[2])
        print(f"GL23 = {GL23}")
        return(GL23,0)

# NEED TO RETURN (0,0) if not running EFFECTIVE MAT PROP (NASA) simulations 