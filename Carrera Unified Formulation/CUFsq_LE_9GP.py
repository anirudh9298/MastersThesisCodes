# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:25:26 2021

@author: aniru
"""

#CUF with Langrange Elements as the Functions over the Cross-section
import numpy as np
import matplotlib
import yaml
import sys
import os
# Function Space

def DOF_Active(Icor,BCs,dofn,nn):
    # Creates a list with the active degrees of freedom based on BCs
    # Returns list of all Dofs(NCOR), activeDofs (ICOR) and DOFtotal (count)
    count = 0
    count1 = 0
    Ncor = np.zeros([nn,dofn],dtype = int)
    for i in range (nn):
        for j in range (dofn):
            Ncor[i,j] = count1 
            count1 = count1 + 1
            if BCs[i,j] == 1:
                count = count + 1
                Icor[i,j] = count
    return Ncor, Icor, count

def Material_prop(E,nu):
    # Creates the material properties matrix relating stress and strain.
    # Currently made for an isotropic material. 
    G = E/(2*(1+nu))
    lam = nu*E/((1+nu)*(1-2*nu))
    Cmat = np.array([[2*G+lam, lam, lam, 0, 0, 0],\
                     [lam, 2*G+lam, lam, 0, 0, 0],\
                     [lam, lam, 2*G+lam, 0, 0, 0],\
                     [0, 0, 0, G, 0, 0],\
                     [0, 0, 0, 0, G, 0],\
                     [0, 0, 0, 0, 0, G]])
    return Cmat

def PLMatedit(Cmat,PLlock):
    #Alleviate Poisson Locking according to Carrera pg.134
    #C22e is effective C22
    #PLlock just as a check
    if (PLlock):
        C22 = Cmat[1,1]
        C11 = Cmat[0,0]
        C12 = Cmat[0,1]
        C33 = Cmat[2,2]
        C13 = Cmat[0,2]
        C23 = Cmat[1,2]
        C22e = C22-(C12*(C33*C12-C13*C23)/(C11*C33-C13**2))\
                  -(C23*(C23*C11-C13*C12)/(C11*C33-C13**2))
        Cmat[1,1] = C22e
        print('Cmat is edited', C22/Cmat[1,1])
    return Cmat

def Line_Int(N1,N2,eln):
    # Computes the line integral of a product of functions defined by N1,N2
    # Uses two gauss points since linear beam
    # N1 = [a,b] ; a is the order of derivative, b is the shape function index 
    intl = 0.
    wgp = 1.
    gps = [-0.5773503692,0.5773503692]
    for gpt in gps:
        jac = 0.
        Ns = Nvals(gpt)
        for i in range(len(eln)):
            jac = jac + Ns[1][i]*eln[i] 
        j1 = 1
        j2 = 1
        if N1[0] ==1:
            j1 = 1/jac
        if N2[0] ==1:
            j2 = 1/jac
        intl = intl + j1*Ns[N1[0]][N1[1]] * j2*Ns[N2[0]][N2[1]]*wgp*jac
    return intl 

def Nvals(r):
    # Returns a list with rows signifying derivative order and the columns
    # signifying the shape function index
    # Currently uses linear beam with parametric coordinates.
    Ns = [(1-r)/2,(1+r)/2]
    Nys = [-1/2,1/2]
    return [Ns,Nys]

def Area_Int(F1,F2,CrossNode,CrossCon,NCrossEls,M):
    # Computes the integral across the cross-section
    # Uses the Crossectional data to from Gauss points within each element
    # Elements are currently right triangular
    # F1[a,b]; a = derivative order (0,x,z); b = Function index  
    # 6 gauss points in use
    try:
        inta = 0
        for i in range(NCrossEls):
            xs = np.zeros([3])
            zs = np.zeros([3])
            for j in range(3):
                nn = CrossCon[i,j]
                xs[j] = CrossNode[nn,0]
                zs[j] = CrossNode[nn,2]
            xm = (xs[0] + xs[1] + xs[2])/3
            zm = (zs[0] + zs[1] + zs[2])/3
            AreaTri = 0.5 * np.linalg.det(np.array([[xs[0],xs[1],xs[2]],\
                                                    [zs[0],zs[1],zs[2]],\
                                                    [1,1,1]]))
            if (AreaTri <= 0):
                print ('zero area')
                breakpoint()
            trigp = np.array([[0.6590276223,0.2319333685,0.1090390090],\
                              [0.6590276223,0.1090390090,0.2319333685],\
                              [0.1090390090,0.6590276223,0.2319333685],\
                              [0.2319333685,0.6590276223,0.1090390090],\
                              [0.2319333685,0.1090390090,0.6590276223],\
                              [0.1090390090,0.2319333685,0.6590276223]])
            wgp = 0.1666666667
            ngp = trigp.shape[0]
            intb = 0
            for k in range(ngp):
                L1,L2,L3 = trigp[k,:]
                xp = L1*xs[0] + L2*xs[1] + L3*xs[2]
                zp = L1*zs[0] + L2*zs[1] + L3*zs[2]
                Flist = Fvals(xp,zp,M)
                intb = intb + Flist[F1[0]][F1[1]]*Flist[F2[0]][F2[1]] * wgp
            inta = inta + AreaTri*intb
        return inta
    except Exception as inst: 
        print('\n#####ERROR in logs#######\n',type(inst),inst)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(fname,'LINE', exc_tb.tb_lineno,'\n')
        print(F1,F2,Flist)
        return 0

def Fvals(x,z,M):
    # Returns values of F as a list with rows covering derivative orders and
    # columns covering function indices
    Fs = [1, x, z, x**2, x*z, z**2, x**3, x**2*z, x*z**2, z**3]
    Fxs = [0, 1, 0, 2*x, z, 0, 3*x**2, 2*x*z, z**2, 0]
    Fzs = [0, 0, 1, 0, x, 2*z, 0, x**2, 2*x*z, 3*z**2]
    return [Fs[0:M], Fxs[0:M], Fzs[0:M]]

def Fvals_LE(alph,beta,M):
    if (M == 3):
        Fs = [1-alph-beta, alph, beta]
        Fas = [-1,1,0]
        Fbs = [-1,0,1]
    elif (M == 4):
        Fs =  [0.25*(1-alph)*(1-beta), 0.25*(1+alph)*(1-beta),\
               0.25*(1+alph)*(1+beta), 0.25*(1-alph)*(1+beta)]
        Fas = [0.25*(beta-1), 0.25*(1-beta), 0.25*(1+beta), 0.25*(-1-beta)]
        Fbs = [0.25*(alph-1), 0.25*(-1-alph), 0.25*(1+alph), 0.25*(1-alph)]
        #Fbs = [0.25*(1-alph), 0.25*(alph-1), 0.25*(-1-alph), 0.25*(1+alph)]

    return [Fs[0:M], Fas[0:M], Fbs[0:M]]

def Area_Int_LE(F1,F2,M,NodeCoords,CrossCon,elc):
    if (M == 3):
        trigp = np.array([[0.6590276223,0.2319333685,0.1090390090],\
                          [0.6590276223,0.1090390090,0.2319333685],\
                          [0.1090390090,0.6590276223,0.2319333685],\
                          [0.2319333685,0.6590276223,0.1090390090],\
                          [0.2319333685,0.1090390090,0.6590276223],\
                          [0.1090390090,0.2319333685,0.6590276223]])
        wgp = 0.1666666667
        ngp = trigp.shape[0]
        inta = 0
    elif (M == 4):
        r3 = 1/np.sqrt(3)
        r6 = np.sqrt(0.6)
        # gps = np.array([[-r3,-r3],[r3,-r3],[-r3,r3],[r3,r3]])
        gps = np.array([[-r6,-r6],[-r6,0],[-r6,r6],[0,-r6],[0,0],[0,r6],[r6,-r6],[r6,0],[r6,r6]])
        wgp = 5.0/9.0
        inta = 0.0
        ngp = gps.shape[0]
        xs = NodeCoords[:,0]
        zs = NodeCoords[:,2]
        for i in range(ngp):
            alph,beta = gps[i,:]
            if (alph==0):
                wgp1 = 8.0/9.0
            else:
                wgp1 = wgp
            if (beta==0):
                wgp2 = 8.0/9.0
            else:
                wgp2 = wgp
            Flist = Fvals_LE(alph,beta,M)
            Jac,mult,X,Z = Jacoby(Flist,xs,zs,CrossCon[elc],alph,beta)
            Fts = np.array(Flist[1:3][:])
            Fxy = mult @ Fts
            for i in range(Fxy.shape[0]):
                for j in range(Fxy.shape[1]):
                    Flist[i+1][j] = Fxy[i,j]
            Ftau = Flist[F1[0]][F1[1]]
            Fsss = Flist[F2[0]][F2[1]]
            inta += wgp1*wgp2*Ftau*Fsss*Jac

    return inta 

def Jacoby(Flist,xs,zs,C,alph,beta):
    Fs = Flist[0]
    Fas = Flist[1]
    Fbs = Flist[2]
    X = Fs[0]*xs[C[0]] + Fs[1]*xs[C[1]] + Fs[2]*xs[C[2]] + Fs[3]*xs[C[3]]
    xa = Fas[0]*xs[C[0]] + Fas[1]*xs[C[1]] + Fas[2]*xs[C[2]] + Fas[3]*xs[C[3]]
    xb = Fbs[0]*xs[C[0]] + Fbs[1]*xs[C[1]] + Fbs[2]*xs[C[2]] + Fbs[3]*xs[C[3]]
    Z = Fs[0]*zs[C[0]] + Fs[1]*zs[C[1]] + Fs[2]*zs[C[2]] + Fs[3]*zs[C[3]]
    za = Fas[0]*zs[C[0]] + Fas[1]*zs[C[1]] + Fas[2]*zs[C[2]] + Fas[3]*zs[C[3]]
    zb = Fbs[0]*zs[C[0]] + Fbs[1]*zs[C[1]] + Fbs[2]*zs[C[2]] + Fbs[3]*zs[C[3]]
    Jmat = np.array([[xa,za],[xb,zb]])
    Jac = np.linalg.det(Jmat)
#    print(Jac)
    if Jac<=0:
        print('Invalid Jacobian at X,Z: ',X,Z,alph,beta)
    mult = np.array([[zb, -za],[-xb, xa]]) * 1/Jac
    return Jac,mult,X,Z 

def FN_Stiff(C,ival,jval,tval,sval,CrossNode,CrossCon,NCrossEls,M,eln,elc):
    # Main Fundamental Nucleus function. Uses the computed area and line
    # integrals to find 3x3 K matrix which is then assembled in in the main
    # routine. Output is a 3x3 list.
    try:
        Ftx = [1,tval]
        Ftz = [2,tval]
        Fsx = [1,sval]
        Fsz = [2,sval]
        Ft = [0,tval]
        Fs = [0,sval]
        C11 = C[0,0]
        C22 = C[1,1]
        C33 = C[2,2]
        C44 = C[3,3]
        C55 = C[4,4]
        C66 = C[5,5]
        C12 = C[0,1]
        C13 = C[0,2]
        C23 = C[1,2]
        Ni = [0,ival]
        Nj = [0,jval]
        Niy = [1,ival]
        Njy = [1,jval]
        c1,c2,c3,c4 = M,CrossNode,CrossCon,elc
        kxx = C22*Area_Int_LE(Ftx,Fsx,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C66*Area_Int_LE(Ftz,Fsz,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C44*Area_Int_LE(Ft,Fs,c1,c2,c3,c4)*Line_Int(Niy,Njy,eln)
#        breakpoint()
#EDITTED!!!
        kxy = C23*Area_Int_LE(Ftx,Fs,c1,c2,c3,c4)*Line_Int(Ni,Njy,eln) + \
              C44*Area_Int_LE(Ft,Fsx,c1,c2,c3,c4)*Line_Int(Niy,Nj,eln)
        kxz = C12*Area_Int_LE(Ftx,Fsz,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C66*Area_Int_LE(Ftz,Fsx,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln)
        kyx = C44*Area_Int_LE(Ftx,Fs,c1,c2,c3,c4)*Line_Int(Ni,Njy,eln) + \
              C23*Area_Int_LE(Ft,Fsx,c1,c2,c3,c4)*Line_Int(Niy,Nj,eln)
        kyy = C55*Area_Int_LE(Ftz,Fsz,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C44*Area_Int_LE(Ftx,Fsx,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C33*Area_Int_LE(Ft,Fs,c1,c2,c3,c4)*Line_Int(Niy,Njy,eln)
        kyz = C55*Area_Int_LE(Ftz,Fs,c1,c2,c3,c4)*Line_Int(Ni,Njy,eln) + \
              C13*Area_Int_LE(Ft,Fsz,c1,c2,c3,c4)*Line_Int(Niy,Nj,eln)
        kzx = C12*Area_Int_LE(Ftz,Fsx,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C66*Area_Int_LE(Ftx,Fsz,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln)
        kzy = C13*Area_Int_LE(Ftz,Fs,c1,c2,c3,c4)*Line_Int(Ni,Njy,eln) + \
              C55*Area_Int_LE(Ft,Fsz,c1,c2,c3,c4)*Line_Int(Niy,Nj,eln)
        kzz = C11*Area_Int_LE(Ftz,Fsz,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C66*Area_Int_LE(Ftx,Fsx,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C55*Area_Int_LE(Ft,Fs,c1,c2,c3,c4)*Line_Int(Niy,Njy,eln)
        k = [[kxx, kxy, kxz],[kyx, kyy, kyz],[kzx, kzy, kzz]]
        return (k)
    except Exception as inst: 
        print('\n#####ERROR in logs#######\n',type(inst),inst)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(fname,'LINE', exc_tb.tb_lineno)
        return ([[0,0,0],[0,0,0],[0,0,0]])
    
############## MAIN ################
# pylint: disable=E1137
# Data input

with open(r'dataLE.yaml') as file:
    datfile = yaml.load(file, Loader = yaml.FullLoader)

#print(datfile.keys())
for dik in datfile.keys():
    div = datfile[dik]
    if type(div) == list:
        datfile[dik] = np.array(div)

eltype, n_elems, n_nodes_per_elem, n_nodes, Coords, Elemconn, beam_len, \
geomtype, cross_dims, Nexp, Mfun, dof_per_node, BCs, Emat,numat,\
CrossNode, CrossCon = datfile.values()
#Coords = Coords/2
NCrossEls = CrossCon.shape[0] #No. of elements across cross-section
# ux = ux1  + x ux2 + z ux3
#     |____| |____________|
#      N=0         N=1
shapefn_index = n_nodes_per_elem
# Some Geometry Considerations
if geomtype == 'rect':
    a = cross_dims[0]
    b = cross_dims[1]

# CUF Input variables
#dof_per_node = 3 * M # currently : CLEC
PLlock = 0
Icor = np.zeros((dof_per_node, n_nodes)).T #Active DOFS
Ncor, Icor, ndof = DOF_Active(Icor, BCs, dof_per_node, n_nodes)
bk = BCs.astype(bool)
bk = bk.flatten() # Boolean array of Boundary conditions to reduce K and P
Cmat = Material_prop(Emat, numat)
Cmat = PLMatedit(Cmat,PLlock)
chktag = 0
# STARTING ELEMENT LOOP
BigK = np.zeros([dof_per_node*n_nodes,dof_per_node*n_nodes]) #Global K
elc = 0
for ELEM in range(n_elems): # ELEMENT LOOP
    node1,node2 = Elemconn[ELEM] - 1 #Node numbers in the element
    eln = [Coords[node1,1],Coords[node2,1]]  
    imap = np.array([Ncor[node1,:],Ncor[node2,:]]).flatten()
    Kij = np.zeros([3*Mfun*shapefn_index,3*Mfun*shapefn_index])

    for NODEi in range(shapefn_index): #LOOP OVER NODE 1 (i)

        for NODEj in range(shapefn_index): #LOOP OVER NODE 2 (j)
            Kts = np.zeros([3*Mfun,3*Mfun])

            for T in range(Mfun): #LOOP OVER TAU

                for S in range(Mfun): #LOOP OVER S
                    ITERVARS = [ELEM,T,S,NODEi,NODEj]
                    kk = FN_Stiff(Cmat,NODEi,NODEj,T,S,CrossNode,CrossCon,\
                                  NCrossEls,Mfun,eln,elc) # Functional Nucleus 3x3
                    kk = np.array(kk)
                    #if ITERVARS == [0,1,1,0,1]: #To test different kijts vals
                        #print(kk[0,0])
                    if ((eltype == 'B2N1_TSDT') & (NODEi == NODEj) & \
                        (([T,S] == [2,2]) | ([T,S] == [3,3]))):#penaltyTSDT
                        kmax = kk.max()
                        kk[0,0] = kmax * 1.e+03
                        kk[2,2] = kmax * 1.e+03
                        #print(kk)

                    Kts[3*S:3*(S+1),3*T:3*(T+1)] = kk #T,S Stiff mat

            Kij[3*Mfun*NODEj:3*Mfun*(NODEj+1),\
                 3*Mfun*NODEi:3*Mfun*(NODEi+1)] = Kts #i,j stiff mat (elem)
             
#            break
#        break

    if ELEM == 0: #Extracting stiff vals for testing
        print('K at 12,3',Kij[15,3])
    BigK[imap[0]:imap[-1]+1,imap[0]:imap[-1]+1] += Kij  #Global Assembly
    if ELEM%10==0:
        print(ELEM)
LDval = 10e1 #Load value
LoadP = np.zeros([dof_per_node*n_nodes])
nload = n_nodes #node of load
dload = 1 #dof of load in node
locload = (nload-1)*dof_per_node + dload - 1 #dof number to load
#LoadP[locload::3] = LDval #Loading
LoadP[locload::3]=LDval
for i in range(BigK.shape[0]):
    for j in range(i,BigK.shape[1]):
        BigK[i,j]=BigK[j,i]
KK = BigK[bk,:][:,bk] #Truncating stiffness matrix
PP = LoadP[bk] #Truncating Load Vector
KI = np.linalg.inv(KK) #Inversion and Solving
DISP = KI @ PP
a1=a/2;b1=b/2
k2212xx =2/3*Cmat[5,5]*a1*b1*1 - 4/3*Cmat[4,4]*(b1*a1**3)/12 #Testvalue
print('displacement Euler Ref: ',  LDval*beam_len**3/(3*Emat*b*a**3/12)  )
#print('displacement Current: ', DISP)
np.savetxt("KK.csv", BigK, delimiter=",")
print('\n Ratio between displacements = ',
        LDval*beam_len**3/(3*Emat*b*a**3/12)/DISP[-3])

## PLOTTER ##

for i in range(CrossCon.shape[0]):
    Cellpoints = CrossCon[i,:]
    Cellshape = np.zeros_like(CrossNode)
    Cellorig = np.zeros_like(CrossNode)
    scale = 1e3
    for j in range(CrossCon.shape[1]):
        cellnode = Cellpoints[j]
        Cellshape[j,0] = scale * DISP[cellnode*3+0] + CrossNode[cellnode,0]
        Cellshape[j,1] = scale * DISP[cellnode*3+1] + CrossNode[cellnode,1]
        Cellshape[j,2] = scale * DISP[cellnode*3+2] + CrossNode[cellnode,2]
        scal = 0 
        Cellorig[j,0] = scal * DISP[cellnode*3+0] + CrossNode[cellnode,0]
        Cellorig[j,1] = scal * DISP[cellnode*3+1] + CrossNode[cellnode,1]
        Cellorig[j,2] = scal * DISP[cellnode*3+2] + CrossNode[cellnode,2]

        
from matplotlib import pyplot as plt

plt.plot(Cellorig[:,0],Cellorig[:,2])
plt.plot(Cellshape[:,0],Cellshape[:,2])
plt.savefig('foo.png')
