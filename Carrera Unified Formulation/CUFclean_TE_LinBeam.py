#CUF with Taylor-like polynomial expansion over Cross-section
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

def FN_Stiff(C, ival, jval, tval, sval, CrossNode,CrossCon,NCrossEls,M,eln):
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
        c1,c2,c3,c4 = CrossNode,CrossCon,NCrossEls,M
        kxx = C22*Area_Int(Ftx,Fsx,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C66*Area_Int(Ftz,Fsz,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C44*Area_Int(Ft,Fs,c1,c2,c3,c4)*Line_Int(Niy,Njy,eln)
#        breakpoint()
        kxy = C23*Area_Int(Ft,Fsx,c1,c2,c3,c4)*Line_Int(Niy,Nj,eln) + \
              C44*Area_Int(Ftx,Fs,c1,c2,c3,c4)*Line_Int(Ni,Njy,eln)
        kxz = C12*Area_Int(Ftz,Fsx,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C66*Area_Int(Ftx,Fsz,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln)
        kyx = C44*Area_Int(Ft,Fsx,c1,c2,c3,c4)*Line_Int(Niy,Nj,eln) + \
              C23*Area_Int(Ftx,Fs,c1,c2,c3,c4)*Line_Int(Ni,Njy,eln)
        kyy = C55*Area_Int(Ftz,Fsz,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C44*Area_Int(Ftx,Fsx,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C33*Area_Int(Ft,Fs,c1,c2,c3,c4)*Line_Int(Niy,Njy,eln)
        kyz = C55*Area_Int(Ft,Fsz,c1,c2,c3,c4)*Line_Int(Niy,Nj,eln) + \
              C13*Area_Int(Ftz,Fs,c1,c2,c3,c4)*Line_Int(Ni,Njy,eln)
        kzx = C12*Area_Int(Ftx,Fsz,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C66*Area_Int(Ftz,Fsx,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln)
        kzy = C13*Area_Int(Ft,Fsz,c1,c2,c3,c4)*Line_Int(Niy,Nj,eln) + \
              C55*Area_Int(Ftz,Fs,c1,c2,c3,c4)*Line_Int(Ni,Njy,eln)
        kzz = C11*Area_Int(Ftz,Fsz,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C66*Area_Int(Ftx,Fsx,c1,c2,c3,c4)*Line_Int(Ni,Nj,eln) + \
              C55*Area_Int(Ft,Fs,c1,c2,c3,c4)*Line_Int(Niy,Njy,eln)
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

with open(r'dataTE.yaml') as file:
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
PLlock = 1
Icor = np.zeros((dof_per_node, n_nodes)).T #Active DOFS
Ncor, Icor, ndof = DOF_Active(Icor, BCs, dof_per_node, n_nodes)
bk = BCs.astype(bool)
bk = bk.flatten() # Boolean array of Boundary conditions to reduce K and P
Cmat = Material_prop(Emat, numat)
Cmat = PLMatedit(Cmat,PLlock)
chktag = 0
# STARTING ELEMENT LOOP
BigK = np.zeros([dof_per_node*n_nodes,dof_per_node*n_nodes]) #Global K

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
                                  NCrossEls,Mfun,eln) # Functional Nucleus 3x3
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
        print('K at 12,3',Kij[12,3])
    BigK[imap[0]:imap[-1]+1,imap[0]:imap[-1]+1] += Kij  #Global Assembly
    if ELEM%10==0:
        print(ELEM)
LDval = 10e1 #Load value
LoadP = np.zeros([dof_per_node*n_nodes])
nload = n_nodes #node of load
dload = 1 #dof of load in node
locload = (nload-1)*dof_per_node + dload - 1 #dof number to load
LoadP[locload] = LDval #Loading
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
        LDval*beam_len**3/(3*Emat*b*a**3/12)/DISP[-9])
