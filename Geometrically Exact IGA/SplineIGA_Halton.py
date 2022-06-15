# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:13:55 2020

@author: aniru
"""
import numpy as np
import sympy as sp
from sympy import Array
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from time import time 
startt=time()
import sys
import os

import cProfile



def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

def symmetrize(a):
    """
    Return a symmetrized version of NumPy array a.

    Values 0 are replaced by the array value at the symmetric
    position (with respect to the diagonal), i.e. if a_ij = 0,
    then the returned array a' is such that a'_ij = a_ji.

    Diagonal values are left untouched.

    a -- square NumPy array, such that a_ij = 0 or a_ji = 0, 
    for i != j.
    """
    return a + a.T - np.diag(a.diagonal())
# %%
def FindSpan(n,p,u,U):
    if (u==U[n+1]):
        return (n)
    low=p
    high=n+1
    mid=(low+high)/2
    mid=int(mid)
    while (u<U[mid] or u>=U[mid+1]):
        if (u<U[mid]):
            high=mid
        else:
            low=mid
        mid=(low+high)/2
        mid=int(mid)
    return (mid)
# %%
def BasisFuns(i,u,p,U):
    N=[None]*(p+1)
    left=[None]*(p+1)
    right=[None]*(p+1)
    N[0]=1.0
  
    for j in range(1,p+1):
        left[j]=u-U[i+1-j]
        right[j]=U[i+j]-u
        saved=0.0
        for r in range(0,j):
            #print(j,r)
            temp=N[r]/(right[r+1]+left[j-r])
            N[r]=saved+right[r+1]*temp
            saved=left[j-r]*temp
            #print(N)
        N[j]=saved
        
    return(N)   
# %%
def DersBasisFuns(i,u,p,n,U):
    ndu=[[0 for j in range(p+1)] for i in range(p+1)] #[[0]*(p+1)]*(p+1)
    a=[[0 for j in range(p+1)] for i in range(2)]
    ndu[0][0]=1.0
    ders=[[0 for j in range(p+1)] for i in range(n+1)]
    left=[0]*(p+1)
    right=[0]*(p+1)
    for j in range(1,p+1):
        left[j]=u-U[i+1-j]
        right[j]=U[i+j]-u
        saved=0.0
        for r in range(0,j):
            ndu[j][r]=right[r+1]+left[j-r]
            temp=ndu[r][j-1]/ndu[j][r]
            
            ndu[r][j]=saved+right[r+1]*temp
            saved=left[j-r]*temp
        ndu[j][j]=saved
    for j in range(0,p+1):
        ders[0][j]=ndu[j][p]    
        
    for r in range(0,p+1):
        s1=0
        s2=1
        a[0][0]=1.0
        for k in range (1,n+1):
            d=0.0
            rk=r-k
            pk=p-k
            if (r>=k):
                a[s2][0]=a[s1][0]/ndu[pk+1][rk]
                d=a[s2][0]*ndu[rk][pk]
            if (rk>=-1):
                j1=1
            else:
                j1=-rk
            if (r-1<=pk):
                j2=k-1
            else:
                j2=p-r
            
            for j in range(j1,j2+1):
                a[s2][j]=(a[s1][j]-a[s1][j-1])/ndu[pk+1][rk+j]
                d= d+ a[s2][j]*ndu[rk+j][pk]
            if (r<=pk):
                a[s2][k]= -a[s1][k-1]/ndu[pk+1][r]
                d= d+ a[s2][k]*ndu[r][pk]
            ders[k][r]=d
            j=s1
            s1=s2
            s2=j
    r=p
    for k in range(1,n+1):
        for j in range(0,p+1):
            ders[k][j] = ders[k][j]* r
        r = r*(p-k)
    return (ders)

###################################
# %%
    
def rho_to_R(rhos):
    if len(np.shape(rhos))==2:
        rhos=np.squeeze(rhos)
    p0=rhos[0]
    p1=rhos[1]
    p2=rhos[2]
    p3=rhos[3]
    R=np.zeros([3,3])
    p02=p0**2
    p12=p1**2
    p22=p2**2
    p32=p3**2
    R[0,0]=p02+p12-p22-p32
    R[0,1]=2*(p1*p2-p3*p0)
    R[0,2]=2*(p1*p3+p2*p0)
    R[1,0]=2*(p1*p2+p3*p0)
    R[1,1]=p02-p12+p22-p32
    R[1,2]=2*(-p1*p0+p2*p3)
    R[2,0]=2*(p1*p3-p2*p0)
    R[2,1]=2*(p1*p0+p2*p3)
    R[2,2]=p02-p12-p22+p32
    
    return R

def tilde(r):
    rtilde=np.zeros([3,3])
    rtilde[0,1]=-r[2]
    rtilde[0,2]=r[1]
    rtilde[1,2]=-r[0]
    rtilde-=rtilde.T
    return rtilde
# %%

def qconj(rhos):
    rhostar=np.zeros([1,4])
    rhostar[0,0]=rhos[0,0]
    rhostar[0,1]=-rhos[0,1]
    rhostar[0,2]=-rhos[0,2]
    rhostar[0,3]=-rhos[0,3]
    return rhostar

def Control_points(n,nknots,p,U):
    cplist=[]
    spanassoc=[]
    for i in range(nknots):
        cs=[]
        span=FindSpan(n,p,i,U)
        for j in range(3):
            cs.append(span-p+j)
        cplist.append(cs)
    
    for iii in range (nknots):
        ps=[]
        for ii in cplist[iii]:
            for i in range(nknots):
                if ii in cplist[i]:
                    ksp= i #FindSpan(n,p,i,U)
                    if ksp not in ps:
                        ps.append(ksp)
        spanassoc.append(ps)
    return cplist,spanassoc

# %%
def auxcorot(PsRhos,p,span):
    rhoavg=np.zeros([1,4])
    for i in range (p+1):
        rhoavg=rhoavg+PsRhos[span-p+i,:]
    rhoavg=rhoavg/(p+1)
    return rhoavg

def MSEmat(A,B):
    if A.shape!=B.shape:
        print('LENS DONT MATCH',A.shape,B.shape)
        return -1
    e=0
    d=0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            e+= (A[i,j]-B[i,j])**2
            d+= A[i,j]**2
    print('Mean Square ERROR= ',e, '\nMatrix sq sum= ',d)
    return e/d

def perturbq(PsRhos,PsX,PsY,PsZ,tag):
    if tag==0:
        PsRhos[:,:]=1e-10
        PsRhos[:,0]=1
    else:
        xref=np.array([1,0,0])
        zrot=np.array([0,0,1])
        
        for i in range(n+1):
            p=PsRhos[i,:]
            x1=PsX[i]
            y1=PsY[i]
            z1=PsZ[i]
            if i==0:
                x2=PsX[i+1]
                y2=PsY[i+1]
                z2=PsZ[i+1]
                v=[x2-x1,y2-y1,z2-z1]
                theta=np.arccos(v[0]/(modv(v))) #DOT PRODUCT WITH xref
                if v[1] < 0:
                    theta=-theta
                p[0]=np.cos(theta/2)
                p[1:4]=zrot[:]*np.sin(theta/2)
                PsRhos[i,:]=p
                #print(p)
            elif i==n:
                x0=PsX[i-1]
                y0=PsY[i-1]
                z0=PsZ[i-1]
                v=[x1-x0,y1-y0,z1-z0]
                theta=np.arccos(v[0]/modv(v)) #DOT PRODUCT with xref
                if v[1] < 0:
                    theta=-theta
                p[0]=np.cos(theta/2)
                p[1:4]=zrot[:]*np.sin(theta/2)
                PsRhos[i,:]=p
            else:
                x2=PsX[i+1]
                y2=PsY[i+1]
                z2=PsZ[i+1]
                v2=[x2-x1,y2-y1,z2-z1]
                theta2=np.arccos(v2[0]/(modv(v2))) #DOT PRODUCT WITH xref                
                if v2[1] < 0:
                    theta2=-theta2
                x0=PsX[i-1]
                y0=PsY[i-1]
                z0=PsZ[i-1]
                v1=[x1-x0,y1-y0,z1-z0]
                theta1=np.arccos(v1[0]/modv(v1)) #DOT PRODUCT with xref
                if v1[1] < 0:
                    theta1=-theta1
                theta=(theta2+theta1)/2
                p[0]=np.cos(theta/2)
                p[1:4]=zrot[:]*np.sin(theta/2)
                PsRhos[i,:]=p
            print(theta)                   
    for i in range(n+1):
        p=PsRhos[i,:]
        modp=np.sqrt(p[0]**2+p[1]**2+p[2]**2+p[3]**2)
        p=p/modp
        PsRhos[i,:]=p
    
    return PsRhos

def normalq(PsRhosn):
    for i in range(n+1):
        q=PsRhosn[i,:]
        qm=modq(q)
        if qm !=1:
            q=q/qm
        PsRhosn[i,:]=q
    return PsRhosn
#%%
def qimag(rhos):
    irhos=rhos[0,1:4]
    return irhos
def qpure(v):
    rho=np.zeros([1,4])
    rho[0,1:4]=v
    return rho

def cross(v1,v2):
    return np.array([v1[1]*v2[2]-v1[2]-v2[1],v1[2]*v2[0]-v1[0]*v2[2],v1[0]*v2[1]-v1[1]*v2[0]])

def qmul(rho1,rho2):
    if len(np.shape(rho1))==1:
        rho1=np.expand_dims(rho1,0)
    if len(np.shape(rho2))==1:
        rho2=np.expand_dims(rho2,0)
    rho1s=rho1[0,0]
    rho2s=rho2[0,0]
    rho1v=rho1[0,1:4]
    rho2v=rho2[0,1:4]
    #print(rho1v,rho2v)
    outs=rho1s*rho2s - np.dot(rho1v,rho2v)
    outv=rho1s*rho2v + rho2s*rho1v + np.cross(rho1v,rho2v)
    out=np.zeros([1,4])
    out[0,0]=outs
    out[0,1:4]=outv
    return out

def logs1(rhos):
    loggers=np.zeros_like(rhos)
    for i in range (4):
        loggers[0,i]=np.log(rhos[0,i])
    #print(loggers)
    return loggers

def modq(rhos):
    if len(np.shape(rhos))==1:
        rhos=np.expand_dims(rhos,0)
    mod=np.sqrt(rhos[0,0]**2+rhos[0,1]**2+rhos[0,2]**2+rhos[0,3]**2)
    return mod

def modv(v):
    if len(np.shape(v))==1:
        v=np.expand_dims(v,0)
    mod=np.sqrt(v[0,0]**2+v[0,1]**2+v[0,2]**2)
    return mod

def exps(rhos):
    s=rhos[0,0]
    v=rhos[0,1:4]
    modvec=np.sqrt(v[0]**2+v[1]**2+v[2]**2)
    exps=np.cos(modvec/2)
    expv=v/modvec*np.sin(modvec/2)
    out=np.zeros_like(rhos)
    out[0,0]=exps
    out[0,1:4]=expv
    return out
    
def logs(rhos):
    try:
        for i in range(4):
            if rhos[0,i]==0:
                rhos[0,i]=1e-50
        loggers=np.zeros_like(rhos)
        v=rhos[0,1:4]
        s=rhos[0,0]
        mod=modq(rhos)
        modvec=np.sqrt(v[0]**2+v[1]**2+v[2]**2)
        lns=np.log(mod)
        lnv=2*np.arccos(s/mod)*v/modvec
        #print(v,lnv)
        loggers[0,0]=0
        loggers[0,1:4]=lnv
        return loggers
    except Exception as inst: 
        print('\n#####ERROR in logs#######\n',type(inst),inst)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(fname,'LINE', exc_tb.tb_lineno)
        return 0

        
#%%
    
def CurveInitParams(Ns,Nds,PsX,PsY,PsZ,p,span):
    try:
        r=[0,0,0]
        rd=[0,0,0]
        for i in range (p+1):
            r[0]=r[0]+Ns[i]*PsX[span-p+i]
            r[1]=r[1]+Ns[i]*PsY[span-p+i]
            r[2]=r[2]+Ns[i]*PsZ[span-p+i]
            rd[0]=rd[0]+Nds[1][i]*PsX[span-p+i]
            rd[1]=rd[1]+Nds[1][i]*PsY[span-p+i]
            rd[2]=rd[2]+Nds[1][i]*PsZ[span-p+i]
        Jus=(rd[0]**2+rd[1]**2+rd[2]**2)**(0.5)
        return(r,rd,Jus)

    except Exception as inst: 
        print('\n#####ERROR in CurveInitParams#######\n',type(inst),inst)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(fname,'LINE', exc_tb.tb_lineno)
        return 0,0,0



def CurveRots(Ns,Nds,PsRhos,p,span):
    try:
        rhoavg=auxcorot(PsRhos,p,span)
        rhoavgconj=qconj(rhoavg)
        vout=np.zeros([1,3])
        vout[:,:]=1e-50
        for i in range (p+1):
            rhoi=PsRhos[span-p+i,:]
            #print(rhoi,rhoavgconj) #CHECK DIMS
            qmuls=qmul(rhoavgconj,rhoi)
            #print(qmuls,logs(qmuls))
            xi=qimag(logs(qmuls))
            #print(xi)
            vout+=Ns[i]*xi
        qout=exps(qpure(vout))
        #print(qout)
        out=qmul(rhoavg,qout)
        #print(out)
        return out
    except Exception as inst: 
        print('\n#####ERROR in CurveRots#######\n',type(inst),inst)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(fname,'LINE', exc_tb.tb_lineno)
        return 0
    
def Curvatures(Ns,Nds,rhos,p,span,Jus):
    try:    
        kappa=np.zeros([1,3])
        for i in range (p+1):
            rhoi=PsRhos[span-p+i,:]
            holder=qimag(logs(qmul(qconj(rhos),rhoi)))
            holder=np.expand_dims(holder,0)
            #print(holder[0,1])
            for j in range (3):
                kappa[0,j]+= 1/Jus*Nds[1][i]*holder[0,j]
           # print(kappa)
        return (kappa)
    except Exception as inst: 
        print('\n#####ERROR in Curvatures#######\n',type(inst),inst)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(fname,'LINE', exc_tb.tb_lineno)
        return 0
    
def Strains(Ns,Nds,rs,rds,rhos,p,span,Jus):            
    try:
        R=rho_to_R(rhos)
        RT=R.T
        #print(R)
        ex=[1,0,0]
        epsilon=(RT @ rds)*1/Jus - ex
        #print(epsilon)
        return epsilon
    except Exception as inst: 
        print('\n#####ERROR in Strains#######\n',type(inst),inst)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(fname,'LINE', exc_tb.tb_lineno)
        return 0
# %%
def ABDmat(epsilon,kappa,E,A,G,J,Iy,Iz,ky,kz):
    try:        
        C=np.zeros([3,3])
        D=np.zeros([3,3])
        ABD=np.zeros([6,6])
        C[0,0]=E*A
        C[1,1]=G*A*ky
        C[2,2]=G*A*kz
        D[0,0]=G*J
        D[1,1]=E*Iy
        D[2,2]=E*Iz
        ABD[0:3,0:3]=C
        ABD[3:6,3:6]=D
        
        strns=np.zeros([6,1])
        strns[0:3,0]=epsilon
        strns[3:6,0]=kappa
        
        nms=ABD @ strns
        #print(nms)
        n=nms[0:3,0]
        m=nms[3:6,0]
        return n, m
    except Exception as inst: 
        print('\n#####ERROR in ABDmat#######\n',type(inst),inst)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(fname,'LINE', exc_tb.tb_lineno)
        return 0,0
    
def intmatrix(Ns,Nds,rs,rds,rhos,p,span,Jus,Jac1,n,m,l_intmat,wgp):          #INCOMPLETE
    try:
        ll=np.zeros([6,6])
        Fgpcont=np.zeros([p+1,6])
        R=rho_to_R(rhos)
        rdash=(np.array(rds)/Jus).tolist()
        rdtilde=tilde(rdash)
        #print(rdash,rdtilde)
        for ii in range(p+1):
            i=span-p+ii
            #print(i)
            q1=(Nds[1][ii])*np.eye(3,3)
            q2=Jus*Ns[ii]*(R@rdtilde)
            q3=np.zeros([3,3])
            q4=Nds[1][ii]*R
            ll[0:3,0:3]=q1
            ll[0:3,3:6]=q2
            ll[3:6,0:3]=q3
            ll[3:6,3:6]=q4
            ll=wgp*Jac1*ll
            ll=ll.T
           # print(ll)
            na=R@n
            ma=R@m
            nma=np.zeros([6,1])
            nma[0:3,0]=na
            nma[3:6,0]=ma
            #l_intmat.append(ll @ nma)
            fcont=ll @ nma
            #print(fcont.shape)
            #print(l_intmat[None,i],fcont.T[0,:])
            #l_intmat[:,i]= l_intmat[:,i] + fcont.T[0,:]
            Fgpcont[ii,:]+=fcont.T[0,:]

            #print(l_intmat)
            #print(ll)
        if pert==-20:
            print(Fgpcont,'______FCONT')
        l_intmat[:,span-p:span+1]+=Fgpcont.T
        
        return l_intmat,Fgpcont,0
    except Exception as inst:
        print('\n#####ERROR in intmatrix########\n',type(inst),inst)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(fname,'LINE', exc_tb.tb_lineno)
        return 0,1,1

def intstack(Lintmat,span):
    fint=np.zeros([18])
    c=0
    for ii in range(3):
        i=span-p+ii
        for j in range(6):
            fint[c]=Lintmat[j,i]
            c=c+1
    return fint

def intmat2(PsX1,PsY1,PsZ1,PsRhos1,pspan,dts):
    pp=np.zeros([3,4])
    fint2=np.zeros([21,18])
    #print(p,pspan-p)#,PsRhos[pspan-p,0])
    pp[0,0]=PsRhos1[pspan-p+0,0]
    pp[0,1]=PsRhos1[pspan-p+0,1]
    pp[0,2]=PsRhos1[pspan-p+0,2]
    pp[0,3]=PsRhos1[pspan-p+0,3]
    pp[1,0]=PsRhos1[pspan-p+1,0]
    pp[1,1]=PsRhos1[pspan-p+1,1]
    pp[1,2]=PsRhos1[pspan-p+1,2]
    pp[1,3]=PsRhos1[pspan-p+1,3]
    pp[2,0]=PsRhos1[pspan-p+2,0]
    pp[2,1]=PsRhos1[pspan-p+2,1]
    pp[2,2]=PsRhos1[pspan-p+2,2]
    pp[2,3]=PsRhos1[pspan-p+2,3]
    dt=np.zeros([3,3])
    dt[:,:]=1e-10
    for i in range(3):
        fint2[i*7:i*7+7,i*6:i*6+6]= np.array([
                                    [ 1, 0, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         0],
                                    [ 0, 1, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         0],
                                    [ 0, 0, 1,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         0],
                                    [ 0, 0, 0, (pp[i,1]*np.sin(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,0])**2)/((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**3 - (pp[i,1]*np.sin(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2))/((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) - (dt[i,0]*pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) - (pp[i,1]*np.cos(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,0])**2)/(2*((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**2) - (pp[i,2]*np.cos(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,0])*(dt[i,1]))/(2*((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**2) - (pp[i,3]*np.cos(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,0])*(dt[i,2]))/(2*((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**2) + (pp[i,2]*np.sin(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,0])*(dt[i,1]))/((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**3 + (pp[i,3]*np.sin(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,0])*(dt[i,2]))/((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**3, (pp[i,2]*np.sin(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,1])**2)/((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**3 - (pp[i,2]*np.sin(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2))/((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) - (dt[i,1]*pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) - (pp[i,2]*np.cos(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,1])**2)/(2*((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**2) - (pp[i,1]*np.cos(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,0])*(dt[i,1]))/(2*((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**2) - (pp[i,3]*np.cos(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,1])*(dt[i,2]))/(2*((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**2) + (pp[i,1]*np.sin(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,0])*(dt[i,1]))/((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**3 + (pp[i,3]*np.sin(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,1])*(dt[i,2]))/((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**3, (pp[i,3]*np.sin(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,2])**2)/((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**3 - (pp[i,3]*np.sin(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2))/((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) - (dt[i,2]*pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) - (pp[i,3]*np.cos(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,2])**2)/(2*((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**2) - (pp[i,1]*np.cos(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,0])*(dt[i,2]))/(2*((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**2) - (pp[i,2]*np.cos(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,1])*(dt[i,2]))/(2*((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**2) + (pp[i,1]*np.sin(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,0])*(dt[i,2]))/((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**3 + (pp[i,2]*np.sin(((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))/2)*(dt[i,1])*(dt[i,2]))/((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2))**3],
                                    [ 0, 0, 0,                                                                                                                                                                               (pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2) - (dt[i,0]*pp[i,1]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) + (dt[i,0]**2*pp[i,0]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,0]**2*pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) + (dt[i,0]*dt[i,1]*pp[i,3]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,0]*dt[i,2]*pp[i,2]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,0]*dt[i,1]*pp[i,3]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) + (dt[i,0]*dt[i,2]*pp[i,2]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2),                                                                                                                                                                               (pp[i,3]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2) - (dt[i,1]*pp[i,1]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) + (dt[i,1]**2*pp[i,3]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,1]**2*pp[i,3]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) + (dt[i,0]*dt[i,1]*pp[i,0]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,1]*dt[i,2]*pp[i,2]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,0]*dt[i,1]*pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) + (dt[i,1]*dt[i,2]*pp[i,2]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2),                                                                                                                                                                               (dt[i,2]**2*pp[i,2]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) - (dt[i,2]*pp[i,1]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) - (dt[i,2]**2*pp[i,2]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (pp[i,2]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2) + (dt[i,0]*dt[i,2]*pp[i,0]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) + (dt[i,1]*dt[i,2]*pp[i,3]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,0]*dt[i,2]*pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) - (dt[i,1]*dt[i,2]*pp[i,3]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2)],
                                    [ 0, 0, 0,                                                                                                                                                                               (dt[i,0]**2*pp[i,3]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) - (dt[i,0]*pp[i,2]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) - (dt[i,0]**2*pp[i,3]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (pp[i,3]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2) + (dt[i,0]*dt[i,1]*pp[i,0]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) + (dt[i,0]*dt[i,2]*pp[i,1]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,0]*dt[i,1]*pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) - (dt[i,0]*dt[i,2]*pp[i,1]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2),                                                                                                                                                                               (pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2) - (dt[i,1]*pp[i,2]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) + (dt[i,1]**2*pp[i,0]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,1]**2*pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) - (dt[i,0]*dt[i,1]*pp[i,3]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) + (dt[i,1]*dt[i,2]*pp[i,1]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) + (dt[i,0]*dt[i,1]*pp[i,3]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) - (dt[i,1]*dt[i,2]*pp[i,1]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2),                                                                                                                                                                               (pp[i,1]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2) - (dt[i,2]*pp[i,2]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) + (dt[i,2]**2*pp[i,1]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,2]**2*pp[i,1]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) + (dt[i,1]*dt[i,2]*pp[i,0]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,0]*dt[i,2]*pp[i,3]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,1]*dt[i,2]*pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) + (dt[i,0]*dt[i,2]*pp[i,3]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2)],
                                    [ 0, 0, 0,                                                                                                                                                                               (pp[i,2]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2) - (dt[i,0]*pp[i,3]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) + (dt[i,0]**2*pp[i,2]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,0]**2*pp[i,2]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) - (dt[i,0]*dt[i,1]*pp[i,1]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) + (dt[i,0]*dt[i,2]*pp[i,0]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) + (dt[i,0]*dt[i,1]*pp[i,1]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) - (dt[i,0]*dt[i,2]*pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2),                                                                                                                                                                               (dt[i,1]**2*pp[i,1]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) - (dt[i,1]*pp[i,3]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) - (dt[i,1]**2*pp[i,1]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (pp[i,1]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2) + (dt[i,0]*dt[i,1]*pp[i,2]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) + (dt[i,1]*dt[i,2]*pp[i,0]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,0]*dt[i,1]*pp[i,2]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) - (dt[i,1]*dt[i,2]*pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2),                                                                                                                                                                               (pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2) - (dt[i,2]*pp[i,3]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)) + (dt[i,2]**2*pp[i,0]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,2]**2*pp[i,0]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) + (dt[i,0]*dt[i,2]*pp[i,2]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,1]*dt[i,2]*pp[i,1]*np.cos((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(2*(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)) - (dt[i,0]*dt[i,2]*pp[i,2]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2) + (dt[i,1]*dt[i,2]*pp[i,1]*np.sin((dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(1/2)/2))/(dt[i,0]**2 + dt[i,1]**2 + dt[i,2]**2)**(3/2)]
                                    ])
    return fint2

def stiff_eulerbeam(E,G,Iy,Iz,J,A,beam_len):
    L=beam_len
    Kstiff=np.zeros([12,12],dtype=float)
    Kstiff=np.array([[E*A/L,0,0,0,0,0,-E*A/L,0,0,0,0,0],
                     [0,12*E*Iz/(L**3),0,0,0,6*E*Iz/(L**2),0,-12*E*Iz/(L**3),0,0,0,6*E*Iz/(L**2)],
                     [0,0,12*E*Iy/(L**3),0,-6*E*Iy/(L**2),0,0,0,-12*E*Iy/(L**3),0,-6*E*Iy/(L**2),0],
                     [0,0,0,G*J/L,0,0,0,0,0,-G*J/L,0,0],
                     [0,0,0,0,4*E*Iy/L,0,0,0,6*E*Iy/L**2,0,2*E*Iy/L,0],
                     [0,0,0,0,0,4*E*Iz/L,0,-6*E*Iz/L**2,0,0,0,2*E*Iy/L],
                     [0,0,0,0,0,0,E*A/L,0,0,0,0,0],
                     [0,0,0,0,0,0,0,12*E*Iz/(L**3),12*E*Iy/(L**3),0,0,-6*E*Iz/(L**2)],
                     [0,0,0,0,0,0,0,0,12*E*Iy/(L**3),0,6*E*Iy/(L**2),0],
                     [0,0,0,0,0,0,0,0,0,G*J/L,0,0],
                     [0,0,0,0,0,0,0,0,0,0,4*E*Iy/L,0],
                     [0,0,0,0,0,0,0,0,0,0,0,4*E*Iz/L]])
    print(Kstiff.shape)
    Kstiff=symmetrize(Kstiff)
    assert np.all(Kstiff == Kstiff.T)
    print(Kstiff.shape)
    KKstiff=np.zeros([6*(n+1),6*(n+1)])
    for i in range(n):
        KKstiff[6*i:6*(i+2),6*i:6*(i+2)]+=Kstiff
    return KKstiff

def Update_qs(PsXn,PsYn,PsZn,PsRhosn,DOFv):
    PsXnew=np.zeros_like(PsXn)
    PsYnew=np.zeros_like(PsXn)
    PsZnew=np.zeros_like(PsXn)
    PsRhosnew=np.zeros_like(PsRhosn)
    for i in range(n+1):
        dofv=DOFv[i*6:(i+1)*6]    
        tetas=dofv[3:6]
        PsXnew[i]=PsXn[i]+dofv[0]
        PsYnew[i]=PsYn[i]+dofv[1]
        PsZnew[i]=PsZn[i]+dofv[2]
        
        if (modv(tetas)==0):
            PsRhosnew[i]=PsRhosn[i]
        else:
            dt=qpure(tetas)
            drho=exps(dt)
            PsRhosnew[i]=qmul(PsRhosn[i],drho)[0,:]
        
    return PsXnew, PsYnew, PsZnew, PsRhosnew
    
# %% HALTON
def primes_from_2_to(n):
    """Prime number from 2 to n.
    From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
    :param int n: sup bound with ``n >= 6``.
    :return: primes in 2 <= p < n.
    :rtype: list
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def van_der_corput(n_sample, base=2):
    """Van der Corput sequence.
    :param int n_sample: number of element of the sequence.
    :param int base: base of the sequence.
    :return: sequence of Van der Corput.
    :rtype: list (n_samples,)
    """
    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence


def halton(dim, n_sample):
    """Halton sequence.
    :param int dim: dimension
    :param int n_sample: number of samples.
    :return: sequence of Halton.
    :rtype: array_like (n_samples, n_features)
    """
    big_number = 10
    while 'Not enought primes':
        base = primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
    sample = [van_der_corput(n_sample + 1, dim) for dim in base]
    sample = np.stack(sample, axis=-1)[1:]

    return sample

#########################

def halton_perturb(PsXn,PsYn,PsZn,PsRhosn,halps,span,haloop):
    counter=0
    Cs=np.zeros([p+1])
    psx=np.zeros_like(PsXn)
    psy=np.zeros_like(PsXn)
    psz=np.zeros_like(PsXn)
    psx[:]=PsXn[:]
    psy[:]=PsYn[:]
    psz[:]=PsZn[:]
    psrhos=np.zeros_like(PsRhosn)
    psrhos[:,:]=PsRhosn[:,:]
    pertqs=np.zeros([21])
    for ii in range(p+1):
        i=span-p+ii
        Cs[ii]=i
        ps=halps[haloop,:]
       # print(ps)
        psx[i]=PsXn[i]+ps[counter]
        pertqs[counter]=psx[i]
        counter+=1
        psy[i]=PsYn[i]+ps[counter]
        pertqs[counter]=psy[i]
        counter+=1    
        psz[i]=PsZn[i]+ps[counter]
        pertqs[counter]=psz[i]
        counter+=1   
        for pp in range(4):
            psrhos[i,pp]=PsRhosn[i,pp]+ps[counter]
            pertqs[counter]=psrhos[i,pp]
            counter+=1
    return psx,psy,psz,psrhos,pertqs,Cs
#print(van_der_corput(10))
# [0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625, 0.5625]
#print(halton(2, 5))
# [[ 0.5         0.33333333]
#  [ 0.25        0.66666667]
#  [ 0.75        0.11111111]
#  [ 0.125       0.44444444]
#  [ 0.625       0.77777778]]




def Diff_perturb(PsXn,PsYn,PsZn,PsRhosn,CPlist,pnum,pspan,pert,hscale):
    psx=np.zeros_like(PsXn)
    psy=np.zeros_like(PsYn)
    psz=np.zeros_like(PsZn)
    psx[:]=PsXn[:]
    psy[:]=PsYn[:]
    psz[:]=PsZn[:]
    psrhos=np.zeros_like(PsRhosn)
    psrhos[:,:]=PsRhosn[:,:]
    pertq=0
    if pert==-1:
        return psx,psy,psz,psrhos,pertq
    
    L=CPlist[pnum]
    
    cp= int(pert/7)
    v=  int(pert%7)
    
    if v==0:
        pertq=hscale
        psx[L[cp]]=hscale+PsXn[L[cp]]
    elif v==1:
        pertq=hscale
        psy[L[cp]]=hscale+PsYn[L[cp]]
        #print(psz[L[cp]],hscale,PsZn[L[cp]])
    elif v==2:
        pertq=hscale
        psz[L[cp]]=hscale+PsZn[L[cp]]
        #print(psz[L[cp]],hscale,PsZn[L[cp]])
    else:
        pertq=hscale
        psrhos[L[cp],v-3]=hscale+ PsRhosn[L[cp],v-3]
        
    return psx,psy,psz,psrhos,pertq

##$$$$$$$$$$$$$$$$
    
def Prep_Reg(Regin,Regout,nreg,nreg20):
    nreg1=nreg-nreg20
    Ins=np.zeros([nreg1,21])
    Outs=np.zeros([nreg1,18])
    for i in range(nreg1):
        cin=Regin[i]
        cout=Regout[i]
        for j in range(21):
            Ins[i,j]=cin[j]
        for j in range(18):
            Outs[i,j]=cout[j]
    return Ins,Outs

def Regression_coeffs(ins,outs):
    coeffs=np.zeros([18,21])
    for i in range(18):
        X=ins
        Y=outs[:,i]
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)
        coeffs[i,:]=regr.coef_
    return coeffs

def Regression_Test(coeffs,Regtesti,Regtesto,nreg20,fint,pnum):
    for k in range(nreg20):
        Y=coeffs @ Regtesti[k]
        Y0=Regtesto[k]
        MSE=mse(Y0,Y)
        #print(Y0,Y)
        print(MSE,k,pnum)
    return 0
        
# %% MAIN ROUTINE
#######################################
#######################################
#######################################
# Main CODE
file1 = open("MyFile.txt","w")
E=1e9
G=0.2e9
A=3.14
Iy=A/4
Iz=A/4
ky=5/6
kz=5/6
J=3.14*2**4/32
nknots=6
U=[0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6]
p=2
us=np.linspace(0,6,50)
m=len(U)-1
n=m-p-1
nth=2
nstep=1
tag= 0# 0 straight, rest curve
scale=1

#PsX=[0,0.5,1.5,3,4.5,5.5,6]
#PsY=[0,1,2,2.5,2,1,0]
#PsZ=[0,0,0,0,0,0,0]
PsX1=np.array([0.,0.,0.80385,2.19615,3.80385,5.19615,6,6])
PsY1=np.array([0.,0.80385,2.19615,3,3,2.19615,0.80385,0.])
PsZ1=np.array([0.,0.,0.,0.,0.,0.,0.,0.])


if tag==0:
    PsX1=[0,0.5,1.25,2.25,3.5,4.75,5.5,6]
    PsY1=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    PsZ1=[0.,0.,0.,0.,0.,0.,0.,0.]
    beam_len=6
    
for i in range(n+1):
    PsX1[i]=PsX1[i]*scale
    PsY1[i]=PsY1[i]*scale
    PsZ1[i]=PsZ1[i]*scale
    
PsRhos1=np.zeros([n+1,4])
DOFold=np.zeros([6*(n+1)])
DOFs=np.zeros([6*(n+1)])
conn=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6]]
Gp=[-0.7746,0,0.7746]
nks=len(Gp)*nknots
wf=[5/9,8/9,5/9]

EPS0=np.zeros([nks,3])
KAP0=np.zeros([nks,3])
JUS0=np.zeros([nks,1])

Titer=1
#LOAD=[1e8,1e8]
#Load_pos=[19,25]
#Juss=np.zeros([nks,1])
LOAD=[-np.pi*E*Iz/beam_len/5]
Load_pos=[47]
for load_step in range(nstep):
    NRcheck=0
    NRiter=0
    nth=2
    r0s=np.zeros([nks,3])
    rd0s=np.zeros([nks,3])
    #Juss=np.zeros([nks,1])
    FF=np.zeros([1,6*(n+1)])
    FINT=np.zeros([1,6*(n+1)])
    for lli,ll in enumerate(Load_pos):    
        FF[0,ll]=(load_step+1)*LOAD[lli]/nstep
    #print(PsRhos1)
    #FF[0,47]=1e6
    FF=FF.T
    
    hscale=5e-3
    while NRcheck==0:
        
        if NRiter==0:
            PsRhos1=perturbq(PsRhos1,PsX1,PsY1,PsZ1,tag)  
        else:
            PsX1,PsY1,PsZ1,PsRhos1= Update_qs(PsX1,PsY1,PsZ1,PsRhos1,DOFs) 
            PsRhos1=normalq(PsRhos1)
        FINT=np.zeros([6*(n+1),1])

        
        
        count1=0
        countstore=0
        nreg=21
#        nreg20=int(0.2*nreg)
#        sec=0
        #halps=halton(21,nreg) #size [nreg,21]
        #TnT=np.zeros_like(halps,dtype=bool)
        #TnT[sec*nreg20:(sec+1)*nreg20,:]=True
        #halps=halps-0.5
        #halps=halps*hscale
        #halps2=halps[TnT].reshape(nreg20,21)
        #halps1=halps[~TnT].reshape(nreg-nreg20,21)
        KK=np.zeros([6*(n+1),6*(n+1)])
        
        CPlist,Spancon=Control_points(n,nknots,p,U)
        ####
        haloop=0
        
#        FF[0,19]=1e6
#        FF[0,25]=1e6
        
        bk = np.zeros(KK.shape[0], dtype=bool)
        bk[0:6] = True
       # bk[6*n:6*(n+1)]=True
        for i in range(n+1):
            bk[i*6+2]=True
            bk[i*6+3]=True
            bk[i*6+4]=True
        bu= ~bk
        DOFs=np.zeros([6*(n+1)])
        for pnum in range(nknots):
        #    Regin=[]
        #    Regout=[]
        #    cnum=[]
        #    Regtesto=[]
        #    Regtesti=[]
            dts=np.zeros([n+1,6])
            pert=-1
            
        
            pspan=FindSpan(n,p,pnum,U)
            PsX,PsY,PsZ,PsRhos,pertq=Diff_perturb(PsX1,PsY1,PsZ1,PsRhos1,CPlist,pnum,pspan,pert,hscale)
            #print(PsX,PsY,PsZ,PsRhos)
            kf1=np.zeros([18,21])
            Lintmat=np.zeros([6,n+1])
            for i in (Spancon[pnum]):    
                #span=i
                n1=conn[i][0]
                n2=conn[i][1]
                Jac1=(n2-n1)/2
        
                for gpn,gp in enumerate(Gp):
                    #SHAPE FUNS
                    wgp=wf[gpn]
                    u=(n1+n2)/2+(n2-n1)/2*gp
                    #print(u)
                    span=FindSpan(n,p,u,U)
                    Ns=BasisFuns(span,u,p,U)
                    Nds=DersBasisFuns(span,u,p,nth,U)
                    #to find r and r'
                    r0,rd0,Jus0=CurveInitParams(Ns,Nds,PsX,PsY,PsZ,p,span)
                    rhos=CurveRots(Ns,Nds,PsRhos,p,span)
                    if load_step==0 and NRiter==0:
                        Jus=Jus0
                        JUS0[i*(len(Gp))+gpn,0]=Jus0
                        print(i,gpn,Jus0)
                    else:
                        Jus=JUS0[i*(len(Gp))+gpn,0]
                    #print(rhos)
                    kappa=Curvatures(Ns,Nds,rhos,p,span,Jus)
                    epsilon=Strains(Ns,Nds,r0,rd0,rhos,p,span,Jus)
                    if load_step==0 and NRiter==0:
                        EPS0[i*(len(Gp))+gpn,:]=epsilon
                        KAP0[i*(len(Gp))+gpn,:]=kappa
                    
                    epsilon=epsilon-EPS0[i*(len(Gp))+gpn,:]
                    kappa=kappa-KAP0[i*(len(Gp))+gpn,:]
                    #print(epsilon)
#                    print('curvature:' ,kappa)
                    #print(rhos)
                    ns,ms=ABDmat(epsilon,kappa,E,A,G,J,Iy,Iz,ky,kz)
                    if NRiter==20:
                        print(ns,ms)
                    Lintmat,Fgp,flag=intmatrix(Ns,Nds,r0,rd0,rhos,p,span,Jus,Jac1,ns,ms,Lintmat,wgp)
                    if NRiter==20:
                        print(pnum,i,gpn,Fgp)
                    #ERROR CHECK
                    if flag==1:
                        print('\nCODE EXITED AT ITERATION  ',count1)
                        sys.exit()
                    count1+=1
            #print(Lintmat)        
            fint0=intstack(Lintmat,pspan)
            FINT[(pspan-p)*6:(pspan-p+3)*6,0]=fint0[:]
#            print(fint0,'______________________',pnum)
            for pert in range(nreg):
                pspan=FindSpan(n,p,pnum,U)
                PsX,PsY,PsZ,PsRhos,pertq=Diff_perturb(PsX1,PsY1,PsZ1,PsRhos1,CPlist,pnum,pspan,pert,hscale)
        #        print(PsX,PsY,PsZ,PsRhos)
                Lintmat=np.zeros([6,n+1])
                for i in (Spancon[pnum]):    
                    #span=i
                    n1=conn[i][0]
                    n2=conn[i][1]
                    Jac1=(n2-n1)/2
        
                    for gpn,gp in enumerate(Gp):
                        wgp=wf[gpn]
                        #SHAPE FUNS
                        u=(n1+n2)/2+(n2-n1)/2*gp
                        #print(u)
                        span=FindSpan(n,p,u,U)
                        Ns=BasisFuns(span,u,p,U)
                        Nds=DersBasisFuns(span,u,p,nth,U)
                        #to find r and r'
                        r0,rd0,Jus0=CurveInitParams(Ns,Nds,PsX,PsY,PsZ,p,span)
                        rhos=CurveRots(Ns,Nds,PsRhos,p,span)
                        Jus=JUS0[i*(len(Gp))+gpn,0]
                        kappa=Curvatures(Ns,Nds,rhos,p,span,Jus)
                        epsilon=Strains(Ns,Nds,r0,rd0,rhos,p,span,Jus)
                        
#                        if pert==0:
#                            print(rd0)
                        #print('curvature:' ,kappa[0,2])
                        #print(rhos)
                        epsilon=epsilon-EPS0[i*(len(Gp))+gpn,:]
                        kappa=kappa-KAP0[i*(len(Gp))+gpn,:]
                        ns,ms=ABDmat(epsilon,kappa,E,A,G,J,Iy,Iz,ky,kz)
#                        if pert==4:
#                            print('pert0',ns,ms,kappa)
                        Lintmat,Fgp,flag=intmatrix(Ns,Nds,r0,rd0,rhos,p,span,Jus,Jac1,ns,ms,Lintmat,wgp)
                        #ERROR CHECK
                        if flag==1:
                            print('\nCODE EXITED AT ITERATION  ',count1)
                            sys.exit()
                        count1+=1
                       
                fint=intstack(Lintmat,pspan)
                dfdq=(fint-fint0)/pertq
                if pert==-10:
                    print(Fgp)
#                    print(fint[0],fint0[0],pertq)
#                    print(dfdq)
                
                kf1[:,pert]=dfdq
        #        if haloop<(nreg-nreg20):
        #            Regout.append(fint)
        #        else:
        #            Regtesto.append(fint)
        #        print(cnum)
        #    print('fint',fint)
        #    Ins,Outs=Prep_Reg(Regin,Regout,nreg,nreg20)
        #    coeffs=  Regression_coeffs(Ins,Outs)
        #    Regression_Test(coeffs,Regtesti,Regtesto,nreg20,fint,pnum)
            kf2= intmat2(PsX1,PsY1,PsZ1,PsRhos1,pspan,dts)   
            
            Kspan= - kf1 @ kf2  
           # print('elem number' ,pnum,'\n',Kspan[:,3])
        #    print((pspan-p)*6,(pspan-p+3)*6,(pspan-p)*6,(pspan-p+3)*6)
            KK[(pspan-p)*6:(pspan-p+3)*6,(pspan-p)*6:(pspan-p+3)*6]+=Kspan[:,:]
            
        np.savetxt('Stiffness_Matrix.csv', KK, delimiter=',')
        #print(FINT)
#        if tag==0 and load_step==0 and NRiter==0:
#                KK=stiff_eulerbeam(E,G,Iy,Iz,J,A,beam_len)
        Ktangent=KK[bu,:][:,bu]
        ForceVect=FF[bu,:]
        
        
        
        #print(Ktangent[0,0])
        Kinv=np.linalg.inv(Ktangent)
        DOFold[:]=DOFs[:]+0.0
        DOFvect = Kinv @ ForceVect
        DOFs[bu]=DOFvect[:,0]
        #print(PsX1)
          
        #print(PsZ1)
        Dmax=np.max(DOFs)
        if NRiter==0:
            Dmax1=1e-10
            
        
        Posmax=np.argmax(DOFs)
        Dmaxchg=abs((Dmax/Dmax1)-1)
        print( Dmax,Dmax1,Dmaxchg)
        if Dmaxchg < 5e-5:    
            NRcheck=1
            print('converged at iter num ', NRiter)
        NRiter=NRiter+1
        if NRiter==Titer:
            NRcheck=1
            print ('iter num crossed ', Titer)
        Dmax1=Dmax+0.
        
    print('#########LOADSTEP NO. :  ', load_step)
print('\nruntime:')    
print(time()-startt)    





CX=[0]*(50)
CY=[0]*(50)
Ders=DersBasisFuns(4,2.5,p,nth,U)

for index,u in enumerate(us):
    span=FindSpan(n,p,u,U)
    #from sympy.abc import u
    Ns=BasisFuns(span,u,p,U)
    Ders=DersBasisFuns(span,u,p,nth,U)
    for i in range(p+1):
        #print(span,span-p+i)
        CX[index]=CX[index]+Ns[i]*PsX1[span-p+i]
        CY[index]=CY[index]+Ns[i]*PsY1[span-p+i]
    #print(i)
    #print(Ns)
    
plt.plot(CX,CY)   
plt.plot(PsX1,PsY1,linestyle='--',marker='o')
plt.axis([-0.5,6.5,-0.5,6.5])    
#
#print (Ders)

        #            r0s[count1,0]=r0[0]
        #            r0s[count1,1]=r0[1]
        #            r0s[count1,2]=r0[2]
        #            rd0s[count1,0]=rd0[0]
        #            rd0s[count1,1]=rd0[1]
        #            rd0s[count1,2]=rd0[2]    
        #            Juss[count1,0]=Jus 





