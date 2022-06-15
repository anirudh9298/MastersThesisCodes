# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 19:42:52 2021

@author: aniru
VTK FILE FORMAT
"""

import numpy as np
np.set_printoptions(linewidth=np.inf)

jobname = 'Wedge_Linear_1'
def vtk_output_format(jobname,Points,nPoints,Cells,nCells,nodes,eltype,Xout,StrainNode,StressNode,ptdat):
    
    
    outputfile = jobname + '_output.vtk'
    
    with open(outputfile,'w') as output:
        
        # write header
        output.write('# vtk DataFile Version 3.1\n')
        output.write('Wedge Element Output File\n')
        output.write('ASCII\n')
        output.write('DATASET UNSTRUCTURED_GRID\n')
        
        #writing points
        # output.write('POINTS '+ str(nPoints) +' double\n')
        output.write(f"POINTS {nPoints} DOUBLE\n")
        
        for pt in Points:
            output.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
            
        #writing Elems
        output.write("\n")
        output.write(f"CELLS {nCells} {nCells*nodes + nCells}\n")
        for el in Cells:
            output.write(f"{nodes}")
            for i in el:
                output.write(f" {i}")
            output.write("\n")
        
        #writing Cell type
        output.write("\n")
        output.write(f"CELL_TYPES {nCells}\n")
        for i in range(nCells):
            output.write(f"{eltype}\n")
        
        #write Point data
        if ptdat:
            output.write("\n")
            output.write(f"POINT_DATA {nPoints}\n")
            #write Displacement Vector
            output.write(f"VECTORS displacement DOUBLE\n")
            for i in range(nPoints):
                output.write(f"{Xout[i*3]} {Xout[i*3+1]} {Xout[i*3+2]}\n")
        
        #write FIELDS WOOOHOO
        if ptdat:
            output.write("\n")
            output.write(f"FIELD FieldData {2}\n")
            #Strain Time
            output.write(f"Strain {6} {nPoints} DOUBLE\n")
            for i in range(nPoints):
                p = StrainNode[i,:]
                output.write(f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]}\n")
            #Stress Time
            output.write(f"Stress {6} {nPoints} DOUBLE\n")
            for i in range(nPoints):
                p = StressNode[i,:]
                output.write(f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]}\n")
                
            
        output.close()

def vtk_outputgauss_format(jobname,Points,nPoints,nodes,eltype,GPDisp,StrainGPs,StressGPs,ptdat):
    
    outputfile = jobname + '_Gauss_output.vtk'
    
    with open(outputfile,'w') as output:
        
        # write header
        output.write('# vtk DataFile Version 3.1\n')
        output.write('Wedge Element Gauss Point Output File\n')
        output.write('ASCII\n')
        output.write('DATASET POLYDATA\n')
        
        #writing points
        # output.write('POINTS '+ str(nPoints) +' double\n')
        output.write(f"POINTS {nPoints} DOUBLE\n")
        
        for pt in Points:
            output.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
            
        if ptdat:
            output.write("\n")
            output.write(f"POINT_DATA {nPoints}\n")
            #write Displacement Vector
            output.write(f"VECTORS displacement DOUBLE\n")
            for i in range(nPoints):
                output.write(f"{GPDisp[i,0]} {GPDisp[i,1]} {GPDisp[i,2]}\n")
        
        #write FIELDS WOOOHOO
        if ptdat:
            output.write("\n")
            output.write(f"FIELD FieldData {2}\n")
            #Strain Time
            output.write(f"Strain {6} {nPoints} DOUBLE\n")
            for i in range(nPoints):
                p = StrainGPs[i,:]
                output.write(f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]}\n")
            #Stress Time
            output.write(f"Stress {6} {nPoints} DOUBLE\n")
            for i in range(nPoints):
                p = StressGPs[i,:]
                output.write(f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]}\n")            
        
        output.close()




def Update_Position(Xout,Coords,Nel,Tnode,dofn,scale):
    newCoords = np.zeros_like(Coords)
    
    for i in range(Tnode):
        for j in range(dofn):
            # print(i,j)
            newCoords[i,j] = Coords[i,j] + scale*Xout[i*3 + j]
        
    
    return newCoords

def ElemDisp(Xout,Conn,nodes,dofn):
    eldisp = np.zeros([nodes*dofn])
    for i,nd in enumerate(Conn):
        eldisp[i*dofn:(i+1)*dofn] = Xout[nd*dofn:(nd+1)*dofn]
    
    return eldisp

    
    
      
        
