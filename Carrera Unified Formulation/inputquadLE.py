# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:15:25 2021

@author: aniru
"""

#import yaml
import numpy as np
import sys
import meshzoo
import ruamel.yaml
yaml = ruamel.yaml.YAML()
# yaml.preserve_quotes = True

#with open(r'data.yaml') as file:
#    datfile = yaml.load(file, Loader = yaml.FullLoader)
#
#print(datfile.keys())
#eltype, n_elems, n_nodes_per_elem, n_nodes, Coords, beam_len, \
#geomtype, cross_dims, Nexp, Mfun, dof_per_node, BCs,\
#CrossNode, CrossCon = datfile.values()

Xmin = -5.0e-03
Xmax = 5.0e-03
Ymin = -5.0e-03
Ymax = 5.0e-03
Nx = 2
Ny = 2
Nr = 2
Variant = "zigzag"
automesher = 2
points, cells = meshzoo.rectangle_quad(
        (Xmin, Ymin),
        (Xmax, Ymax),
        Nr)
coords = np.zeros([points.shape[0], 3])
coords[:,0:3:2] = points
with open('dataLE.yaml') as fp:
    data = yaml.load(fp)
data['cross_dims'] = [Xmax-Xmin,Ymax-Ymin]
data['CrossNode'] = coords.tolist()
data['CrossCon'] = cells.tolist()
if (automesher == 2):
    nelems = 20 #200
    n_nodes = nelems + 1
    beam_len = 1
    dof_per_node = coords.shape[0]*3
    newCoords = np.zeros([n_nodes,3])
    newBCs = np.ones([n_nodes,dof_per_node],dtype=int)
    newElemconn = np.zeros([nelems,2],dtype=int)
    newBCs[0,:] = 0 
    for i in range(n_nodes):
        newCoords[i,1] = i/nelems*beam_len
    for j in range(nelems):
        newElemconn[j,0] = j+1
        newElemconn[j,1] = j+2
    data['nelems'] = nelems
    data['n_nodes'] = n_nodes
    data['Coords'] = newCoords.tolist()
    data['Elemconn'] = newElemconn.tolist()
    data['beam_len'] = beam_len
    data['dof_per_node'] = dof_per_node
    data['BCs'] = newBCs.tolist()
elif (automesher == 3):
    nelems = 100 #200
    n_nodes = 2*nelems + 1
    beam_len = 1
    newCoords = np.zeros([n_nodes,3])
    newBCs = np.ones([n_nodes,9],dtype=int)
    newElemconn = np.zeros([nelems,3],dtype=int)
    newBCs[0,:] = 0 
    for i in range(n_nodes):
        newCoords[i,1] = i/(n_nodes-1)*beam_len
    for j in range(nelems):
        newElemconn[j,0] = 2*j+1
        newElemconn[j,1] = 2*j+3
        newElemconn[j,2] = 2*j+2
    data['nelems'] = nelems
    data['n_nodes'] = n_nodes
    data['Coords'] = newCoords.tolist()
    data['Elemconn'] = newElemconn.tolist()
    data['beam_len'] = beam_len
    data['dof_per_node'] = dof_per_node
    data['BCs'] = newBCs.tolist()


if (automesher == 2):
    with open('dataLE.yaml',"w") as fp:
        yaml.dump(data,fp)
if (automesher == 3):
    with open('dataLE.yaml',"w") as fp:
        yaml.dump(data,fp)