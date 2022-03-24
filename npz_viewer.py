# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:04:01 2022

@author: Maxpo
"""

import numpy as np

# IDR format
#data = np.load('IDR/data/DTU/scan65/cameras.npz')
#data = np.load('IDR/data/H3D/view3/scan6/cameras.npz')
#data = np.load('h3')

#for keys in data.files:
    #print(keys)
    
#print(data['world_mat_0'])
#print(data['camera_mat_inv_0'])
#print(data['scale_mat_0'])

# H3D-Net format
data1 = np.load('h3ds_v0.2/3b5a2eb92a501d54/cameras.npz')
#print(data1.files)
for keys in data1.files:
    print(keys)

data2 = np.load('OWN_DATA/view3/scan6/cameras.npz')
#print(data2.files)
for keys in data2.files:
    print(keys)
#print(data['world_mat_0'])
#print(data['camera_mat_inv_0'])
#print(data['scale_mat_0'])