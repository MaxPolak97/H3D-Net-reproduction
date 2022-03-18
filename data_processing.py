# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:59:04 2022

@author: Maxpo
"""

from h3ds.dataset import H3DS
import numpy as np
import os

# Load dataset and view configurations
h3ds = H3DS(path='h3ds_v0.2')
scene = '0cd3f3c0bc34a287'
views_configs = h3ds.default_views_configs(scene_id=scene) # '3', '4', '8', '16' and '32'

# Different views
shots = ['3', '4', '8', '16', '32']

# iteration
scan = 1 

for view_id in shots:
    # Load data of given scene and view configuration
    _, images, masks, _, cameras_OWN = h3ds.load_scene(scene_id=scene, views_config_id=view_id, normalized=False)

    folder = 'scan' + str(scan)
    our_dataset = 'OWN_DATA/'

    # Creates a directory to save the data
    if not os.path.isdir(our_dataset + folder):
        os.mkdir(our_dataset + folder) 
    
    # Creates a directory to save the data
    if not os.path.isdir(our_dataset + folder + '/image'):
        os.mkdir(our_dataset + folder + '/image') 
    
    # Save images
    for n, image in enumerate(images):
        image.save(our_dataset + folder + '/image/img_000' + str(n) + '.jpg')
        
    # Creates a directory to save the data
    if not os.path.isdir(our_dataset + folder + '/mask'):
        os.mkdir(our_dataset + folder + '/mask') 
    
    # Save masks
    for n, mask in enumerate(masks):
        mask.save(our_dataset + folder + '/mask/mask_000' + str(n) + '.jpg')
    
    # save camera as npz file
    np.savez( our_dataset + '/' + folder + '/' + 'cameras.npz', **cameras_OWN)  # data is a dict here
    
    scan += 1
    

    
    