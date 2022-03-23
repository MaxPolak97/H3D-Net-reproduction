# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:59:04 2022

@author: Maxpo
"""

from h3ds.dataset import H3DS
import numpy as np
import os

# Load dataset and view configurations
h3ds_path = 'h3ds_v0.2'

file_list = []
for root,dirs,files in os.walk(h3ds_path):
    file_list.append(dirs)
scene_list = file_list[0]

h3ds = H3DS(path=h3ds_path)
#scene = '0cd3f3c0bc34a287'
#views_configs = h3ds.default_views_configs(scene_id=scene) # '3', '4', '8', '16' and '32'

# Different views
shots = ['3', '4', '8', '16', '32']

# iteration
#scan = 1
our_dataset = 'OWN_DATA'
if not os.path.isdir(our_dataset):
    os.makedirs(our_dataset)

list_dir = our_dataset + '/scan_list.txt'
scan_list = open(list_dir, 'w')
for scan, scene in enumerate(scene_list):
    scan_list.write("scan" + str(scan) + ":" + str(scene) + "\n")
    scan_list.flush()
    
for view_id in shots:
    # Create a directory to save the data
    view_folder = our_dataset + '/view' + view_id
    if not os.path.isdir(view_folder):
        os.makedirs(view_folder)

    for scan, scene in enumerate(scene_list):
        print('Scene ID:' + scene)
        print('Number of views:' + view_id)

        # Load data of given scene and view configuration
        _, images, masks, _, cameras_OWN = h3ds.load_scene(scene_id=scene, views_config_id=view_id, normalized=False)

        scan_folder = view_folder + '/scan' + str(scan)
        #our_dataset = 'OWN_DATA/'
        print('Output folder:' + scan_folder)

        # Creates a directory to save the data
        if not os.path.isdir(scan_folder):
            os.makedirs(scan_folder)
    
        # Creates a directory to save the data
        if not os.path.isdir(scan_folder + '/image'):
            os.makedirs(scan_folder + '/image')
    
        # Save images
        for n, image in enumerate(images):
            image.save(scan_folder + '/image/img_000' + str(n) + '.jpg')
        
        # Creates a directory to save the data
        if not os.path.isdir(scan_folder + '/mask'):
            os.makedirs(scan_folder + '/mask')
    
        # Save masks
        for n, mask in enumerate(masks):
            mask.save(scan_folder + '/mask/mask_000' + str(n) + '.jpg')
    
        # save camera as npz file
        np.savez(scan_folder + '/cameras.npz', **cameras_OWN)  # data is a dict here

        print('done')